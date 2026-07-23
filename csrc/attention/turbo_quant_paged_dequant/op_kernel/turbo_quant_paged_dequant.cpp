/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace turbo_quant_detail {
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t FLOAT_BYTES = static_cast<uint32_t>(sizeof(float));
constexpr uint32_t INT32_BYTES = static_cast<uint32_t>(sizeof(int32_t));
constexpr uint32_t HALF_BYTES = 2U;
constexpr float NORM_EPSILON = 1.0e-16F;

__aicore__ constexpr inline uint32_t AlignUbBytes(uint32_t value) {
  return (value + DATA_BLOCK_BYTES - 1U) & ~(DATA_BLOCK_BYTES - 1U);
}
}  // namespace turbo_quant_detail

template <typename T, uint32_t HEAD_DIM, uint32_t KEY_BITS, uint32_t VALUE_BITS, bool NORM_CORRECTION>
class KernelTurboQuantPagedDequant {
 public:
  static_assert(KEY_BITS == 3U || KEY_BITS == 4U);
  static_assert(VALUE_BITS == 3U || VALUE_BITS == 4U);
  static_assert(HEAD_DIM >= 32U && HEAD_DIM <= 256U && (HEAD_DIM & (HEAD_DIM - 1U)) == 0U);
  static_assert((HEAD_DIM * KEY_BITS) % 8U == 0U);
  static_assert((HEAD_DIM * VALUE_BITS) % 8U == 0U);

  static constexpr uint32_t KEY_DATA_BYTES = HEAD_DIM * KEY_BITS / 8U;
  static constexpr uint32_t KEY_PACKED_SIZE = KEY_DATA_BYTES + turbo_quant_detail::HALF_BYTES;
  static constexpr uint32_t VALUE_DATA_BYTES = HEAD_DIM * VALUE_BITS / 8U;
  static constexpr uint32_t CENTROID_COUNT = 1U << KEY_BITS;

  __aicore__ inline KernelTurboQuantPagedDequant() = default;

  __aicore__ inline void Init(GM_ADDR kvCache, GM_ADDR blockTable, GM_ADDR seqLens, GM_ADDR pageTable,
                              GM_ADDR centroids, GM_ADDR key, GM_ADDR value,
                              const TurboQuantPagedDequantTilingData& tilingData) {
    batchSize_ = tilingData.batchSize;
    maxSeqLen_ = tilingData.maxSeqLen;
    maxPages_ = tilingData.maxPages;
    numBlocks_ = tilingData.numBlocks;
    blockSize_ = tilingData.blockSize;
    numKvHeads_ = tilingData.numKvHeads;
    slotSize_ = tilingData.slotSize;
    activePageCount_ = tilingData.activePageCount;
    totalTasks_ = tilingData.totalTasks;

    kvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    pageTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pageTable));
    centroidsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
    keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));

    // DataCopyPad writes complete 32-byte data blocks into local memory.
    pipe_.InitBuffer(slotBuffer_, turbo_quant_detail::AlignUbBytes(slotSize_));
    pipe_.InitBuffer(centroidBuffer_,
                     turbo_quant_detail::AlignUbBytes(CENTROID_COUNT * turbo_quant_detail::FLOAT_BYTES));
    pipe_.InitBuffer(indexBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * turbo_quant_detail::INT32_BYTES));
    pipe_.InitBuffer(keyFloatBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * turbo_quant_detail::FLOAT_BYTES));
    pipe_.InitBuffer(valueFloatBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * turbo_quant_detail::FLOAT_BYTES));
    pipe_.InitBuffer(squaredBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * turbo_quant_detail::FLOAT_BYTES));
    pipe_.InitBuffer(reduceBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * turbo_quant_detail::FLOAT_BYTES));
    pipe_.InitBuffer(normBuffer_, turbo_quant_detail::DATA_BLOCK_BYTES);
    pipe_.InitBuffer(keyOutputBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * static_cast<uint32_t>(sizeof(T))));
    pipe_.InitBuffer(valueOutputBuffer_, turbo_quant_detail::AlignUbBytes(HEAD_DIM * static_cast<uint32_t>(sizeof(T))));

    slotLocal_ = slotBuffer_.Get<uint8_t>();
    centroidLocal_ = centroidBuffer_.Get<float>();
    indexLocal_ = indexBuffer_.Get<int32_t>();
    keyFloatLocal_ = keyFloatBuffer_.Get<float>();
    valueFloatLocal_ = valueFloatBuffer_.Get<float>();
    squaredLocal_ = squaredBuffer_.Get<float>();
    reduceLocal_ = reduceBuffer_.Get<float>();
    normLocal_ = normBuffer_.Get<float>();
    keyOutputLocal_ = keyOutputBuffer_.Get<T>();
    valueOutputLocal_ = valueOutputBuffer_.Get<T>();
  }

  __aicore__ inline void Process() {
    LoadCentroids();
    const uint32_t coreIndex = GetBlockIdx();
    const uint32_t coreCount = GetBlockNum();
    for (uint32_t task = coreIndex; task < totalTasks_; task += coreCount) {
      ProcessPageHead(task);
    }
  }

 private:
  __aicore__ inline void LoadCentroids() {
    DataCopyExtParams params{
        1, CENTROID_COUNT * turbo_quant_detail::FLOAT_BYTES, 0, 0, 0,
    };
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(centroidLocal_, centroidsGm_, params, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ inline void ProcessPageHead(uint32_t task) {
    const uint32_t headIndex = task % numKvHeads_;
    const uint32_t activePageIndex = task / numKvHeads_;
    if (activePageIndex >= activePageCount_) {
      return;
    }
    const uint64_t pageTableOffset = static_cast<uint64_t>(activePageIndex) * 2U;
    const int32_t batchIndexValue = pageTableGm_.GetValue(pageTableOffset);
    const int32_t pageIndexValue = pageTableGm_.GetValue(pageTableOffset + 1U);
    if (batchIndexValue < 0 || pageIndexValue < 0 || static_cast<uint32_t>(batchIndexValue) >= batchSize_) {
      return;
    }
    const uint32_t batchIndex = static_cast<uint32_t>(batchIndexValue);
    const uint32_t pageIndex = static_cast<uint32_t>(pageIndexValue);
    if (pageIndex >= maxPages_) {
      return;
    }
    const uint64_t pageStartWide = static_cast<uint64_t>(pageIndex) * blockSize_;
    const int32_t sequenceLengthValue = seqLensGm_.GetValue(batchIndex);
    if (sequenceLengthValue <= 0 || pageStartWide >= maxSeqLen_ ||
        pageStartWide >= static_cast<uint32_t>(sequenceLengthValue)) {
      return;
    }
    const uint32_t pageStart = static_cast<uint32_t>(pageStartWide);

    const uint64_t blockTableOffset = static_cast<uint64_t>(batchIndex) * maxPages_ + pageIndex;
    const int32_t physicalBlockValue = blockTableGm_.GetValue(blockTableOffset);
    if (physicalBlockValue < 0 || static_cast<uint32_t>(physicalBlockValue) >= numBlocks_) {
      return;
    }
    const uint32_t sequenceLengthValueUnsigned = static_cast<uint32_t>(sequenceLengthValue);
    const uint32_t sequenceLength = sequenceLengthValueUnsigned < maxSeqLen_ ? sequenceLengthValueUnsigned : maxSeqLen_;
    const uint32_t remainingTokens = sequenceLength - pageStart;
    const uint32_t tokenCount = blockSize_ < remainingTokens ? blockSize_ : remainingTokens;
    for (uint32_t pageOffset = 0; pageOffset < tokenCount; ++pageOffset) {
      const uint64_t slotIndex =
          (static_cast<uint64_t>(physicalBlockValue) * blockSize_ + pageOffset) * numKvHeads_ + headIndex;
      const uint64_t outputIndex =
          ((static_cast<uint64_t>(batchIndex) * numKvHeads_ + headIndex) * maxSeqLen_ + pageStart + pageOffset) *
          HEAD_DIM;
      DequantizeSlot(slotIndex * slotSize_, outputIndex);
    }
  }

  template <uint32_t BITS, uint32_t INDEX_MULTIPLIER>
  __aicore__ inline void UnpackIndices(uint32_t byteBase) {
    if constexpr (BITS == 4U) {
      for (uint32_t byteIndex = 0; byteIndex < HEAD_DIM / 2U; ++byteIndex) {
        const uint32_t packed = static_cast<uint32_t>(slotLocal_.GetValue(byteBase + byteIndex));
        const uint32_t dimension = byteIndex * 2U;
        indexLocal_.SetValue(dimension, static_cast<int32_t>((packed & 0xFU) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 1U, static_cast<int32_t>((packed >> 4U) * INDEX_MULTIPLIER));
      }
    } else {
      for (uint32_t group = 0; group < HEAD_DIM / 8U; ++group) {
        const uint32_t byteOffset = byteBase + group * 3U;
        const uint32_t packed = static_cast<uint32_t>(slotLocal_.GetValue(byteOffset)) |
                                (static_cast<uint32_t>(slotLocal_.GetValue(byteOffset + 1U)) << 8U) |
                                (static_cast<uint32_t>(slotLocal_.GetValue(byteOffset + 2U)) << 16U);
        const uint32_t dimension = group * 8U;
        indexLocal_.SetValue(dimension, static_cast<int32_t>((packed & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 1U, static_cast<int32_t>(((packed >> 3U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 2U, static_cast<int32_t>(((packed >> 6U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 3U, static_cast<int32_t>(((packed >> 9U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 4U, static_cast<int32_t>(((packed >> 12U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 5U, static_cast<int32_t>(((packed >> 15U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 6U, static_cast<int32_t>(((packed >> 18U) & 0x7U) * INDEX_MULTIPLIER));
        indexLocal_.SetValue(dimension + 7U, static_cast<int32_t>(((packed >> 21U) & 0x7U) * INDEX_MULTIPLIER));
      }
    }
  }

  __aicore__ inline void DequantizeSlot(uint64_t cacheOffset, uint64_t outputOffset) {
    DataCopyExtParams cacheCopyParams{1, slotSize_, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> cachePadParams{false, 0, 0, 0};
    DataCopyPad(slotLocal_, kvCacheGm_[cacheOffset], cacheCopyParams, cachePadParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    LocalTensor<half> slotHalf = slotLocal_.ReinterpretCast<half>();
    const float cachedKeyScale =
        static_cast<float>(slotHalf.GetValue(KEY_DATA_BYTES / turbo_quant_detail::HALF_BYTES));
    const uint32_t valueBase = KEY_PACKED_SIZE;
    const uint32_t valueMetadataBase = valueBase + VALUE_DATA_BYTES;
    const float valueScale = static_cast<float>(slotHalf.GetValue(valueMetadataBase / turbo_quant_detail::HALF_BYTES));
    const float valueMinimum =
        static_cast<float>(slotHalf.GetValue(valueMetadataBase / turbo_quant_detail::HALF_BYTES + 1U));

    UnpackIndices<KEY_BITS, turbo_quant_detail::FLOAT_BYTES>(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Gather(keyFloatLocal_, centroidLocal_, indexLocal_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), HEAD_DIM);
    PipeBarrier<PIPE_V>();

    float keyScale = cachedKeyScale;
    if constexpr (!NORM_CORRECTION) {
      Mul(squaredLocal_, keyFloatLocal_, keyFloatLocal_, HEAD_DIM);
      PipeBarrier<PIPE_V>();
      ReduceSum(normLocal_, squaredLocal_, reduceLocal_, HEAD_DIM);
      PipeBarrier<PIPE_V>();
      Adds(normLocal_, normLocal_, turbo_quant_detail::NORM_EPSILON, 1);
      PipeBarrier<PIPE_V>();
      Sqrt(normLocal_, normLocal_, 1);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      keyScale *= normLocal_.GetValue(0);
    } else {
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Muls(keyFloatLocal_, keyFloatLocal_, keyScale, HEAD_DIM);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    UnpackIndices<VALUE_BITS, 1U>(valueBase);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Cast(valueFloatLocal_, indexLocal_, RoundMode::CAST_ROUND, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Muls(valueFloatLocal_, valueFloatLocal_, valueScale, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Adds(valueFloatLocal_, valueFloatLocal_, valueMinimum, HEAD_DIM);
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<T, bfloat16_t>::value) {
      Cast(keyOutputLocal_, keyFloatLocal_, RoundMode::CAST_RINT, HEAD_DIM);
      Cast(valueOutputLocal_, valueFloatLocal_, RoundMode::CAST_RINT, HEAD_DIM);
    } else {
      Cast(keyOutputLocal_, keyFloatLocal_, RoundMode::CAST_NONE, HEAD_DIM);
      Cast(valueOutputLocal_, valueFloatLocal_, RoundMode::CAST_NONE, HEAD_DIM);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams outputCopyParams{
        1, HEAD_DIM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0,
    };
    DataCopyPad(keyGm_[outputOffset], keyOutputLocal_, outputCopyParams);
    DataCopyPad(valueGm_[outputOffset], valueOutputLocal_, outputCopyParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> slotBuffer_;
  TBuf<TPosition::VECCALC> centroidBuffer_;
  TBuf<TPosition::VECCALC> indexBuffer_;
  TBuf<TPosition::VECCALC> keyFloatBuffer_;
  TBuf<TPosition::VECCALC> valueFloatBuffer_;
  TBuf<TPosition::VECCALC> squaredBuffer_;
  TBuf<TPosition::VECCALC> reduceBuffer_;
  TBuf<TPosition::VECCALC> normBuffer_;
  TBuf<TPosition::VECCALC> keyOutputBuffer_;
  TBuf<TPosition::VECCALC> valueOutputBuffer_;

  GlobalTensor<uint8_t> kvCacheGm_;
  GlobalTensor<int32_t> blockTableGm_;
  GlobalTensor<int32_t> seqLensGm_;
  GlobalTensor<int32_t> pageTableGm_;
  GlobalTensor<float> centroidsGm_;
  GlobalTensor<T> keyGm_;
  GlobalTensor<T> valueGm_;
  LocalTensor<uint8_t> slotLocal_;
  LocalTensor<float> centroidLocal_;
  LocalTensor<int32_t> indexLocal_;
  LocalTensor<float> keyFloatLocal_;
  LocalTensor<float> valueFloatLocal_;
  LocalTensor<float> squaredLocal_;
  LocalTensor<float> reduceLocal_;
  LocalTensor<float> normLocal_;
  LocalTensor<T> keyOutputLocal_;
  LocalTensor<T> valueOutputLocal_;

  uint32_t batchSize_ = 0;
  uint32_t maxSeqLen_ = 0;
  uint32_t maxPages_ = 0;
  uint32_t numBlocks_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t numKvHeads_ = 0;
  uint32_t slotSize_ = 0;
  uint32_t activePageCount_ = 0;
  uint32_t totalTasks_ = 0;
};

extern "C" __global__ __aicore__ void turbo_quant_paged_dequant(GM_ADDR query, GM_ADDR kvCache, GM_ADDR blockTable,
                                                                GM_ADDR seqLens, GM_ADDR pageTable, GM_ADDR centroids,
                                                                GM_ADDR key, GM_ADDR value, GM_ADDR workspace,
                                                                GM_ADDR tiling) {
  (void)query;
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tilingData, tiling);

  // CANN discovers relocatable kernel entries from literal TILING_KEY_IS
  // invocations before regular C++ macro expansion. Keep these branches
  // explicit so every host-generated key has a matching binary entry.
#define RUN_TURBOQUANT_VARIANT(HEAD_DIM, KEY_BITS, VALUE_BITS, NORM_CORRECTION)                  \
  KernelTurboQuantPagedDequant<DTYPE_QUERY, HEAD_DIM, KEY_BITS, VALUE_BITS, NORM_CORRECTION> op; \
  op.Init(kvCache, blockTable, seqLens, pageTable, centroids, key, value, tilingData);           \
  op.Process();                                                                                  \
  return

  if (TILING_KEY_IS(330032)) {
    RUN_TURBOQUANT_VARIANT(32, 3, 3, false);
  } else if (TILING_KEY_IS(330064)) {
    RUN_TURBOQUANT_VARIANT(64, 3, 3, false);
  } else if (TILING_KEY_IS(330128)) {
    RUN_TURBOQUANT_VARIANT(128, 3, 3, false);
  } else if (TILING_KEY_IS(330256)) {
    RUN_TURBOQUANT_VARIANT(256, 3, 3, false);
  } else if (TILING_KEY_IS(331032)) {
    RUN_TURBOQUANT_VARIANT(32, 3, 3, true);
  } else if (TILING_KEY_IS(331064)) {
    RUN_TURBOQUANT_VARIANT(64, 3, 3, true);
  } else if (TILING_KEY_IS(331128)) {
    RUN_TURBOQUANT_VARIANT(128, 3, 3, true);
  } else if (TILING_KEY_IS(331256)) {
    RUN_TURBOQUANT_VARIANT(256, 3, 3, true);
  } else if (TILING_KEY_IS(340032)) {
    RUN_TURBOQUANT_VARIANT(32, 3, 4, false);
  } else if (TILING_KEY_IS(340064)) {
    RUN_TURBOQUANT_VARIANT(64, 3, 4, false);
  } else if (TILING_KEY_IS(340128)) {
    RUN_TURBOQUANT_VARIANT(128, 3, 4, false);
  } else if (TILING_KEY_IS(340256)) {
    RUN_TURBOQUANT_VARIANT(256, 3, 4, false);
  } else if (TILING_KEY_IS(341032)) {
    RUN_TURBOQUANT_VARIANT(32, 3, 4, true);
  } else if (TILING_KEY_IS(341064)) {
    RUN_TURBOQUANT_VARIANT(64, 3, 4, true);
  } else if (TILING_KEY_IS(341128)) {
    RUN_TURBOQUANT_VARIANT(128, 3, 4, true);
  } else if (TILING_KEY_IS(341256)) {
    RUN_TURBOQUANT_VARIANT(256, 3, 4, true);
  } else if (TILING_KEY_IS(430032)) {
    RUN_TURBOQUANT_VARIANT(32, 4, 3, false);
  } else if (TILING_KEY_IS(430064)) {
    RUN_TURBOQUANT_VARIANT(64, 4, 3, false);
  } else if (TILING_KEY_IS(430128)) {
    RUN_TURBOQUANT_VARIANT(128, 4, 3, false);
  } else if (TILING_KEY_IS(430256)) {
    RUN_TURBOQUANT_VARIANT(256, 4, 3, false);
  } else if (TILING_KEY_IS(431032)) {
    RUN_TURBOQUANT_VARIANT(32, 4, 3, true);
  } else if (TILING_KEY_IS(431064)) {
    RUN_TURBOQUANT_VARIANT(64, 4, 3, true);
  } else if (TILING_KEY_IS(431128)) {
    RUN_TURBOQUANT_VARIANT(128, 4, 3, true);
  } else if (TILING_KEY_IS(431256)) {
    RUN_TURBOQUANT_VARIANT(256, 4, 3, true);
  } else if (TILING_KEY_IS(440032)) {
    RUN_TURBOQUANT_VARIANT(32, 4, 4, false);
  } else if (TILING_KEY_IS(440064)) {
    RUN_TURBOQUANT_VARIANT(64, 4, 4, false);
  } else if (TILING_KEY_IS(440128)) {
    RUN_TURBOQUANT_VARIANT(128, 4, 4, false);
  } else if (TILING_KEY_IS(440256)) {
    RUN_TURBOQUANT_VARIANT(256, 4, 4, false);
  } else if (TILING_KEY_IS(441032)) {
    RUN_TURBOQUANT_VARIANT(32, 4, 4, true);
  } else if (TILING_KEY_IS(441064)) {
    RUN_TURBOQUANT_VARIANT(64, 4, 4, true);
  } else if (TILING_KEY_IS(441128)) {
    RUN_TURBOQUANT_VARIANT(128, 4, 4, true);
  } else if (TILING_KEY_IS(441256)) {
    RUN_TURBOQUANT_VARIANT(256, 4, 4, true);
  }

#undef RUN_TURBOQUANT_VARIANT
}
