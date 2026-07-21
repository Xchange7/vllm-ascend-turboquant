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
}  // namespace turbo_quant_detail

template <typename T>
class KernelTurboQuantPagedDequant {
 public:
  __aicore__ inline KernelTurboQuantPagedDequant() = default;

  __aicore__ inline void Init(GM_ADDR kvCache, GM_ADDR blockTable, GM_ADDR seqLens, GM_ADDR centroids, GM_ADDR key,
                              GM_ADDR value, const TurboQuantPagedDequantTilingData& tilingData) {
    maxSeqLen_ = tilingData.maxSeqLen;
    maxPages_ = tilingData.maxPages;
    activePages_ = tilingData.activePages;
    numBlocks_ = tilingData.numBlocks;
    blockSize_ = tilingData.blockSize;
    numKvHeads_ = tilingData.numKvHeads;
    headDim_ = tilingData.headDim;
    slotSize_ = tilingData.slotSize;
    keyBits_ = tilingData.keyBits;
    keyDataBytes_ = tilingData.keyDataBytes;
    keyPackedSize_ = tilingData.keyPackedSize;
    valueBits_ = tilingData.valueBits;
    valueDataBytes_ = tilingData.valueDataBytes;
    centroidCount_ = tilingData.centroidCount;
    normCorrection_ = tilingData.normCorrection != 0;
    totalTasks_ = tilingData.totalTasks;

    kvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    centroidsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
    keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));

    // TPipe::InitBuffer pads byte lengths to a 32-byte data block.
    pipe_.InitBuffer(slotBuffer_, slotSize_);
    pipe_.InitBuffer(centroidBuffer_, centroidCount_ * turbo_quant_detail::FLOAT_BYTES);
    pipe_.InitBuffer(indexBuffer_, headDim_ * turbo_quant_detail::INT32_BYTES);
    pipe_.InitBuffer(keyFloatBuffer_, headDim_ * turbo_quant_detail::FLOAT_BYTES);
    pipe_.InitBuffer(valueFloatBuffer_, headDim_ * turbo_quant_detail::FLOAT_BYTES);
    pipe_.InitBuffer(squaredBuffer_, headDim_ * turbo_quant_detail::FLOAT_BYTES);
    pipe_.InitBuffer(reduceBuffer_, headDim_ * turbo_quant_detail::FLOAT_BYTES);
    pipe_.InitBuffer(normBuffer_, turbo_quant_detail::DATA_BLOCK_BYTES);
    pipe_.InitBuffer(keyOutputBuffer_, headDim_ * static_cast<uint32_t>(sizeof(T)));
    pipe_.InitBuffer(valueOutputBuffer_, headDim_ * static_cast<uint32_t>(sizeof(T)));

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
        1, centroidCount_ * turbo_quant_detail::FLOAT_BYTES, 0, 0, 0,
    };
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(centroidLocal_, centroidsGm_, params, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ inline void ProcessPageHead(uint32_t task) {
    const uint32_t headIndex = task % numKvHeads_;
    const uint32_t batchPage = task / numKvHeads_;
    const uint32_t pageIndex = batchPage % activePages_;
    const uint32_t batchIndex = batchPage / activePages_;
    const uint32_t pageStart = pageIndex * blockSize_;
    const int32_t sequenceLengthValue = seqLensGm_.GetValue(batchIndex);
    if (sequenceLengthValue <= 0 || pageStart >= maxSeqLen_ ||
        pageStart >= static_cast<uint32_t>(sequenceLengthValue)) {
      return;
    }

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
          headDim_;
      DequantizeSlot(slotIndex * slotSize_, outputIndex);
    }
  }

  __aicore__ inline uint32_t UnpackIndex(uint32_t dimension, uint32_t bits, uint32_t byteBase) {
    const uint32_t bitOffset = dimension * bits;
    const uint32_t byteIndex = bitOffset >> 3;
    const uint32_t shift = bitOffset & 7U;
    uint32_t packed = static_cast<uint32_t>(slotLocal_.GetValue(byteBase + byteIndex));
    if (shift + bits > 8U) {
      packed |= static_cast<uint32_t>(slotLocal_.GetValue(byteBase + byteIndex + 1U)) << 8U;
    }
    return (packed >> shift) & ((1U << bits) - 1U);
  }

  __aicore__ inline void DequantizeSlot(uint64_t cacheOffset, uint64_t outputOffset) {
    DataCopyExtParams cacheCopyParams{1, slotSize_, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> cachePadParams{false, 0, 0, 0};
    DataCopyPad(slotLocal_, kvCacheGm_[cacheOffset], cacheCopyParams, cachePadParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    LocalTensor<half> slotHalf = slotLocal_.ReinterpretCast<half>();
    const float originalNorm = static_cast<float>(slotHalf.GetValue(keyDataBytes_ / turbo_quant_detail::HALF_BYTES));
    const uint32_t valueBase = keyPackedSize_;
    const uint32_t valueMetadataBase = valueBase + valueDataBytes_;
    const float valueScale = static_cast<float>(slotHalf.GetValue(valueMetadataBase / turbo_quant_detail::HALF_BYTES));
    const float valueMinimum =
        static_cast<float>(slotHalf.GetValue(valueMetadataBase / turbo_quant_detail::HALF_BYTES + 1U));

    for (uint32_t dimension = 0; dimension < headDim_; ++dimension) {
      const uint32_t centroidIndex = UnpackIndex(dimension, keyBits_, 0);
      indexLocal_.SetValue(dimension, static_cast<int32_t>(centroidIndex * turbo_quant_detail::FLOAT_BYTES));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Gather(keyFloatLocal_, centroidLocal_, indexLocal_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), headDim_);
    PipeBarrier<PIPE_V>();

    float keyScale = originalNorm;
    if (normCorrection_) {
      Mul(squaredLocal_, keyFloatLocal_, keyFloatLocal_, headDim_);
      PipeBarrier<PIPE_V>();
      ReduceSum(normLocal_, squaredLocal_, reduceLocal_, headDim_);
      PipeBarrier<PIPE_V>();
      Adds(normLocal_, normLocal_, turbo_quant_detail::NORM_EPSILON, 1);
      PipeBarrier<PIPE_V>();
      Sqrt(normLocal_, normLocal_, 1);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      keyScale /= normLocal_.GetValue(0);
    } else {
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Muls(keyFloatLocal_, keyFloatLocal_, keyScale, headDim_);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t dimension = 0; dimension < headDim_; ++dimension) {
      indexLocal_.SetValue(dimension, static_cast<int32_t>(UnpackIndex(dimension, valueBits_, valueBase)));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Cast(valueFloatLocal_, indexLocal_, RoundMode::CAST_ROUND, headDim_);
    PipeBarrier<PIPE_V>();
    Muls(valueFloatLocal_, valueFloatLocal_, valueScale, headDim_);
    PipeBarrier<PIPE_V>();
    Adds(valueFloatLocal_, valueFloatLocal_, valueMinimum, headDim_);
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<T, bfloat16_t>::value) {
      Cast(keyOutputLocal_, keyFloatLocal_, RoundMode::CAST_RINT, headDim_);
      Cast(valueOutputLocal_, valueFloatLocal_, RoundMode::CAST_RINT, headDim_);
    } else {
      Cast(keyOutputLocal_, keyFloatLocal_, RoundMode::CAST_NONE, headDim_);
      Cast(valueOutputLocal_, valueFloatLocal_, RoundMode::CAST_NONE, headDim_);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams outputCopyParams{
        1, headDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0,
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

  uint32_t maxSeqLen_ = 0;
  uint32_t maxPages_ = 0;
  uint32_t activePages_ = 0;
  uint32_t numBlocks_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t numKvHeads_ = 0;
  uint32_t headDim_ = 0;
  uint32_t slotSize_ = 0;
  uint32_t keyBits_ = 0;
  uint32_t keyDataBytes_ = 0;
  uint32_t keyPackedSize_ = 0;
  uint32_t valueBits_ = 0;
  uint32_t valueDataBytes_ = 0;
  uint32_t centroidCount_ = 0;
  uint32_t totalTasks_ = 0;
  bool normCorrection_ = false;
};

extern "C" __global__ __aicore__ void turbo_quant_paged_dequant(GM_ADDR query, GM_ADDR kvCache, GM_ADDR blockTable,
                                                                GM_ADDR seqLens, GM_ADDR centroids, GM_ADDR key,
                                                                GM_ADDR value, GM_ADDR workspace, GM_ADDR tiling) {
  (void)query;
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tilingData, tiling);
  KernelTurboQuantPagedDequant<DTYPE_QUERY> op;
  op.Init(kvCache, blockTable, seqLens, centroids, key, value, tilingData);
  op.Process();
}
