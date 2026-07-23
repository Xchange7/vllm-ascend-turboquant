/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace turbo_quant_attention_detail {
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t GROUP_SIZE = 8U;
constexpr uint32_t CUBE_M = 16U;
constexpr uint32_t TILE_TOKENS = 384U;
constexpr uint32_t CUBE_TILE_TOKENS = 256U;
constexpr uint32_t ROWS_PER_AIV = GROUP_SIZE / 2U;
constexpr uint32_t TOKENS_PER_AIV = TILE_TOKENS / 2U;
constexpr uint32_t KEY_DATA_BYTES = HEAD_DIM / 2U;
constexpr uint32_t KEY_PACKED_SIZE = KEY_DATA_BYTES + 2U;
constexpr uint32_t VALUE_DATA_BYTES = HEAD_DIM / 2U;
constexpr uint32_t DATA_BLOCK_BYTES = 32U;
constexpr uint32_t CENTROID_COUNT = 16U;
constexpr uint32_t ELEMENT_BYTES = 2U;
constexpr uint32_t FLOAT_BYTES = 4U;
constexpr uint32_t INT32_BYTES = 4U;
constexpr uint32_t STATE_ELEMENTS = DATA_BLOCK_BYTES / FLOAT_BYTES;
constexpr uint32_t STATE_LINES_PER_TASK = 2U;
constexpr uint32_t STATE_STRIDE =
    STATE_LINES_PER_TASK * STATE_ELEMENTS;
constexpr uint32_t DEQUANT_METADATA_PLANES = 3U;
constexpr uint32_t SOFTMAX_TMP_BYTES = 32U * 1024U;
constexpr uint32_t PIPELINE_BUFFER_COUNT = 1U;
constexpr uint32_t SLOT_LOAD_BATCH = 64U;
constexpr float NORM_EPSILON = 1.0e-16F;
constexpr float NEGATIVE_INFINITY = -3.402823466e38F;

constexpr uint32_t KV_TILE_ELEMENTS = TILE_TOKENS * HEAD_DIM;
constexpr uint32_t KV_TILE_BYTES = KV_TILE_ELEMENTS * ELEMENT_BYTES;
constexpr uint32_t KEY_TILE_OFFSET = 0U;
constexpr uint32_t VALUE_TILE_OFFSET =
    KEY_TILE_OFFSET + PIPELINE_BUFFER_COUNT * KV_TILE_BYTES;
constexpr uint32_t KEY_SCALE_OFFSET =
    VALUE_TILE_OFFSET + PIPELINE_BUFFER_COUNT * KV_TILE_BYTES;
constexpr uint32_t VALUE_SCALE_OFFSET =
    KEY_SCALE_OFFSET + TILE_TOKENS * FLOAT_BYTES;
constexpr uint32_t VALUE_MINIMUM_OFFSET =
    VALUE_SCALE_OFFSET + TILE_TOKENS * FLOAT_BYTES;
constexpr uint32_t SCORES_OFFSET =
    KEY_SCALE_OFFSET +
    PIPELINE_BUFFER_COUNT * DEQUANT_METADATA_PLANES *
        TILE_TOKENS * FLOAT_BYTES;
constexpr uint32_t PROBABILITY_OFFSET =
    SCORES_OFFSET +
    PIPELINE_BUFFER_COUNT * CUBE_M * TILE_TOKENS * FLOAT_BYTES;
constexpr uint32_t TILE_OUTPUT_OFFSET =
    PROBABILITY_OFFSET +
    PIPELINE_BUFFER_COUNT * CUBE_M * TILE_TOKENS * ELEMENT_BYTES;

// Each pipeline edge owns two event IDs. Alternating them by tile prevents a
// producer from re-arming an event while the consumer is still retiring the
// preceding iteration.
constexpr uint16_t SYNC_DEQUANT_READY = 0U;
constexpr uint16_t SYNC_SCORES_READY = 2U;
constexpr uint16_t SYNC_PROBABILITY_READY = 4U;
constexpr uint16_t SYNC_TILE_OUTPUT_READY = 6U;
constexpr uint32_t SYNC_MODE2 = 2U;

constexpr SoftmaxConfig SOFTMAX_CONFIG = {
    false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr IsResetLoad3dConfig LOAD3D_CONFIG = {true, true};

__aicore__ constexpr inline uint32_t Align16(uint32_t value) {
  return (value + 15U) & ~15U;
}

__aicore__ constexpr inline uint32_t AlignUbBytes(uint32_t value) {
  return (value + DATA_BLOCK_BYTES - 1U) & ~(DATA_BLOCK_BYTES - 1U);
}
}  // namespace turbo_quant_attention_detail

template <typename T>
class KernelTurboQuantPagedAttentionVector {
 public:
  __aicore__ inline KernelTurboQuantPagedAttentionVector() = default;

  __aicore__ inline void Init(
      GM_ADDR query, GM_ADDR kvCache, GM_ADDR blockTable, GM_ADDR seqLens,
      GM_ADDR centroids, GM_ADDR attentionOut, __gm__ uint8_t* userWorkspace,
      const TurboQuantPagedAttentionTilingData& tilingData, TPipe* pipe) {
    (void)query;
    pipe_ = pipe;
    batchSize_ = tilingData.batchSize;
    numQueryHeads_ = tilingData.numQueryHeads;
    numKvHeads_ = tilingData.numKvHeads;
    numBlocks_ = tilingData.numBlocks;
    blockSize_ = tilingData.blockSize;
    maxPages_ = tilingData.maxPages;
    slotSize_ = tilingData.slotSize;
    numSplits_ = tilingData.numSplits;
    totalTasks_ = tilingData.totalTasks;
    usedCoreNum_ = tilingData.usedCoreNum;
    coreWorkspaceBytes_ = tilingData.coreWorkspaceBytes;
    partialAccumOffset_ = tilingData.partialAccumOffset;
    partialSumOffset_ = tilingData.partialSumOffset;
    partialMaxOffset_ = tilingData.partialMaxOffset;
    scale_ = tilingData.scale;

    const uint32_t vectorBlock = GetBlockIdx();
    aiCoreIndex_ = vectorBlock / 2U;
    subBlockIndex_ = GetSubBlockIdx();

    kvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(kvCache));
    blockTableGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(blockTable));
    seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    centroidsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(centroids));
    attentionOutGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(attentionOut));

    __gm__ uint8_t* coreWorkspace =
        userWorkspace + static_cast<uint64_t>(aiCoreIndex_) *
                            coreWorkspaceBytes_;
    keyTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::KEY_TILE_OFFSET));
    valueTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::VALUE_TILE_OFFSET));
    keyScaleTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace +
            turbo_quant_attention_detail::KEY_SCALE_OFFSET));
    valueScaleTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace +
            turbo_quant_attention_detail::VALUE_SCALE_OFFSET));
    valueMinimumTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace +
            turbo_quant_attention_detail::VALUE_MINIMUM_OFFSET));
    scoresGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace + turbo_quant_attention_detail::SCORES_OFFSET));
    probabilityGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::PROBABILITY_OFFSET));
    tileOutputGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace +
            turbo_quant_attention_detail::TILE_OUTPUT_OFFSET));
    partialAccumGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(userWorkspace +
                                        partialAccumOffset_));
    partialSumGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(userWorkspace +
                                        partialSumOffset_));
    partialMaxGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(userWorkspace +
                                        partialMaxOffset_));

    pipe_->InitBuffer(
        slotBuffer_,
        turbo_quant_attention_detail::SLOT_LOAD_BATCH *
            turbo_quant_attention_detail::AlignUbBytes(slotSize_));
    pipe_->InitBuffer(
        centroidBuffer_,
        turbo_quant_attention_detail::AlignUbBytes(
            turbo_quant_attention_detail::CENTROID_COUNT *
            turbo_quant_attention_detail::FLOAT_BYTES));
    pipe_->InitBuffer(
        indexBuffer_,
        turbo_quant_attention_detail::AlignUbBytes(
            turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::INT32_BYTES));
    pipe_->InitBuffer(
        interleaveIndexBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::INT32_BYTES);
    pipe_->InitBuffer(
        valueGatherIndexBuffer_,
        turbo_quant_attention_detail::KEY_DATA_BYTES / sizeof(uint16_t) *
            turbo_quant_attention_detail::INT32_BYTES);
    pipe_->InitBuffer(
        packedLowBuffer_,
        turbo_quant_attention_detail::KEY_DATA_BYTES);
    pipe_->InitBuffer(
        packedHighBuffer_,
        turbo_quant_attention_detail::KEY_DATA_BYTES);
    pipe_->InitBuffer(
        nibbleMaskBuffer_,
        turbo_quant_attention_detail::KEY_DATA_BYTES);
    pipe_->InitBuffer(
        nibbleHalfBuffer_,
        turbo_quant_attention_detail::HEAD_DIM * sizeof(half));
    pipe_->InitBuffer(
        keyFloatBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        valueFloatBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        squaredBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        reduceBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        keyOutputBuffer_,
        turbo_quant_attention_detail::TOKENS_PER_AIV *
            turbo_quant_attention_detail::HEAD_DIM * sizeof(T));
    pipe_->InitBuffer(
        valueOutputBuffer_,
        turbo_quant_attention_detail::TOKENS_PER_AIV *
            turbo_quant_attention_detail::HEAD_DIM * sizeof(T));
    pipe_->InitBuffer(
        metadataStagingBuffer_,
        turbo_quant_attention_detail::DEQUANT_METADATA_PLANES *
            turbo_quant_attention_detail::TOKENS_PER_AIV *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        metadataTileBuffer_,
        turbo_quant_attention_detail::DEQUANT_METADATA_PLANES *
            turbo_quant_attention_detail::TILE_TOKENS *
            turbo_quant_attention_detail::FLOAT_BYTES);

    constexpr uint32_t vectorRows =
        turbo_quant_attention_detail::ROWS_PER_AIV;
    constexpr uint32_t tileElements =
        vectorRows * turbo_quant_attention_detail::TILE_TOKENS;
    pipe_->InitBuffer(
        scoresBuffer_,
        tileElements * turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        probabilityBuffer_,
        tileElements * sizeof(T));
    pipe_->InitBuffer(
        tileOutputBuffer_,
        vectorRows * turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        accumulatorBuffer_,
        vectorRows * turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        reductionBuffer_,
        turbo_quant_attention_detail::HEAD_DIM *
            turbo_quant_attention_detail::FLOAT_BYTES);
    pipe_->InitBuffer(
        outputBuffer_,
        turbo_quant_attention_detail::HEAD_DIM * sizeof(T));
    pipe_->InitBuffer(softmaxTmpBuffer_,
                      turbo_quant_attention_detail::SOFTMAX_TMP_BYTES);
    pipe_->InitBuffer(runningSumBuffer_,
                      turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(runningMaxBuffer_,
                      turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(newSumBuffer_,
                      turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(newMaxBuffer_,
                      turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(rescaleBuffer_,
                      turbo_quant_attention_detail::PIPELINE_BUFFER_COUNT *
                          turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(weightBuffer_,
                      turbo_quant_attention_detail::DATA_BLOCK_BYTES);
    pipe_->InitBuffer(
        valueMinimumContributionBuffer_,
        turbo_quant_attention_detail::PIPELINE_BUFFER_COUNT *
            turbo_quant_attention_detail::DATA_BLOCK_BYTES);

    slotLocal_ = slotBuffer_.Get<uint8_t>();
    centroidLocal_ = centroidBuffer_.Get<float>();
    indexLocal_ = indexBuffer_.Get<int32_t>();
    interleaveIndexLocal_ =
        interleaveIndexBuffer_.Get<int32_t>();
    valueGatherIndexLocal_ =
        valueGatherIndexBuffer_.Get<int32_t>();
    packedLowLocal_ = packedLowBuffer_.Get<uint8_t>();
    packedHighLocal_ = packedHighBuffer_.Get<uint8_t>();
    nibbleMaskLocal_ = nibbleMaskBuffer_.Get<uint16_t>();
    nibbleHalfLocal_ = nibbleHalfBuffer_.Get<half>();
    keyFloatLocal_ = keyFloatBuffer_.Get<float>();
    valueFloatLocal_ = valueFloatBuffer_.Get<float>();
    squaredLocal_ = squaredBuffer_.Get<float>();
    reduceLocal_ = reduceBuffer_.Get<float>();
    keyOutputLocal_ = keyOutputBuffer_.Get<T>();
    valueOutputLocal_ = valueOutputBuffer_.Get<T>();
    keyScaleStagingLocal_ = metadataStagingBuffer_.Get<float>();
    valueScaleStagingLocal_ =
        keyScaleStagingLocal_[
            turbo_quant_attention_detail::TOKENS_PER_AIV];
    valueMinimumStagingLocal_ =
        valueScaleStagingLocal_[
            turbo_quant_attention_detail::TOKENS_PER_AIV];
    keyScaleTileLocal_ = metadataTileBuffer_.Get<float>();
    valueScaleTileLocal_ =
        keyScaleTileLocal_[turbo_quant_attention_detail::TILE_TOKENS];
    valueMinimumTileLocal_ =
        valueScaleTileLocal_[turbo_quant_attention_detail::TILE_TOKENS];
    scoresLocal_ = scoresBuffer_.Get<float>();
    probabilityLocal_ = probabilityBuffer_.Get<T>();
    tileOutputLocal_ = tileOutputBuffer_.Get<float>();
    accumulatorLocal_ = accumulatorBuffer_.Get<float>();
    reductionLocal_ = reductionBuffer_.Get<float>();
    outputLocal_ = outputBuffer_.Get<T>();
    softmaxTmpLocal_ = softmaxTmpBuffer_.Get<uint8_t>();
    runningSumLocal_ = runningSumBuffer_.Get<float>();
    runningMaxLocal_ = runningMaxBuffer_.Get<float>();
    newSumLocal_ = newSumBuffer_.Get<float>();
    newMaxLocal_ = newMaxBuffer_.Get<float>();
    rescaleLocal_ = rescaleBuffer_.Get<float>();
    weightLocal_ = weightBuffer_.Get<float>();
    valueMinimumContributionLocal_ =
        valueMinimumContributionBuffer_.Get<float>();
  }

  __aicore__ inline void Process() {
    LoadCentroids();
    InitializeDecodeConstants();
    for (uint32_t task = aiCoreIndex_; task < totalTasks_;
         task += usedCoreNum_) {
      ProcessTask(task);
    }

    // Every vector core has published its split states before the reduction.
    SyncAll();
    ReduceSplits();
  }

 private:
  __aicore__ inline void LoadCentroids() {
    DataCopyExtParams params{
        1,
        turbo_quant_attention_detail::CENTROID_COUNT *
            turbo_quant_attention_detail::FLOAT_BYTES,
        0,
        0,
        0,
    };
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(centroidLocal_, centroidsGm_, params, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ inline void DecodeTask(
      uint32_t task, uint32_t& batchIndex, uint32_t& kvHeadIndex,
      uint32_t& splitIndex) const {
    splitIndex = task % numSplits_;
    const uint32_t batchHead = task / numSplits_;
    kvHeadIndex = batchHead % numKvHeads_;
    batchIndex = batchHead / numKvHeads_;
  }

  __aicore__ inline void InitializeDecodeConstants() {
    for (uint32_t dimension = 0U;
         dimension < turbo_quant_attention_detail::HEAD_DIM;
         ++dimension) {
      const uint32_t source =
          (dimension >> 1U) +
          (dimension & 1U) *
              turbo_quant_attention_detail::KEY_DATA_BYTES;
      interleaveIndexLocal_.SetValue(
          dimension,
          static_cast<int32_t>(
              source *
              turbo_quant_attention_detail::FLOAT_BYTES));
    }
    constexpr uint32_t valueIntraBlockOffset =
        turbo_quant_attention_detail::KEY_PACKED_SIZE %
        turbo_quant_attention_detail::DATA_BLOCK_BYTES;
    for (uint32_t pair = 0U;
         pair < turbo_quant_attention_detail::KEY_DATA_BYTES /
                    sizeof(uint16_t);
         ++pair) {
      valueGatherIndexLocal_.SetValue(
          pair,
          static_cast<int32_t>(
              valueIntraBlockOffset + pair * sizeof(uint16_t)));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(nibbleMaskLocal_, static_cast<uint16_t>(0x0F0FU),
              turbo_quant_attention_detail::KEY_DATA_BYTES /
                  sizeof(uint16_t));
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline uint32_t SequenceLength(uint32_t batchIndex) {
    const int32_t value = seqLensGm_.GetValue(batchIndex);
    if (value <= 0) {
      return 0U;
    }
    const uint32_t capacity = maxPages_ * blockSize_;
    return static_cast<uint32_t>(value) < capacity
               ? static_cast<uint32_t>(value)
               : capacity;
  }

  __aicore__ inline void ProcessTask(uint32_t task) {
    uint32_t batchIndex = 0U;
    uint32_t kvHeadIndex = 0U;
    uint32_t splitIndex = 0U;
    DecodeTask(task, batchIndex, kvHeadIndex, splitIndex);
    const uint32_t sequenceLength = SequenceLength(batchIndex);
    const uint32_t splitStart =
        static_cast<uint64_t>(sequenceLength) * splitIndex / numSplits_;
    const uint32_t splitEnd =
        static_cast<uint64_t>(sequenceLength) * (splitIndex + 1U) /
        numSplits_;

    constexpr uint32_t rowCount =
        turbo_quant_attention_detail::ROWS_PER_AIV;
    constexpr uint32_t accumElements =
        rowCount * turbo_quant_attention_detail::HEAD_DIM;
    Duplicate(accumulatorLocal_, 0.0F, accumElements);
    Duplicate(runningSumLocal_, 0.0F,
              turbo_quant_attention_detail::STATE_ELEMENTS);
    Duplicate(runningMaxLocal_,
              turbo_quant_attention_detail::NEGATIVE_INFINITY,
              turbo_quant_attention_detail::STATE_ELEMENTS);
    PipeBarrier<PIPE_V>();

    uint32_t tileIndex = 0U;
    for (uint32_t tileStart = splitStart; tileStart < splitEnd;
         tileStart += turbo_quant_attention_detail::TILE_TOKENS,
                  ++tileIndex) {
      const uint32_t remaining = splitEnd - tileStart;
      const uint32_t tokenCount =
          remaining < turbo_quant_attention_detail::TILE_TOKENS
              ? remaining
              : turbo_quant_attention_detail::TILE_TOKENS;
      const uint16_t syncOffset =
          static_cast<uint16_t>(tileIndex & 1U);
      DequantizeTile(batchIndex, kvHeadIndex, tileStart,
                     tokenCount, 0U);
      CrossCoreSetFlag<
          turbo_quant_attention_detail::SYNC_MODE2, PIPE_MTE3>(
          turbo_quant_attention_detail::SYNC_DEQUANT_READY +
          syncOffset);

      CrossCoreWaitFlag(
          turbo_quant_attention_detail::SYNC_SCORES_READY +
          syncOffset);
      ComputeSoftmax(tokenCount, 0U);
      CrossCoreSetFlag<turbo_quant_attention_detail::SYNC_MODE2, PIPE_MTE3>(
          turbo_quant_attention_detail::SYNC_PROBABILITY_READY +
          syncOffset);

      CrossCoreWaitFlag(
          turbo_quant_attention_detail::SYNC_TILE_OUTPUT_READY +
          syncOffset);
      UpdateAccumulator(0U);
    }
    StorePartial(task);
  }

  __aicore__ inline void DequantizeTile(
      uint32_t batchIndex, uint32_t kvHeadIndex, uint32_t tileStart,
      uint32_t tokenCount, uint32_t bufferIndex) {
    const uint32_t localTokenStart =
        subBlockIndex_ * turbo_quant_attention_detail::TOKENS_PER_AIV;
    const uint32_t localTokenEnd =
        localTokenStart + turbo_quant_attention_detail::TOKENS_PER_AIV;
    const uint32_t end =
        tokenCount < localTokenEnd ? tokenCount : localTokenEnd;
    uint32_t tokenIndex = tileStart + localTokenStart;
    uint32_t pageIndex = tokenIndex / blockSize_;
    uint32_t pageOffset = tokenIndex % blockSize_;
    int32_t physicalBlock = blockTableGm_.GetValue(
        static_cast<uint64_t>(batchIndex) * maxPages_ + pageIndex);
    uint32_t tileRow = localTokenStart;
    while (tileRow < end) {
      const uint32_t localRow = tileRow - localTokenStart;
      const uint32_t localOutputOffset =
          localRow * turbo_quant_attention_detail::HEAD_DIM;
      if (physicalBlock < 0 ||
          static_cast<uint32_t>(physicalBlock) >= numBlocks_) {
        Duplicate(keyOutputLocal_[localOutputOffset],
                  static_cast<T>(0),
                  turbo_quant_attention_detail::HEAD_DIM);
        Duplicate(valueOutputLocal_[localOutputOffset],
                  static_cast<T>(0),
                  turbo_quant_attention_detail::HEAD_DIM);
        currentKeyScale_ = 0.0F;
        currentValueScale_ = 0.0F;
        currentValueMinimum_ = 0.0F;
        PipeBarrier<PIPE_V>();
        keyScaleStagingLocal_.SetValue(
            localRow, currentKeyScale_);
        valueScaleStagingLocal_.SetValue(
            localRow, currentValueScale_);
        valueMinimumStagingLocal_.SetValue(
            localRow, currentValueMinimum_);
        ++tileRow;
        ++pageOffset;
      } else {
        uint32_t loadCount = end - tileRow;
        const uint32_t pageRemaining = blockSize_ - pageOffset;
        if (loadCount > pageRemaining) {
          loadCount = pageRemaining;
        }
        if (loadCount >
            turbo_quant_attention_detail::SLOT_LOAD_BATCH) {
          loadCount =
              turbo_quant_attention_detail::SLOT_LOAD_BATCH;
        }
        const uint64_t slotIndex =
            (static_cast<uint64_t>(physicalBlock) * numKvHeads_ +
             kvHeadIndex) *
                blockSize_ +
            pageOffset;
        LoadSlotBatch(slotIndex * slotSize_, loadCount);
        const uint32_t slotUbStride =
            turbo_quant_attention_detail::AlignUbBytes(slotSize_);
        for (uint32_t loadIndex = 0U; loadIndex < loadCount;
             ++loadIndex) {
          const uint32_t batchLocalRow = localRow + loadIndex;
          DequantizeStagedSlot(
              loadIndex * slotUbStride,
              batchLocalRow *
                  turbo_quant_attention_detail::HEAD_DIM);
          keyScaleStagingLocal_.SetValue(
              batchLocalRow, currentKeyScale_);
          valueScaleStagingLocal_.SetValue(
              batchLocalRow, currentValueScale_);
          valueMinimumStagingLocal_.SetValue(
              batchLocalRow, currentValueMinimum_);
        }
        tileRow += loadCount;
        pageOffset += loadCount;
      }
      if (pageOffset == blockSize_ && tileRow < end) {
        ++pageIndex;
        pageOffset = 0U;
        physicalBlock = blockTableGm_.GetValue(
            static_cast<uint64_t>(batchIndex) * maxPages_ + pageIndex);
      }
    }
    if (end > localTokenStart) {
      StoreDequantizedTile(localTokenStart,
                           end - localTokenStart, bufferIndex);
      SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
      DataCopyExtParams metadataCopyParams{
          static_cast<uint16_t>(
              turbo_quant_attention_detail::DEQUANT_METADATA_PLANES),
          turbo_quant_attention_detail::TOKENS_PER_AIV *
              turbo_quant_attention_detail::FLOAT_BYTES,
          0,
          (turbo_quant_attention_detail::TILE_TOKENS -
           turbo_quant_attention_detail::TOKENS_PER_AIV) *
              turbo_quant_attention_detail::FLOAT_BYTES,
          0,
      };
      DataCopyPad(
          keyScaleTileGm_[
              bufferIndex *
                      turbo_quant_attention_detail::
                          DEQUANT_METADATA_PLANES *
                      turbo_quant_attention_detail::TILE_TOKENS +
              localTokenStart],
          keyScaleStagingLocal_, metadataCopyParams);
    }
  }

  __aicore__ inline void ExtractFourBitNibbles(uint32_t byteBase) {
    constexpr uint32_t packedCount =
        turbo_quant_attention_detail::KEY_DATA_BYTES;
    constexpr uint32_t packedPairs = packedCount / sizeof(uint16_t);
    LocalTensor<uint16_t> packedLow16 =
        packedLowLocal_.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> packedHigh16 =
        packedHighLocal_.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> packedSource16;
    if ((byteBase %
         turbo_quant_attention_detail::DATA_BLOCK_BYTES) == 0U) {
      packedSource16 =
          slotLocal_[byteBase].ReinterpretCast<uint16_t>();
    } else {
      const uint32_t alignedBase =
          byteBase -
          byteBase % turbo_quant_attention_detail::DATA_BLOCK_BYTES;
      Gather(packedLow16,
             slotLocal_[alignedBase].ReinterpretCast<uint16_t>(),
             valueGatherIndexLocal_.ReinterpretCast<uint32_t>(), 0U,
             packedPairs);
      PipeBarrier<PIPE_V>();
      packedSource16 = packedLow16;
    }
    // Shifting each uint16 pair mixes the next byte's low nibble into the
    // upper half of the first result byte. Masking with 0x0F0F removes that
    // carry and yields both high nibbles in their original byte lanes.
    ShiftRight(packedHigh16, packedSource16,
               static_cast<uint16_t>(4U), packedPairs);
    PipeBarrier<PIPE_V>();
    And(packedLow16, packedSource16, nibbleMaskLocal_,
        static_cast<int32_t>(packedPairs));
    And(packedHigh16, packedHigh16, nibbleMaskLocal_,
        static_cast<int32_t>(packedPairs));
    PipeBarrier<PIPE_V>();
    Cast(nibbleHalfLocal_, packedLowLocal_,
         RoundMode::CAST_NONE, packedCount);
    Cast(nibbleHalfLocal_[packedCount], packedHighLocal_,
         RoundMode::CAST_NONE, packedCount);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void DequantizeKeyCodes(uint32_t slotByteBase) {
    ExtractFourBitNibbles(slotByteBase);
    Cast(indexLocal_, nibbleHalfLocal_,
         RoundMode::CAST_ROUND,
         turbo_quant_attention_detail::HEAD_DIM);
    PipeBarrier<PIPE_V>();
    ShiftLeft(indexLocal_, indexLocal_, static_cast<int32_t>(2),
              turbo_quant_attention_detail::HEAD_DIM);
    PipeBarrier<PIPE_V>();
    // Packed low nibbles are the even dimensions and packed high nibbles
    // are the odd dimensions. Keep this nibble-major layout through both
    // Cube matmuls. The query rotation emits the same layout, and the final
    // output is interleaved only once per query head in ReduceSplits().
    Gather(keyFloatLocal_, centroidLocal_,
           indexLocal_.ReinterpretCast<uint32_t>(), 0U,
           turbo_quant_attention_detail::HEAD_DIM);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void DequantizeValueCodes(uint32_t byteBase) {
    ExtractFourBitNibbles(byteBase);
    Cast(valueFloatLocal_, nibbleHalfLocal_,
         RoundMode::CAST_NONE,
         turbo_quant_attention_detail::HEAD_DIM);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void LoadSlotBatch(
      uint64_t cacheOffset, uint32_t slotCount) {
    DataCopyExtParams cacheCopyParams{
        static_cast<uint16_t>(slotCount), slotSize_, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> cachePadParams{false, 0, 0, 0};
    DataCopyPad(slotLocal_, kvCacheGm_[cacheOffset], cacheCopyParams,
                cachePadParams);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  }

  __aicore__ inline void DequantizeStagedSlot(
      uint32_t slotByteBase, uint32_t localOutputOffset) {
    LocalTensor<half> slotHalf =
        slotLocal_[slotByteBase].ReinterpretCast<half>();
    const float cachedKeyScale = static_cast<float>(
        slotHalf.GetValue(
            turbo_quant_attention_detail::KEY_DATA_BYTES / 2U));
    constexpr uint32_t valueBase =
        turbo_quant_attention_detail::KEY_PACKED_SIZE;
    constexpr uint32_t valueMetadataBase =
        valueBase + turbo_quant_attention_detail::VALUE_DATA_BYTES;
    const float valueScale = static_cast<float>(
        slotHalf.GetValue(valueMetadataBase / 2U));
    const float valueMinimum = static_cast<float>(
        slotHalf.GetValue(valueMetadataBase / 2U + 1U));
    currentValueScale_ = valueScale;
    currentValueMinimum_ = valueMinimum;

    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    DequantizeKeyCodes(slotByteBase);
    // The value unpack reuses the packed-nibble UB buffers. A PIPE_V barrier
    // orders vector instructions but does not make those buffers safe for
    // scalar-controlled reuse; retire the key decode before starting V.
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    currentKeyScale_ = cachedKeyScale * scale_;

    DequantizeValueCodes(slotByteBase + valueBase);

    if constexpr (IsSameType<T, bfloat16_t>::value) {
      Cast(keyOutputLocal_[localOutputOffset], keyFloatLocal_,
           RoundMode::CAST_RINT,
           turbo_quant_attention_detail::HEAD_DIM);
      Cast(valueOutputLocal_[localOutputOffset], valueFloatLocal_,
           RoundMode::CAST_RINT,
           turbo_quant_attention_detail::HEAD_DIM);
    } else {
      Cast(keyOutputLocal_[localOutputOffset], keyFloatLocal_,
           RoundMode::CAST_NONE,
           turbo_quant_attention_detail::HEAD_DIM);
      Cast(valueOutputLocal_[localOutputOffset], valueFloatLocal_,
           RoundMode::CAST_NONE,
           turbo_quant_attention_detail::HEAD_DIM);
    }
  }

  __aicore__ inline void StoreDequantizedTile(
      uint32_t localTokenStart, uint32_t localTokenCount,
      uint32_t bufferIndex) {
    DataCopyExtParams outputCopyParams{
        1,
        static_cast<uint32_t>(
            localTokenCount *
            turbo_quant_attention_detail::HEAD_DIM * sizeof(T)),
        0,
        0,
        0,
    };
    const uint64_t outputOffset =
        (static_cast<uint64_t>(bufferIndex) *
             turbo_quant_attention_detail::TILE_TOKENS +
         localTokenStart) *
        turbo_quant_attention_detail::HEAD_DIM;
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(keyTileGm_[outputOffset], keyOutputLocal_,
                outputCopyParams);
    DataCopyPad(valueTileGm_[outputOffset], valueOutputLocal_,
                outputCopyParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

  __aicore__ inline void ComputeSoftmax(
      uint32_t tokenCount, uint32_t bufferIndex) {
    constexpr uint32_t rowCount =
        turbo_quant_attention_detail::ROWS_PER_AIV;
    const uint32_t rowStart = subBlockIndex_ * rowCount;
    const uint32_t tokenCountAligned =
        turbo_quant_attention_detail::Align16(tokenCount);
    DataCopyExtParams metadataParams{
        1,
        turbo_quant_attention_detail::DEQUANT_METADATA_PLANES *
            turbo_quant_attention_detail::TILE_TOKENS *
            turbo_quant_attention_detail::FLOAT_BYTES,
        0,
        0,
        0,
    };
    DataCopyPadExtParams<float> metadataPadParams{false, 0, 0, 0};
    DataCopyPad(
        keyScaleTileLocal_,
        keyScaleTileGm_[
            bufferIndex *
            turbo_quant_attention_detail::DEQUANT_METADATA_PLANES *
            turbo_quant_attention_detail::TILE_TOKENS],
        metadataParams, metadataPadParams);
    for (uint32_t row = 0; row < rowCount; ++row) {
      DataCopyExtParams params{
          1,
          tokenCountAligned *
              turbo_quant_attention_detail::FLOAT_BYTES,
          0,
          0,
          0,
      };
      DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
      DataCopyPad(
          scoresLocal_[row *
                       turbo_quant_attention_detail::TILE_TOKENS],
          scoresGm_[
              (bufferIndex * turbo_quant_attention_detail::CUBE_M +
               rowStart + row) *
              turbo_quant_attention_detail::TILE_TOKENS],
          params, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    for (uint32_t row = 0; row < rowCount; ++row) {
      Mul(scoresLocal_[
              row * turbo_quant_attention_detail::TILE_TOKENS],
          scoresLocal_[
              row * turbo_quant_attention_detail::TILE_TOKENS],
          keyScaleTileLocal_, tokenCount);
    }
    PipeBarrier<PIPE_V>();

    SoftMaxShapeInfo shape{
        rowCount,
        turbo_quant_attention_detail::TILE_TOKENS,
        rowCount,
        tokenCount,
    };
    SoftMaxTiling tiling = SoftMaxFlashV2TilingFunc(
        shape, sizeof(float), sizeof(float),
        softmaxTmpLocal_.GetSize(), true, false);
    SoftmaxFlashV2<
        float, true, true, false, false,
        turbo_quant_attention_detail::SOFTMAX_CONFIG>(
        scoresLocal_, newSumLocal_, newMaxLocal_, scoresLocal_,
        rescaleLocal_[
            bufferIndex *
            turbo_quant_attention_detail::STATE_ELEMENTS],
        runningSumLocal_, runningMaxLocal_,
        softmaxTmpLocal_, tiling, shape);
    PipeBarrier<PIPE_V>();

    // V = code * scale + minimum. Apply the per-token scale to the
    // probability before PV, and retain sum(probability * minimum) as one
    // scalar per query head. This avoids two 128-lane affine operations for
    // every decoded token.
    for (uint32_t row = 0; row < rowCount; ++row) {
      const uint32_t rowOffset =
          row * turbo_quant_attention_detail::TILE_TOKENS;
      Mul(squaredLocal_, scoresLocal_[rowOffset],
          valueMinimumTileLocal_, tokenCount);
      PipeBarrier<PIPE_V>();
      ReduceSum(valueMinimumContributionLocal_[
                    bufferIndex *
                        turbo_quant_attention_detail::STATE_ELEMENTS +
                    row],
                squaredLocal_, reduceLocal_, tokenCount);
      PipeBarrier<PIPE_V>();
      Mul(scoresLocal_[rowOffset], scoresLocal_[rowOffset],
          valueScaleTileLocal_, tokenCount);
    }
    PipeBarrier<PIPE_V>();

    // Explicitly zero padded K lanes. MM2 uses an aligned K dimension.
    Duplicate(
        probabilityLocal_, static_cast<T>(0),
        rowCount * turbo_quant_attention_detail::TILE_TOKENS);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < rowCount; ++row) {
      if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(
            probabilityLocal_[
                row * turbo_quant_attention_detail::TILE_TOKENS],
            scoresLocal_[
                row * turbo_quant_attention_detail::TILE_TOKENS],
            RoundMode::CAST_RINT, tokenCount);
      } else {
        Cast(
            probabilityLocal_[
                row * turbo_quant_attention_detail::TILE_TOKENS],
            scoresLocal_[
                row * turbo_quant_attention_detail::TILE_TOKENS],
            RoundMode::CAST_NONE, tokenCount);
      }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    for (uint32_t row = 0; row < rowCount; ++row) {
      DataCopyExtParams params{
          1,
          static_cast<uint32_t>(tokenCountAligned * sizeof(T)),
          0,
          0,
          0,
      };
      DataCopyPad(
          probabilityGm_[
              (bufferIndex * turbo_quant_attention_detail::CUBE_M +
               rowStart + row) *
              turbo_quant_attention_detail::TILE_TOKENS],
          probabilityLocal_[
              row * turbo_quant_attention_detail::TILE_TOKENS],
          params);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }

  __aicore__ inline void UpdateAccumulator(uint32_t bufferIndex) {
    constexpr uint32_t rowCount =
        turbo_quant_attention_detail::ROWS_PER_AIV;
    constexpr uint32_t rowElements =
        turbo_quant_attention_detail::HEAD_DIM;
    const uint32_t rowStart = subBlockIndex_ * rowCount;
    for (uint32_t row = 0; row < rowCount; ++row) {
      DataCopyExtParams params{
          1,
          rowElements * turbo_quant_attention_detail::FLOAT_BYTES,
          0,
          0,
          0,
      };
      DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
      DataCopyPad(
          tileOutputLocal_[row * rowElements],
          tileOutputGm_[
              (bufferIndex * turbo_quant_attention_detail::CUBE_M +
               rowStart + row) *
              rowElements],
          params, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    for (uint32_t row = 0; row < rowCount; ++row) {
      const uint32_t stateOffset =
          bufferIndex *
              turbo_quant_attention_detail::STATE_ELEMENTS +
          row;
      const float oldWeight =
          rescaleLocal_.GetValue(stateOffset);
      const float minimumContribution =
          valueMinimumContributionLocal_.GetValue(stateOffset);
      Muls(accumulatorLocal_[row * rowElements],
           accumulatorLocal_[row * rowElements], oldWeight,
           rowElements);
      Adds(tileOutputLocal_[row * rowElements],
           tileOutputLocal_[row * rowElements],
           minimumContribution, rowElements);
    }
    PipeBarrier<PIPE_V>();
    Add(accumulatorLocal_, accumulatorLocal_, tileOutputLocal_,
        rowCount * rowElements);
    PipeBarrier<PIPE_V>();
    DataCopy(runningSumLocal_, newSumLocal_,
             turbo_quant_attention_detail::STATE_ELEMENTS);
    DataCopy(runningMaxLocal_, newMaxLocal_,
             turbo_quant_attention_detail::STATE_ELEMENTS);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void StorePartial(uint32_t task) {
    constexpr uint32_t rowCount =
        turbo_quant_attention_detail::ROWS_PER_AIV;
    constexpr uint32_t rowElements =
        turbo_quant_attention_detail::HEAD_DIM;
    const uint32_t rowStart = subBlockIndex_ * rowCount;
    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);

    // Give each sibling AIV its own 32-byte state cache line. Besides avoiding
    // false sharing, this lets MTE3 publish all four row states in one aligned
    // transfer, so the later cross-core reduction observes coherent values.
    const uint64_t stateLineOffset =
        static_cast<uint64_t>(task) *
            turbo_quant_attention_detail::STATE_STRIDE +
        subBlockIndex_ *
            turbo_quant_attention_detail::STATE_ELEMENTS;
    DataCopy(partialSumGm_[stateLineOffset], runningSumLocal_,
             turbo_quant_attention_detail::STATE_ELEMENTS);
    DataCopy(partialMaxGm_[stateLineOffset], runningMaxLocal_,
             turbo_quant_attention_detail::STATE_ELEMENTS);

    for (uint32_t row = 0; row < rowCount; ++row) {
      const uint64_t accumRow =
          static_cast<uint64_t>(task) *
              turbo_quant_attention_detail::GROUP_SIZE +
          rowStart + row;
      DataCopyExtParams params{
          1,
          rowElements * turbo_quant_attention_detail::FLOAT_BYTES,
          0,
          0,
          0,
      };
      const uint64_t accumOffset =
          accumRow * turbo_quant_attention_detail::HEAD_DIM;
      DataCopyPad(partialAccumGm_[accumOffset],
                  accumulatorLocal_[row * rowElements], params);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
  }

  __aicore__ inline void ReduceSplits() {
    const uint32_t vectorCoreCount = usedCoreNum_ * 2U;
    const uint32_t vectorCoreIndex =
        aiCoreIndex_ * 2U + subBlockIndex_;
    const uint32_t totalQueryHeads = batchSize_ * numQueryHeads_;
    for (uint32_t flatQueryHead = vectorCoreIndex;
         flatQueryHead < totalQueryHeads;
         flatQueryHead += vectorCoreCount) {
      const uint32_t batchIndex = flatQueryHead / numQueryHeads_;
      const uint32_t queryHeadIndex =
          flatQueryHead % numQueryHeads_;
      const uint32_t kvHeadIndex =
          queryHeadIndex /
          turbo_quant_attention_detail::GROUP_SIZE;
      const uint32_t rowWithinGroup =
          queryHeadIndex %
          turbo_quant_attention_detail::GROUP_SIZE;
      const uint32_t taskBase =
          (batchIndex * numKvHeads_ + kvHeadIndex) * numSplits_;

      float globalMax = turbo_quant_attention_detail::NEGATIVE_INFINITY;
      for (uint32_t split = 0; split < numSplits_; ++split) {
        const uint64_t stateOffset =
            static_cast<uint64_t>(taskBase + split) *
                turbo_quant_attention_detail::STATE_STRIDE +
            rowWithinGroup /
                turbo_quant_attention_detail::ROWS_PER_AIV *
                turbo_quant_attention_detail::STATE_ELEMENTS +
            rowWithinGroup %
                turbo_quant_attention_detail::ROWS_PER_AIV;
        const float splitMax = partialMaxGm_.GetValue(stateOffset);
        globalMax = splitMax > globalMax ? splitMax : globalMax;
      }

      Duplicate(reductionLocal_, 0.0F,
                turbo_quant_attention_detail::HEAD_DIM);
      Duplicate(accumulatorLocal_, 0.0F,
                turbo_quant_attention_detail::HEAD_DIM);
      PipeBarrier<PIPE_V>();
      float totalSum = 0.0F;
      for (uint32_t split = 0; split < numSplits_; ++split) {
        const uint64_t stateOffset =
            static_cast<uint64_t>(taskBase + split) *
                turbo_quant_attention_detail::STATE_STRIDE +
            rowWithinGroup /
                turbo_quant_attention_detail::ROWS_PER_AIV *
                turbo_quant_attention_detail::STATE_ELEMENTS +
            rowWithinGroup %
                turbo_quant_attention_detail::ROWS_PER_AIV;
        const float splitMax = partialMaxGm_.GetValue(stateOffset);
        const float splitSum = partialSumGm_.GetValue(stateOffset);
        if (splitSum <= 0.0F) {
          continue;
        }
        weightLocal_.SetValue(0, splitMax - globalMax);
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
        Exp(weightLocal_, weightLocal_, 1U);
        SetFlag<HardEvent::V_S>(EVENT_ID4);
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        const float weight = weightLocal_.GetValue(0);
        totalSum += splitSum * weight;

        const uint64_t accumRow =
            static_cast<uint64_t>(taskBase + split) *
                turbo_quant_attention_detail::GROUP_SIZE +
            rowWithinGroup;
        const uint64_t accumOffset =
            accumRow * turbo_quant_attention_detail::HEAD_DIM;
        DataCopyExtParams params{
            1,
            turbo_quant_attention_detail::HEAD_DIM *
                turbo_quant_attention_detail::FLOAT_BYTES,
            0,
            0,
            0,
        };
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(reductionLocal_, partialAccumGm_[accumOffset],
                    params, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        Muls(reductionLocal_, reductionLocal_, weight,
             turbo_quant_attention_detail::HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Add(accumulatorLocal_, accumulatorLocal_, reductionLocal_,
            turbo_quant_attention_detail::HEAD_DIM);
        PipeBarrier<PIPE_V>();
      }

      if (totalSum > 0.0F) {
        Muls(accumulatorLocal_, accumulatorLocal_, 1.0F / totalSum,
             turbo_quant_attention_detail::HEAD_DIM);
      } else {
        Duplicate(accumulatorLocal_, 0.0F,
                  turbo_quant_attention_detail::HEAD_DIM);
      }
      PipeBarrier<PIPE_V>();
      Gather(reductionLocal_, accumulatorLocal_,
             interleaveIndexLocal_.ReinterpretCast<uint32_t>(), 0U,
             turbo_quant_attention_detail::HEAD_DIM);
      PipeBarrier<PIPE_V>();
      if constexpr (IsSameType<T, bfloat16_t>::value) {
        Cast(outputLocal_, reductionLocal_, RoundMode::CAST_RINT,
             turbo_quant_attention_detail::HEAD_DIM);
      } else {
        Cast(outputLocal_, reductionLocal_, RoundMode::CAST_NONE,
             turbo_quant_attention_detail::HEAD_DIM);
      }
      SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
      DataCopyExtParams outputParams{
          1,
          turbo_quant_attention_detail::HEAD_DIM * sizeof(T),
          0,
          0,
          0,
      };
      DataCopyPad(
          attentionOutGm_[
              static_cast<uint64_t>(flatQueryHead) *
              turbo_quant_attention_detail::HEAD_DIM],
          outputLocal_, outputParams);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    }
  }

  TPipe* pipe_ = nullptr;
  TBuf<TPosition::VECCALC> slotBuffer_;
  TBuf<TPosition::VECCALC> centroidBuffer_;
  TBuf<TPosition::VECCALC> indexBuffer_;
  TBuf<TPosition::VECCALC> interleaveIndexBuffer_;
  TBuf<TPosition::VECCALC> valueGatherIndexBuffer_;
  TBuf<TPosition::VECCALC> packedLowBuffer_;
  TBuf<TPosition::VECCALC> packedHighBuffer_;
  TBuf<TPosition::VECCALC> nibbleMaskBuffer_;
  TBuf<TPosition::VECCALC> nibbleHalfBuffer_;
  TBuf<TPosition::VECCALC> keyFloatBuffer_;
  TBuf<TPosition::VECCALC> valueFloatBuffer_;
  TBuf<TPosition::VECCALC> squaredBuffer_;
  TBuf<TPosition::VECCALC> reduceBuffer_;
  TBuf<TPosition::VECCALC> keyOutputBuffer_;
  TBuf<TPosition::VECCALC> valueOutputBuffer_;
  TBuf<TPosition::VECCALC> metadataStagingBuffer_;
  TBuf<TPosition::VECCALC> metadataTileBuffer_;
  TBuf<TPosition::VECCALC> scoresBuffer_;
  TBuf<TPosition::VECCALC> probabilityBuffer_;
  TBuf<TPosition::VECCALC> tileOutputBuffer_;
  TBuf<TPosition::VECCALC> accumulatorBuffer_;
  TBuf<TPosition::VECCALC> reductionBuffer_;
  TBuf<TPosition::VECCALC> outputBuffer_;
  TBuf<TPosition::VECCALC> softmaxTmpBuffer_;
  TBuf<TPosition::VECCALC> runningSumBuffer_;
  TBuf<TPosition::VECCALC> runningMaxBuffer_;
  TBuf<TPosition::VECCALC> newSumBuffer_;
  TBuf<TPosition::VECCALC> newMaxBuffer_;
  TBuf<TPosition::VECCALC> rescaleBuffer_;
  TBuf<TPosition::VECCALC> weightBuffer_;
  TBuf<TPosition::VECCALC> valueMinimumContributionBuffer_;

  GlobalTensor<uint8_t> kvCacheGm_;
  GlobalTensor<int32_t> blockTableGm_;
  GlobalTensor<int32_t> seqLensGm_;
  GlobalTensor<float> centroidsGm_;
  GlobalTensor<T> attentionOutGm_;
  GlobalTensor<T> keyTileGm_;
  GlobalTensor<T> valueTileGm_;
  GlobalTensor<float> keyScaleTileGm_;
  GlobalTensor<float> valueScaleTileGm_;
  GlobalTensor<float> valueMinimumTileGm_;
  GlobalTensor<float> scoresGm_;
  GlobalTensor<T> probabilityGm_;
  GlobalTensor<float> tileOutputGm_;
  GlobalTensor<float> partialAccumGm_;
  GlobalTensor<float> partialSumGm_;
  GlobalTensor<float> partialMaxGm_;

  LocalTensor<uint8_t> slotLocal_;
  LocalTensor<float> centroidLocal_;
  LocalTensor<int32_t> indexLocal_;
  LocalTensor<int32_t> interleaveIndexLocal_;
  LocalTensor<int32_t> valueGatherIndexLocal_;
  LocalTensor<uint8_t> packedLowLocal_;
  LocalTensor<uint8_t> packedHighLocal_;
  LocalTensor<uint16_t> nibbleMaskLocal_;
  LocalTensor<half> nibbleHalfLocal_;
  LocalTensor<float> keyFloatLocal_;
  LocalTensor<float> valueFloatLocal_;
  LocalTensor<float> squaredLocal_;
  LocalTensor<float> reduceLocal_;
  LocalTensor<T> keyOutputLocal_;
  LocalTensor<T> valueOutputLocal_;
  LocalTensor<float> keyScaleStagingLocal_;
  LocalTensor<float> valueScaleStagingLocal_;
  LocalTensor<float> valueMinimumStagingLocal_;
  LocalTensor<float> keyScaleTileLocal_;
  LocalTensor<float> valueScaleTileLocal_;
  LocalTensor<float> valueMinimumTileLocal_;
  LocalTensor<float> scoresLocal_;
  LocalTensor<T> probabilityLocal_;
  LocalTensor<float> tileOutputLocal_;
  LocalTensor<float> accumulatorLocal_;
  LocalTensor<float> reductionLocal_;
  LocalTensor<T> outputLocal_;
  LocalTensor<uint8_t> softmaxTmpLocal_;
  LocalTensor<float> runningSumLocal_;
  LocalTensor<float> runningMaxLocal_;
  LocalTensor<float> newSumLocal_;
  LocalTensor<float> newMaxLocal_;
  LocalTensor<float> rescaleLocal_;
  LocalTensor<float> weightLocal_;
  LocalTensor<float> valueMinimumContributionLocal_;

  uint32_t batchSize_ = 0U;
  uint32_t numQueryHeads_ = 0U;
  uint32_t numKvHeads_ = 0U;
  uint32_t numBlocks_ = 0U;
  uint32_t blockSize_ = 0U;
  uint32_t maxPages_ = 0U;
  uint32_t slotSize_ = 0U;
  uint32_t numSplits_ = 0U;
  uint32_t totalTasks_ = 0U;
  uint32_t usedCoreNum_ = 0U;
  uint32_t coreWorkspaceBytes_ = 0U;
  uint32_t aiCoreIndex_ = 0U;
  uint32_t subBlockIndex_ = 0U;
  uint64_t partialAccumOffset_ = 0U;
  uint64_t partialSumOffset_ = 0U;
  uint64_t partialMaxOffset_ = 0U;
  float scale_ = 1.0F;
  float currentKeyScale_ = 1.0F;
  float currentValueScale_ = 1.0F;
  float currentValueMinimum_ = 0.0F;
};

template <typename T>
class KernelTurboQuantPagedAttentionCube {
 public:
  __aicore__ inline KernelTurboQuantPagedAttentionCube() = default;

  __aicore__ inline void Init(
      GM_ADDR query, GM_ADDR seqLens, __gm__ uint8_t* userWorkspace,
      const TurboQuantPagedAttentionTilingData& tilingData, TPipe* pipe) {
    pipe_ = pipe;
    numQueryHeads_ = tilingData.numQueryHeads;
    numKvHeads_ = tilingData.numKvHeads;
    blockSize_ = tilingData.blockSize;
    maxPages_ = tilingData.maxPages;
    numSplits_ = tilingData.numSplits;
    totalTasks_ = tilingData.totalTasks;
    usedCoreNum_ = tilingData.usedCoreNum;
    coreWorkspaceBytes_ = tilingData.coreWorkspaceBytes;
    aiCoreIndex_ = GetBlockIdx();

    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
    seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLens));
    __gm__ uint8_t* coreWorkspace =
        userWorkspace + static_cast<uint64_t>(aiCoreIndex_) *
                            coreWorkspaceBytes_;
    keyTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::KEY_TILE_OFFSET));
    valueTileGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::VALUE_TILE_OFFSET));
    scoresGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace + turbo_quant_attention_detail::SCORES_OFFSET));
    probabilityGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(
            coreWorkspace +
            turbo_quant_attention_detail::PROBABILITY_OFFSET));
    tileOutputGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(
            coreWorkspace +
            turbo_quant_attention_detail::TILE_OUTPUT_OFFSET));

    constexpr uint32_t queryL1Bytes =
        turbo_quant_attention_detail::CUBE_M *
        turbo_quant_attention_detail::HEAD_DIM * sizeof(T);
    constexpr uint32_t probabilityL1Bytes =
        turbo_quant_attention_detail::CUBE_M *
        turbo_quant_attention_detail::CUBE_TILE_TOKENS * sizeof(T);
    constexpr uint32_t kvL1Bytes =
        turbo_quant_attention_detail::CUBE_TILE_TOKENS *
        turbo_quant_attention_detail::HEAD_DIM * sizeof(T);
    constexpr uint32_t l0ABytes = queryL1Bytes;
    constexpr uint32_t l0BBytes = kvL1Bytes;
    constexpr uint32_t l0CBytes =
        turbo_quant_attention_detail::CUBE_M *
        turbo_quant_attention_detail::CUBE_TILE_TOKENS *
        turbo_quant_attention_detail::FLOAT_BYTES;
    pipe_->InitBuffer(aL1Buffer_, queryL1Bytes);
    pipe_->InitBuffer(probabilityL1Buffer_, probabilityL1Bytes);
    pipe_->InitBuffer(bL1Buffer_, kvL1Bytes);
    pipe_->InitBuffer(aL0Buffer_, l0ABytes);
    pipe_->InitBuffer(bL0Buffer_, l0BBytes);
    pipe_->InitBuffer(cL0Buffer_, l0CBytes);
    aL1Local_ = aL1Buffer_.Get<T>();
    probabilityL1Local_ = probabilityL1Buffer_.Get<T>();
    bL1Local_ = bL1Buffer_.Get<T>();
    aL0Local_ = aL0Buffer_.Get<T>();
    bL0Local_ = bL0Buffer_.Get<T>();
    cL0Local_ = cL0Buffer_.Get<float>();
  }

  __aicore__ inline void Process() {
    for (uint32_t task = aiCoreIndex_; task < totalTasks_;
         task += usedCoreNum_) {
      ProcessTask(task);
    }
  }

 private:
  __aicore__ inline void DecodeTask(
      uint32_t task, uint32_t& batchIndex, uint32_t& kvHeadIndex,
      uint32_t& splitIndex) const {
    splitIndex = task % numSplits_;
    const uint32_t batchHead = task / numSplits_;
    kvHeadIndex = batchHead % numKvHeads_;
    batchIndex = batchHead / numKvHeads_;
  }

  __aicore__ inline uint32_t SequenceLength(uint32_t batchIndex) {
    const int32_t value = seqLensGm_.GetValue(batchIndex);
    if (value <= 0) {
      return 0U;
    }
    const uint32_t capacity = maxPages_ * blockSize_;
    return static_cast<uint32_t>(value) < capacity
               ? static_cast<uint32_t>(value)
               : capacity;
  }

  __aicore__ inline void ProcessTask(uint32_t task) {
    uint32_t batchIndex = 0U;
    uint32_t kvHeadIndex = 0U;
    uint32_t splitIndex = 0U;
    DecodeTask(task, batchIndex, kvHeadIndex, splitIndex);
    const uint32_t sequenceLength = SequenceLength(batchIndex);
    const uint32_t splitStart =
        static_cast<uint64_t>(sequenceLength) * splitIndex / numSplits_;
    const uint32_t splitEnd =
        static_cast<uint64_t>(sequenceLength) * (splitIndex + 1U) /
        numSplits_;
    const uint32_t queryHeadStart =
        kvHeadIndex * turbo_quant_attention_detail::GROUP_SIZE;
    LoadQuery(batchIndex, queryHeadStart);

    uint32_t tileIndex = 0U;
    for (uint32_t tileStart = splitStart; tileStart < splitEnd;
         tileStart += turbo_quant_attention_detail::TILE_TOKENS,
                  ++tileIndex) {
      const uint32_t remaining = splitEnd - tileStart;
      const uint32_t tokenCount =
          remaining < turbo_quant_attention_detail::TILE_TOKENS
              ? remaining
              : turbo_quant_attention_detail::TILE_TOKENS;
      const uint16_t syncOffset =
          static_cast<uint16_t>(tileIndex & 1U);
      CrossCoreWaitFlag(
          turbo_quant_attention_detail::SYNC_DEQUANT_READY +
          syncOffset);
      ComputeQK(tokenCount, 0U);
      CrossCoreSetFlag<turbo_quant_attention_detail::SYNC_MODE2, PIPE_FIX>(
          turbo_quant_attention_detail::SYNC_SCORES_READY +
          syncOffset);

      CrossCoreWaitFlag(
          turbo_quant_attention_detail::SYNC_PROBABILITY_READY +
          syncOffset);
      ComputePV(tokenCount, 0U);
      CrossCoreSetFlag<turbo_quant_attention_detail::SYNC_MODE2, PIPE_FIX>(
          turbo_quant_attention_detail::SYNC_TILE_OUTPUT_READY +
          syncOffset);
    }
  }

  __aicore__ inline void CopyGmNdToL1(
      LocalTensor<T>& destination, GlobalTensor<T>& source,
      uint32_t rows, uint32_t columns, uint32_t sourceStride,
      uint32_t destinationRowAlignment) {
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = columns;
    params.srcDValue = sourceStride;
    params.dstNzC0Stride = destinationRowAlignment;
    params.dstNzNStride = 1;
    params.srcNdMatrixStride = 0;
    params.dstNzMatrixStride = 0;
    DataCopy(destination, source, params);
  }

  __aicore__ inline void LoadQuery(
      uint32_t batchIndex, uint32_t queryHeadStart) {
    const uint64_t queryOffset =
        (static_cast<uint64_t>(batchIndex) * numQueryHeads_ +
         queryHeadStart) *
        turbo_quant_attention_detail::HEAD_DIM;
    GlobalTensor<T> querySource =
        queryGm_[queryOffset];
    CopyGmNdToL1(
        aL1Local_, querySource,
        turbo_quant_attention_detail::GROUP_SIZE,
        turbo_quant_attention_detail::HEAD_DIM,
        turbo_quant_attention_detail::HEAD_DIM,
        turbo_quant_attention_detail::CUBE_M);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
  }

  __aicore__ inline void LoadA(
      LocalTensor<T>& source, uint32_t mSize, uint32_t kSize) {
    LoadData3DParamsV2<T> params;
    params.l1H = mSize / 16U;
    params.l1W = 16U;
    params.padList[0] = 0;
    params.padList[1] = 0;
    params.padList[2] = 0;
    params.padList[3] = 255;
    params.mExtension = mSize;
    params.kExtension = kSize;
    params.mStartPt = 0;
    params.kStartPt = 0;
    params.strideW = 1;
    params.strideH = 1;
    params.filterW = 1;
    params.filterSizeW = false;
    params.filterH = 1;
    params.filterSizeH = false;
    params.dilationFilterW = 1;
    params.dilationFilterH = 1;
    params.enTranspose = 0;
    params.fMatrixCtrl = 0;
    params.channelSize = kSize;
    LoadData<T, turbo_quant_attention_detail::LOAD3D_CONFIG>(
        aL0Local_, source, params);
  }

  __aicore__ inline void ComputeQK(
      uint32_t tokenCount, uint32_t bufferIndex) {
    LoadA(aL1Local_, turbo_quant_attention_detail::CUBE_M,
          turbo_quant_attention_detail::HEAD_DIM);
    for (uint32_t cubeStart = 0U; cubeStart < tokenCount;
         cubeStart +=
             turbo_quant_attention_detail::CUBE_TILE_TOKENS) {
      const uint32_t cubeRemaining = tokenCount - cubeStart;
      const uint32_t cubeTokenCount =
          cubeRemaining <
                  turbo_quant_attention_detail::CUBE_TILE_TOKENS
              ? cubeRemaining
              : turbo_quant_attention_detail::CUBE_TILE_TOKENS;
      const uint32_t cubeTokenCountAligned =
          turbo_quant_attention_detail::Align16(cubeTokenCount);
      GlobalTensor<T> keySource =
          keyTileGm_[
              bufferIndex *
                      turbo_quant_attention_detail::KV_TILE_ELEMENTS +
              cubeStart *
                  turbo_quant_attention_detail::HEAD_DIM];
      CopyGmNdToL1(
          bL1Local_, keySource, cubeTokenCount,
          turbo_quant_attention_detail::HEAD_DIM,
          turbo_quant_attention_detail::HEAD_DIM,
          cubeTokenCountAligned);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);

      LoadData2DParams loadBParams;
      loadBParams.startIndex = 0;
      loadBParams.repeatTimes =
          (cubeTokenCountAligned / 16U) *
          (turbo_quant_attention_detail::HEAD_DIM /
           (32U / sizeof(T)));
      loadBParams.srcStride = 1;
      loadBParams.dstGap = 0;
      loadBParams.ifTranspose = false;
      LoadData(bL0Local_, bL1Local_, loadBParams);
      SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
      WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);

      MmadParams mmParams;
      mmParams.m = turbo_quant_attention_detail::CUBE_M;
      mmParams.n = cubeTokenCountAligned;
      mmParams.k = turbo_quant_attention_detail::HEAD_DIM;
      mmParams.cmatrixInitVal = true;
      mmParams.cmatrixSource = false;
      mmParams.unitFlag = 0b11;
      Mmad(cL0Local_, aL0Local_, bL0Local_, mmParams);
      PipeBarrier<PIPE_M>();
      SetFlag<HardEvent::M_FIX>(EVENT_ID3);
      WaitFlag<HardEvent::M_FIX>(EVENT_ID3);

      FixpipeParamsV220 fixParams;
      fixParams.nSize = cubeTokenCountAligned;
      fixParams.mSize = turbo_quant_attention_detail::CUBE_M;
      fixParams.srcStride = turbo_quant_attention_detail::CUBE_M;
      fixParams.dstStride =
          turbo_quant_attention_detail::TILE_TOKENS;
      fixParams.unitFlag = 0b11;
      fixParams.ndNum = 1;
      Fixpipe(
          scoresGm_[
              bufferIndex *
                      turbo_quant_attention_detail::CUBE_M *
                      turbo_quant_attention_detail::TILE_TOKENS +
              cubeStart],
          cL0Local_, fixParams);
    }
  }

  __aicore__ inline void ComputePV(
      uint32_t tokenCount, uint32_t bufferIndex) {
    uint32_t cubeIndex = 0U;
    for (uint32_t cubeStart = 0U; cubeStart < tokenCount;
         cubeStart +=
             turbo_quant_attention_detail::CUBE_TILE_TOKENS,
                  ++cubeIndex) {
      const uint32_t cubeRemaining = tokenCount - cubeStart;
      const uint32_t cubeTokenCount =
          cubeRemaining <
                  turbo_quant_attention_detail::CUBE_TILE_TOKENS
              ? cubeRemaining
              : turbo_quant_attention_detail::CUBE_TILE_TOKENS;
      const uint32_t cubeTokenCountAligned =
          turbo_quant_attention_detail::Align16(cubeTokenCount);
      GlobalTensor<T> probabilitySource =
          probabilityGm_[
              bufferIndex *
                      turbo_quant_attention_detail::CUBE_M *
                      turbo_quant_attention_detail::TILE_TOKENS +
              cubeStart];
      CopyGmNdToL1(
          probabilityL1Local_, probabilitySource,
          turbo_quant_attention_detail::GROUP_SIZE,
          cubeTokenCountAligned,
          turbo_quant_attention_detail::TILE_TOKENS,
          turbo_quant_attention_detail::CUBE_M);
      GlobalTensor<T> valueSource =
          valueTileGm_[
              bufferIndex *
                      turbo_quant_attention_detail::KV_TILE_ELEMENTS +
              cubeStart *
                  turbo_quant_attention_detail::HEAD_DIM];
      CopyGmNdToL1(
          bL1Local_, valueSource, cubeTokenCount,
          turbo_quant_attention_detail::HEAD_DIM,
          turbo_quant_attention_detail::HEAD_DIM,
          cubeTokenCountAligned);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
      WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

      LoadA(probabilityL1Local_,
            turbo_quant_attention_detail::CUBE_M,
            cubeTokenCountAligned);

      LoadData3DParamsV2<T> loadBParams;
      loadBParams.l1H = cubeTokenCountAligned / 16U;
      loadBParams.l1W = 16U;
      loadBParams.padList[0] = 0;
      loadBParams.padList[1] = 0;
      loadBParams.padList[2] = 0;
      loadBParams.padList[3] = 255;
      loadBParams.mExtension = cubeTokenCountAligned;
      loadBParams.kExtension =
          turbo_quant_attention_detail::HEAD_DIM;
      loadBParams.mStartPt = 0;
      loadBParams.kStartPt = 0;
      loadBParams.strideW = 1;
      loadBParams.strideH = 1;
      loadBParams.filterW = 1;
      loadBParams.filterSizeW = false;
      loadBParams.filterH = 1;
      loadBParams.filterSizeH = false;
      loadBParams.dilationFilterW = 1;
      loadBParams.dilationFilterH = 1;
      loadBParams.enTranspose = 1;
      loadBParams.fMatrixCtrl = 0;
      loadBParams.channelSize =
          turbo_quant_attention_detail::HEAD_DIM;
      LoadData<T, turbo_quant_attention_detail::LOAD3D_CONFIG>(
          bL0Local_, bL1Local_, loadBParams);
      SetFlag<HardEvent::MTE1_M>(EVENT_ID4);
      WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);

      MmadParams mmParams;
      mmParams.m = turbo_quant_attention_detail::CUBE_M;
      mmParams.n = turbo_quant_attention_detail::HEAD_DIM;
      mmParams.k = cubeTokenCountAligned;
      mmParams.cmatrixInitVal = cubeIndex == 0U;
      mmParams.cmatrixSource = false;
      mmParams.unitFlag = 0b11;
      Mmad(cL0Local_, aL0Local_, bL0Local_, mmParams);
      PipeBarrier<PIPE_M>();
      SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
      WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
    }
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_FIX>(EVENT_ID4);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID4);

    FixpipeParamsV220 fixParams;
    fixParams.nSize = turbo_quant_attention_detail::HEAD_DIM;
    fixParams.mSize = turbo_quant_attention_detail::CUBE_M;
    fixParams.srcStride = turbo_quant_attention_detail::CUBE_M;
    fixParams.dstStride = turbo_quant_attention_detail::HEAD_DIM;
    fixParams.unitFlag = 0b11;
    fixParams.ndNum = 1;
    Fixpipe(
        tileOutputGm_[
            bufferIndex *
            turbo_quant_attention_detail::CUBE_M *
            turbo_quant_attention_detail::HEAD_DIM],
        cL0Local_, fixParams);
  }

  TPipe* pipe_ = nullptr;
  TBuf<TPosition::A1> aL1Buffer_;
  TBuf<TPosition::A1> probabilityL1Buffer_;
  TBuf<TPosition::B1> bL1Buffer_;
  TBuf<TPosition::A2> aL0Buffer_;
  TBuf<TPosition::B2> bL0Buffer_;
  TBuf<TPosition::CO1> cL0Buffer_;
  LocalTensor<T> aL1Local_;
  LocalTensor<T> probabilityL1Local_;
  LocalTensor<T> bL1Local_;
  LocalTensor<T> aL0Local_;
  LocalTensor<T> bL0Local_;
  LocalTensor<float> cL0Local_;
  GlobalTensor<T> queryGm_;
  GlobalTensor<int32_t> seqLensGm_;
  GlobalTensor<T> keyTileGm_;
  GlobalTensor<T> valueTileGm_;
  GlobalTensor<float> scoresGm_;
  GlobalTensor<T> probabilityGm_;
  GlobalTensor<float> tileOutputGm_;
  uint32_t numQueryHeads_ = 0U;
  uint32_t numKvHeads_ = 0U;
  uint32_t blockSize_ = 0U;
  uint32_t maxPages_ = 0U;
  uint32_t numSplits_ = 0U;
  uint32_t totalTasks_ = 0U;
  uint32_t usedCoreNum_ = 0U;
  uint32_t coreWorkspaceBytes_ = 0U;
  uint32_t aiCoreIndex_ = 0U;
};

extern "C" __global__ __aicore__ void turbo_quant_paged_attention(
    GM_ADDR query, GM_ADDR kvCache, GM_ADDR blockTable, GM_ADDR seqLens,
    GM_ADDR centroids, GM_ADDR attentionOut, GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(tilingData, tiling);
  __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspace);
  TPipe pipe;

  if (TILING_KEY_IS(1)) {
    if ASCEND_IS_AIV {
      KernelTurboQuantPagedAttentionVector<DTYPE_QUERY> op;
      op.Init(query, kvCache, blockTable, seqLens, centroids,
              attentionOut, userWorkspace, tilingData, &pipe);
      op.Process();
    } else {
      KernelTurboQuantPagedAttentionCube<DTYPE_QUERY> op;
      op.Init(query, seqLens, userWorkspace, tilingData, &pipe);
      op.Process();
    }
  }
}
