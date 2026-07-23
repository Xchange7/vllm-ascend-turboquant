/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "turbo_quant_paged_attention_tiling.h"

namespace {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KV_CACHE_INPUT_INDEX = 1;
constexpr size_t BLOCK_TABLE_INPUT_INDEX = 2;
constexpr size_t SEQ_LENS_INPUT_INDEX = 3;
constexpr size_t CENTROIDS_INPUT_INDEX = 4;

constexpr size_t SCALE_ATTR_INDEX = 0;
constexpr size_t MAX_SEQ_LEN_ATTR_INDEX = 1;
constexpr size_t KEY_BITS_ATTR_INDEX = 2;
constexpr size_t KEY_PACKED_SIZE_ATTR_INDEX = 3;
constexpr size_t VALUE_BITS_ATTR_INDEX = 4;
constexpr size_t NORM_CORRECTION_ATTR_INDEX = 5;
constexpr size_t MAX_NUM_SPLITS_ATTR_INDEX = 6;

constexpr uint32_t SUPPORTED_HEAD_DIM = 128;
constexpr uint32_t SUPPORTED_GROUP_SIZE = 8;
constexpr uint32_t TILE_TOKENS = 384;
constexpr uint32_t CUBE_ROWS = 16;
constexpr uint32_t ELEMENT_BYTES = 2;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t STATE_ELEMENTS_PER_LINE = 8;
constexpr uint32_t STATE_LINES_PER_TASK = 2;
constexpr uint32_t PIPELINE_BUFFER_COUNT = 1;
constexpr uint32_t KEY_METADATA_BYTES = 2;
constexpr uint32_t VALUE_METADATA_BYTES = 4;
constexpr uint32_t WORKSPACE_ALIGNMENT = 512;

constexpr uint64_t AlignWorkspace(uint64_t value) {
  return (value + WORKSPACE_ALIGNMENT - 1U) &
         ~(static_cast<uint64_t>(WORKSPACE_ALIGNMENT) - 1U);
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  OPS_CHECK(context == nullptr,
            OPS_LOG_E("TurboQuantPagedAttention",
                      "tiling context is nullptr"),
            return ge::GRAPH_FAILED);
  const char* nodeName = context->GetNodeName();
  const auto* queryInputShape = context->GetInputShape(QUERY_INPUT_INDEX);
  const auto* cacheInputShape =
      context->GetInputShape(KV_CACHE_INPUT_INDEX);
  const auto* blockTableInputShape =
      context->GetInputShape(BLOCK_TABLE_INPUT_INDEX);
  const auto* seqLensInputShape =
      context->GetInputShape(SEQ_LENS_INPUT_INDEX);
  const auto* centroidsInputShape =
      context->GetInputShape(CENTROIDS_INPUT_INDEX);
  OPS_CHECK(queryInputShape == nullptr || cacheInputShape == nullptr ||
                blockTableInputShape == nullptr ||
                seqLensInputShape == nullptr ||
                centroidsInputShape == nullptr,
            OPS_LOG_E(nodeName, "required input shape is missing"),
            return ge::GRAPH_FAILED);

  const auto queryShape = queryInputShape->GetStorageShape();
  const auto cacheShape = cacheInputShape->GetStorageShape();
  const auto blockTableShape = blockTableInputShape->GetStorageShape();
  const auto seqLensShape = seqLensInputShape->GetStorageShape();
  const auto centroidsShape = centroidsInputShape->GetStorageShape();
  OPS_CHECK(queryShape.GetDimNum() != 3 || cacheShape.GetDimNum() != 4 ||
                blockTableShape.GetDimNum() != 2 ||
                seqLensShape.GetDimNum() != 1 ||
                centroidsShape.GetDimNum() != 1,
            OPS_LOG_E(nodeName, "invalid input rank"),
            return ge::GRAPH_FAILED);

  auto attrs = context->GetAttrs();
  OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr"),
            return ge::GRAPH_FAILED);
  const float* scale = attrs->GetAttrPointer<float>(SCALE_ATTR_INDEX);
  const int64_t* maxSeqLen =
      attrs->GetAttrPointer<int64_t>(MAX_SEQ_LEN_ATTR_INDEX);
  const int64_t* keyBits =
      attrs->GetAttrPointer<int64_t>(KEY_BITS_ATTR_INDEX);
  const int64_t* keyPackedSize =
      attrs->GetAttrPointer<int64_t>(KEY_PACKED_SIZE_ATTR_INDEX);
  const int64_t* valueBits =
      attrs->GetAttrPointer<int64_t>(VALUE_BITS_ATTR_INDEX);
  const bool* normCorrection =
      attrs->GetAttrPointer<bool>(NORM_CORRECTION_ATTR_INDEX);
  const int64_t* maxNumSplits =
      attrs->GetAttrPointer<int64_t>(MAX_NUM_SPLITS_ATTR_INDEX);
  OPS_CHECK(scale == nullptr || maxSeqLen == nullptr || keyBits == nullptr ||
                keyPackedSize == nullptr || valueBits == nullptr ||
                normCorrection == nullptr || maxNumSplits == nullptr,
            OPS_LOG_E(nodeName, "required attribute is missing"),
            return ge::GRAPH_FAILED);

  const int64_t batchSize = queryShape.GetDim(0);
  const int64_t numQueryHeads = queryShape.GetDim(1);
  const int64_t headDim = queryShape.GetDim(2);
  const int64_t numBlocks = cacheShape.GetDim(0);
  const int64_t blockSize = cacheShape.GetDim(1);
  const int64_t numKvHeads = cacheShape.GetDim(2);
  const int64_t slotSize = cacheShape.GetDim(3);
  const int64_t maxPages = blockTableShape.GetDim(1);
  OPS_CHECK(batchSize <= 0 || numQueryHeads <= 0 || numBlocks <= 0 ||
                blockSize <= 0 || numKvHeads <= 0 || slotSize <= 0 ||
                maxPages <= 0 ||
                blockTableShape.GetDim(0) != batchSize ||
                seqLensShape.GetDim(0) != batchSize,
            OPS_LOG_E(nodeName, "input dimensions are inconsistent"),
            return ge::GRAPH_FAILED);
  OPS_CHECK(headDim != SUPPORTED_HEAD_DIM ||
                numQueryHeads % numKvHeads != 0 ||
                numQueryHeads / numKvHeads != SUPPORTED_GROUP_SIZE,
            OPS_LOG_E(nodeName,
                      "the first pipeline supports D=128 and GQA group=8"),
            return ge::GRAPH_FAILED);
  OPS_CHECK(*keyBits != 4 || *valueBits != 4 || !*normCorrection,
            OPS_LOG_E(nodeName,
                      "the first pipeline supports K4V4 with norm correction"),
            return ge::GRAPH_FAILED);
  OPS_CHECK(!std::isfinite(*scale) || *scale <= 0.0F ||
                *maxSeqLen <= 0 || *maxNumSplits <= 0,
            OPS_LOG_E(nodeName, "attention attributes are invalid"),
            return ge::GRAPH_FAILED);
  OPS_CHECK(*maxSeqLen > maxPages * blockSize,
            OPS_LOG_E(nodeName,
                      "max_seq_len exceeds block-table capacity"),
            return ge::GRAPH_FAILED);

  const int64_t keyDataBytes = headDim * *keyBits / 8;
  const int64_t valueDataBytes = headDim * *valueBits / 8;
  const int64_t expectedKeyPackedSize =
      keyDataBytes + KEY_METADATA_BYTES;
  const int64_t payloadBytes = expectedKeyPackedSize + valueDataBytes +
                               VALUE_METADATA_BYTES;
  const int64_t expectedSlotSize = payloadBytes + payloadBytes % 2;
  OPS_CHECK(*keyPackedSize != expectedKeyPackedSize ||
                slotSize != expectedSlotSize ||
                centroidsShape.GetDim(0) < (int64_t{1} << *keyBits),
            OPS_LOG_E(nodeName, "packed cache layout is inconsistent"),
            return ge::GRAPH_FAILED);

  auto* platformInfo =
      const_cast<fe::PlatFormInfos*>(context->GetPlatformInfo());
  OPS_CHECK(platformInfo == nullptr,
            OPS_LOG_E(nodeName, "platform info is nullptr"),
            return ge::GRAPH_FAILED);
  auto platform = platform_ascendc::PlatformAscendC(platformInfo);
  const uint32_t aicCoreCount = platform.GetCoreNumAic();
  OPS_CHECK(aicCoreCount == 0,
            OPS_LOG_E(nodeName, "no AIC core is available"),
            return ge::GRAPH_FAILED);

  const uint64_t baseTasks =
      static_cast<uint64_t>(batchSize) * numKvHeads;
  const uint32_t minimumSplits = static_cast<uint32_t>(
      (aicCoreCount + baseTasks - 1U) / baseTasks);
  const uint32_t splitLimit =
      std::max(1U, std::min({static_cast<uint32_t>(*maxNumSplits),
                            static_cast<uint32_t>(*maxSeqLen), 8U}));
  uint32_t numSplits =
      std::max(1U, std::min(minimumSplits, splitLimit));

  // Merely filling all cores can leave a long straggler wave. For example,
  // Qwen3-32B TP2 at batch 16 has 64 KV-head tasks: one split schedules
  // 3/3/.../2 tasks on 24 AICs, while three splits schedule exactly eight
  // one-third-length tasks per core. Search a small split range for the
  // lowest normalized critical-path work and prefer fewer splits on ties.
  if (*maxSeqLen >= static_cast<int64_t>(16U * TILE_TOKENS)) {
    uint64_t bestWaves =
        (baseTasks * numSplits + aicCoreCount - 1U) / aicCoreCount;
    for (uint32_t candidate = numSplits + 1U;
         candidate <= splitLimit; ++candidate) {
      const uint64_t candidateWaves =
          (baseTasks * candidate + aicCoreCount - 1U) / aicCoreCount;
      if (candidateWaves * numSplits < bestWaves * candidate) {
        numSplits = candidate;
        bestWaves = candidateWaves;
      }
    }
  }
  const uint64_t totalTasksWide = baseTasks * numSplits;
  OPS_CHECK(totalTasksWide > std::numeric_limits<uint32_t>::max(),
            OPS_LOG_E(nodeName, "task count exceeds uint32 range"),
            return ge::GRAPH_FAILED);
  const uint32_t totalTasks = static_cast<uint32_t>(totalTasksWide);
  const uint32_t usedCoreNum = std::min(totalTasks, aicCoreCount);

  constexpr uint64_t keyTileBytes =
      TILE_TOKENS * SUPPORTED_HEAD_DIM * ELEMENT_BYTES;
  constexpr uint64_t valueTileBytes = keyTileBytes;
  constexpr uint64_t keyScaleBytes =
      TILE_TOKENS * FLOAT_BYTES;
  constexpr uint64_t valueScaleBytes =
      TILE_TOKENS * FLOAT_BYTES;
  constexpr uint64_t valueMinimumBytes =
      TILE_TOKENS * FLOAT_BYTES;
  constexpr uint64_t scoresBytes =
      CUBE_ROWS * TILE_TOKENS * FLOAT_BYTES;
  constexpr uint64_t probabilityBytes =
      CUBE_ROWS * TILE_TOKENS * ELEMENT_BYTES;
  constexpr uint64_t tileOutputBytes =
      CUBE_ROWS * SUPPORTED_HEAD_DIM * FLOAT_BYTES;
  constexpr uint64_t coreWorkspaceBytes = AlignWorkspace(
      PIPELINE_BUFFER_COUNT *
          (keyTileBytes + valueTileBytes + keyScaleBytes +
           valueScaleBytes + valueMinimumBytes + scoresBytes +
           probabilityBytes + tileOutputBytes));
  const uint64_t coreWorkspaceTotal =
      static_cast<uint64_t>(usedCoreNum) * coreWorkspaceBytes;
  const uint64_t partialAccumOffset = coreWorkspaceTotal;
  const uint64_t partialAccumBytes =
      totalTasksWide * SUPPORTED_GROUP_SIZE * SUPPORTED_HEAD_DIM *
      FLOAT_BYTES;
  const uint64_t partialSumOffset =
      AlignWorkspace(partialAccumOffset + partialAccumBytes);
  const uint64_t partialStateBytes =
      totalTasksWide * STATE_LINES_PER_TASK *
      STATE_ELEMENTS_PER_LINE * FLOAT_BYTES;
  const uint64_t partialMaxOffset =
      AlignWorkspace(partialSumOffset + partialStateBytes);
  const uint64_t userWorkspaceBytes =
      AlignWorkspace(partialMaxOffset + partialStateBytes);
  const uint64_t totalWorkspaceBytes =
      userWorkspaceBytes + platform.GetLibApiWorkSpaceSize();
  OPS_CHECK(totalWorkspaceBytes > std::numeric_limits<size_t>::max(),
            OPS_LOG_E(nodeName, "workspace size exceeds size_t"),
            return ge::GRAPH_FAILED);

  TurboQuantPagedAttentionTilingData tiling;
  tiling.set_batchSize(static_cast<uint32_t>(batchSize));
  tiling.set_numQueryHeads(static_cast<uint32_t>(numQueryHeads));
  tiling.set_numKvHeads(static_cast<uint32_t>(numKvHeads));
  tiling.set_numBlocks(static_cast<uint32_t>(numBlocks));
  tiling.set_blockSize(static_cast<uint32_t>(blockSize));
  tiling.set_maxPages(static_cast<uint32_t>(maxPages));
  tiling.set_slotSize(static_cast<uint32_t>(slotSize));
  tiling.set_groupSize(SUPPORTED_GROUP_SIZE);
  tiling.set_numSplits(numSplits);
  tiling.set_totalTasks(totalTasks);
  tiling.set_usedCoreNum(usedCoreNum);
  tiling.set_coreWorkspaceBytes(static_cast<uint32_t>(coreWorkspaceBytes));
  tiling.set_partialAccumOffset(partialAccumOffset);
  tiling.set_partialSumOffset(partialSumOffset);
  tiling.set_partialMaxOffset(partialMaxOffset);
  tiling.set_scale(*scale);

  context->SetTilingKey(1);
  context->SetBlockDim(usedCoreNum);
  size_t* workspaceSizes = context->GetWorkspaceSizes(1);
  auto* rawTilingData = context->GetRawTilingData();
  OPS_CHECK(workspaceSizes == nullptr || rawTilingData == nullptr,
            OPS_LOG_E(nodeName, "workspace or tiling buffer is nullptr"),
            return ge::GRAPH_FAILED);
  workspaceSizes[0] = static_cast<size_t>(totalWorkspaceBytes);
  tiling.SaveToBuffer(rawTilingData->GetData(),
                      rawTilingData->GetCapacity());
  rawTilingData->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

struct TurboQuantPagedAttentionCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantPagedAttention)
    .Tiling(TilingFunc)
    .TilingParse<TurboQuantPagedAttentionCompileInfo>(TilingParse);
}  // namespace optiling
