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

#include <algorithm>
#include <cstdint>
#include <limits>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "turbo_quant_paged_dequant_tiling.h"

namespace {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KV_CACHE_INPUT_INDEX = 1;
constexpr size_t BLOCK_TABLE_INPUT_INDEX = 2;
constexpr size_t SEQ_LENS_INPUT_INDEX = 3;
constexpr size_t CENTROIDS_INPUT_INDEX = 4;
constexpr size_t MAX_SEQ_LEN_ATTR_INDEX = 0;
constexpr size_t KEY_BITS_ATTR_INDEX = 1;
constexpr size_t KEY_PACKED_SIZE_ATTR_INDEX = 2;
constexpr size_t VALUE_BITS_ATTR_INDEX = 3;
constexpr size_t NORM_CORRECTION_ATTR_INDEX = 4;
constexpr int64_t METADATA_BYTES = 2;
constexpr int64_t VALUE_METADATA_BYTES = 4;
constexpr int64_t MIN_HEAD_DIM = 32;
constexpr int64_t MAX_HEAD_DIM = 256;
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  OPS_CHECK(context == nullptr, OPS_LOG_E("TurboQuantPagedDequant", "tiling context is nullptr"),
            return ge::GRAPH_FAILED);
  const char* nodeName = context->GetNodeName();
  const auto* queryInputShape = context->GetInputShape(QUERY_INPUT_INDEX);
  const auto* cacheInputShape = context->GetInputShape(KV_CACHE_INPUT_INDEX);
  const auto* blockTableInputShape = context->GetInputShape(BLOCK_TABLE_INPUT_INDEX);
  const auto* seqLensInputShape = context->GetInputShape(SEQ_LENS_INPUT_INDEX);
  const auto* centroidsInputShape = context->GetInputShape(CENTROIDS_INPUT_INDEX);
  OPS_CHECK(queryInputShape == nullptr || cacheInputShape == nullptr || blockTableInputShape == nullptr ||
                seqLensInputShape == nullptr || centroidsInputShape == nullptr,
            OPS_LOG_E(nodeName, "required TurboQuant input shape is missing"), return ge::GRAPH_FAILED);
  const auto queryShape = queryInputShape->GetStorageShape();
  const auto cacheShape = cacheInputShape->GetStorageShape();
  const auto blockTableShape = blockTableInputShape->GetStorageShape();
  const auto seqLensShape = seqLensInputShape->GetStorageShape();
  const auto centroidsShape = centroidsInputShape->GetStorageShape();
  auto attrs = context->GetAttrs();
  OPS_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr"), return ge::GRAPH_FAILED);

  const int64_t* maxSeqLen = attrs->GetAttrPointer<int64_t>(MAX_SEQ_LEN_ATTR_INDEX);
  const int64_t* keyBits = attrs->GetAttrPointer<int64_t>(KEY_BITS_ATTR_INDEX);
  const int64_t* keyPackedSize = attrs->GetAttrPointer<int64_t>(KEY_PACKED_SIZE_ATTR_INDEX);
  const int64_t* valueBits = attrs->GetAttrPointer<int64_t>(VALUE_BITS_ATTR_INDEX);
  const bool* normCorrection = attrs->GetAttrPointer<bool>(NORM_CORRECTION_ATTR_INDEX);
  OPS_CHECK(maxSeqLen == nullptr || keyBits == nullptr || keyPackedSize == nullptr || valueBits == nullptr ||
                normCorrection == nullptr,
            OPS_LOG_E(nodeName, "required TurboQuant attribute is missing"), return ge::GRAPH_FAILED);

  OPS_CHECK(queryShape.GetDimNum() != 3 || cacheShape.GetDimNum() != 4 || blockTableShape.GetDimNum() != 2 ||
                seqLensShape.GetDimNum() != 1 || centroidsShape.GetDimNum() != 1,
            OPS_LOG_E(nodeName, "invalid TurboQuant input rank"), return ge::GRAPH_FAILED);

  const int64_t batchSize = queryShape.GetDim(0);
  const int64_t numQueryHeads = queryShape.GetDim(1);
  const int64_t headDim = queryShape.GetDim(2);
  const int64_t numBlocks = cacheShape.GetDim(0);
  const int64_t blockSize = cacheShape.GetDim(1);
  const int64_t numKvHeads = cacheShape.GetDim(2);
  const int64_t slotSize = cacheShape.GetDim(3);
  const int64_t maxPages = blockTableShape.GetDim(1);
  OPS_CHECK(batchSize <= 0 || numQueryHeads <= 0 || numBlocks <= 0 || blockSize <= 0 || numKvHeads <= 0 ||
                slotSize <= 0 || maxPages <= 0 || blockTableShape.GetDim(0) != batchSize ||
                seqLensShape.GetDim(0) != batchSize,
            OPS_LOG_E(nodeName, "TurboQuant input dimensions are invalid or inconsistent"), return ge::GRAPH_FAILED);
  OPS_CHECK(headDim < MIN_HEAD_DIM || headDim > MAX_HEAD_DIM || headDim % MIN_HEAD_DIM != 0,
            OPS_LOG_E(nodeName, "headDim must be a multiple of 32 in [32, 256]"), return ge::GRAPH_FAILED);
  OPS_CHECK((*keyBits != 3 && *keyBits != 4) || (*valueBits != 3 && *valueBits != 4),
            OPS_LOG_E(nodeName, "only 3-bit and 4-bit TurboQuant layouts are supported"), return ge::GRAPH_FAILED);
  OPS_CHECK(*maxSeqLen <= 0, OPS_LOG_E(nodeName, "maxSeqLen must be positive"), return ge::GRAPH_FAILED);
  const int64_t activePages = (*maxSeqLen - 1) / blockSize + 1;
  OPS_CHECK(activePages > maxPages, OPS_LOG_E(nodeName, "maxSeqLen exceeds block-table capacity"),
            return ge::GRAPH_FAILED);

  const int64_t keyDataBytes = (headDim * *keyBits + 7) / 8;
  const int64_t valueDataBytes = (headDim * *valueBits + 7) / 8;
  OPS_CHECK(*keyPackedSize != keyDataBytes + METADATA_BYTES,
            OPS_LOG_E(nodeName, "packed cache layout does not match TurboQuant attributes"), return ge::GRAPH_FAILED);
  const int64_t minimumSlotSize = keyDataBytes + METADATA_BYTES + valueDataBytes + VALUE_METADATA_BYTES;
  OPS_CHECK(slotSize < minimumSlotSize, OPS_LOG_E(nodeName, "packed cache slot is smaller than the TurboQuant payload"),
            return ge::GRAPH_FAILED);
  const int64_t centroidCount = int64_t{1} << *keyBits;
  OPS_CHECK(centroidsShape.GetDim(0) < centroidCount,
            OPS_LOG_E(nodeName, "centroid table is smaller than the key codebook"), return ge::GRAPH_FAILED);

  TurboQuantPagedDequantTilingData tiling;
  constexpr int64_t UINT32_MAX_VALUE = std::numeric_limits<uint32_t>::max();
  OPS_CHECK(batchSize > UINT32_MAX_VALUE || *maxSeqLen > UINT32_MAX_VALUE || maxPages > UINT32_MAX_VALUE ||
                activePages > UINT32_MAX_VALUE || numBlocks > UINT32_MAX_VALUE || blockSize > UINT32_MAX_VALUE ||
                numKvHeads > UINT32_MAX_VALUE || slotSize > UINT32_MAX_VALUE,
            OPS_LOG_E(nodeName, "TurboQuant dimensions exceed the uint32 tiling range"), return ge::GRAPH_FAILED);
  OPS_CHECK(batchSize > UINT32_MAX_VALUE / activePages || numKvHeads > UINT32_MAX_VALUE / (batchSize * activePages),
            OPS_LOG_E(nodeName, "TurboQuant task count exceeds the uint32 tiling range"), return ge::GRAPH_FAILED);
  const int64_t totalTasks = batchSize * activePages * numKvHeads;
  tiling.set_batchSize(static_cast<uint32_t>(batchSize));
  tiling.set_maxSeqLen(static_cast<uint32_t>(*maxSeqLen));
  tiling.set_maxPages(static_cast<uint32_t>(maxPages));
  tiling.set_activePages(static_cast<uint32_t>(activePages));
  tiling.set_numBlocks(static_cast<uint32_t>(numBlocks));
  tiling.set_blockSize(static_cast<uint32_t>(blockSize));
  tiling.set_numKvHeads(static_cast<uint32_t>(numKvHeads));
  tiling.set_headDim(static_cast<uint32_t>(headDim));
  tiling.set_slotSize(static_cast<uint32_t>(slotSize));
  tiling.set_keyBits(static_cast<uint32_t>(*keyBits));
  tiling.set_keyDataBytes(static_cast<uint32_t>(keyDataBytes));
  tiling.set_keyPackedSize(static_cast<uint32_t>(*keyPackedSize));
  tiling.set_valueBits(static_cast<uint32_t>(*valueBits));
  tiling.set_valueDataBytes(static_cast<uint32_t>(valueDataBytes));
  tiling.set_centroidCount(static_cast<uint32_t>(centroidCount));
  tiling.set_normCorrection(*normCorrection ? 1U : 0U);
  tiling.set_totalTasks(static_cast<uint32_t>(totalTasks));

  const auto* platformInfo = context->GetPlatformInfo();
  OPS_CHECK(platformInfo == nullptr, OPS_LOG_E(nodeName, "platform info is nullptr"), return ge::GRAPH_FAILED);
  auto platform = platform_ascendc::PlatformAscendC(platformInfo);
  const uint32_t aivCoreCount = platform.GetCoreNumAiv();
  OPS_CHECK(aivCoreCount == 0, OPS_LOG_E(nodeName, "no AIV core is available"), return ge::GRAPH_FAILED);
  const uint32_t blockDim = std::min(static_cast<uint32_t>(totalTasks), aivCoreCount);
  context->SetTilingKey(0);
  context->SetBlockDim(blockDim);
  size_t* workspaceSizes = context->GetWorkspaceSizes(1);
  auto* rawTilingData = context->GetRawTilingData();
  OPS_CHECK(workspaceSizes == nullptr || rawTilingData == nullptr,
            OPS_LOG_E(nodeName, "workspace or raw tiling buffer is nullptr"), return ge::GRAPH_FAILED);
  workspaceSizes[0] = 0;
  tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
  rawTilingData->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

struct TurboQuantPagedDequantCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantPagedDequant).Tiling(TilingFunc).TilingParse<TurboQuantPagedDequantCompileInfo>(TilingParse);
}  // namespace optiling
