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

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
namespace {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KV_CACHE_INPUT_INDEX = 1;
constexpr size_t MAX_SEQ_LEN_ATTR_INDEX = 0;
}  // namespace

static ge::graphStatus InferShapeTurboQuantPagedDequant(gert::InferShapeContext* context) {
  OPS_ERR_IF(context == nullptr, OPS_LOG_E("TurboQuantPagedDequant", "InferShapeContext is nullptr"),
             return ge::GRAPH_FAILED);
  const gert::Shape* queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
  const gert::Shape* cacheShape = context->GetInputShape(KV_CACHE_INPUT_INDEX);
  OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
  OPS_LOG_E_IF_NULL(context, cacheShape, return ge::GRAPH_FAILED);
  auto attrs = context->GetAttrs();
  OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
  const int64_t* maxSeqLen = attrs->GetAttrPointer<int64_t>(MAX_SEQ_LEN_ATTR_INDEX);
  OPS_LOG_E_IF_NULL(context, maxSeqLen, return ge::GRAPH_FAILED);

  OPS_ERR_IF(queryShape->GetDimNum() != 3, OPS_LOG_E(context, "query must have shape [B, N, D]"),
             return ge::GRAPH_FAILED);
  OPS_ERR_IF(cacheShape->GetDimNum() != 4, OPS_LOG_E(context, "kvCache must have shape [blocks, block, N, slot]"),
             return ge::GRAPH_FAILED);

  for (size_t outputIndex = 0; outputIndex < 2; ++outputIndex) {
    gert::Shape* outputShape = context->GetOutputShape(outputIndex);
    OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, queryShape->GetDim(0));
    outputShape->SetDim(1, cacheShape->GetDim(2));
    outputShape->SetDim(2, *maxSeqLen);
    outputShape->SetDim(3, queryShape->GetDim(2));
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTurboQuantPagedDequant(gert::InferDataTypeContext* context) {
  OPS_ERR_IF(context == nullptr, OPS_LOG_E("TurboQuantPagedDequant", "InferDataTypeContext is nullptr"),
             return ge::GRAPH_FAILED);
  const auto queryDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
  context->SetOutputDataType(0, queryDataType);
  context->SetOutputDataType(1, queryDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantPagedDequant)
    .InferShape(InferShapeTurboQuantPagedDequant)
    .InferDataType(InferDataTypeTurboQuantPagedDequant);
}  // namespace ops
