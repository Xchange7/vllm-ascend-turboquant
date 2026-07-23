/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
static ge::graphStatus InferShapeTurboQuantPagedAttention(
    gert::InferShapeContext* context) {
  OPS_ERR_IF(context == nullptr,
             OPS_LOG_E("TurboQuantPagedAttention",
                       "InferShapeContext is nullptr"),
             return ge::GRAPH_FAILED);
  const gert::Shape* queryShape = context->GetInputShape(0);
  OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
  gert::Shape* outputShape = context->GetOutputShape(0);
  OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED);
  *outputShape = *queryShape;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTurboQuantPagedAttention(
    gert::InferDataTypeContext* context) {
  OPS_ERR_IF(context == nullptr,
             OPS_LOG_E("TurboQuantPagedAttention",
                       "InferDataTypeContext is nullptr"),
             return ge::GRAPH_FAILED);
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantPagedAttention)
    .InferShape(InferShapeTurboQuantPagedAttention)
    .InferDataType(InferDataTypeTurboQuantPagedAttention);
}  // namespace ops
