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

#include "register/op_def_registry.h"

namespace ops {
class TurboQuantPagedDequant : public OpDef {
 public:
  explicit TurboQuantPagedDequant(const char* name) : OpDef(name) {
    this->Input("query")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("kvCache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8, ge::DT_UINT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("blockTable")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("seqLens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("centroids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("key")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("value")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("max_seq_len").AttrType(REQUIRED).Int();
    this->Attr("key_bits").AttrType(REQUIRED).Int();
    this->Attr("key_packed_size").AttrType(REQUIRED).Int();
    this->Attr("value_bits").AttrType(REQUIRED).Int();
    this->Attr("norm_correction").AttrType(REQUIRED).Bool();

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
    this->AICore().AddConfig("ascend910b", aicoreConfig);
    this->AICore().AddConfig("ascend910_93", aicoreConfig);
  }
};

OP_ADD(TurboQuantPagedDequant);
}  // namespace ops
