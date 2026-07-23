/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops {
class TurboQuantPagedAttention : public OpDef {
 public:
  explicit TurboQuantPagedAttention(const char* name) : OpDef(name) {
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
    this->Output("attentionOut")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("scale").AttrType(REQUIRED).Float();
    this->Attr("max_seq_len").AttrType(REQUIRED).Int();
    this->Attr("key_bits").AttrType(REQUIRED).Int();
    this->Attr("key_packed_size").AttrType(REQUIRED).Int();
    this->Attr("value_bits").AttrType(REQUIRED).Int();
    this->Attr("norm_correction").AttrType(REQUIRED).Bool();
    this->Attr("max_num_splits").AttrType(REQUIRED).Int();

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

OP_ADD(TurboQuantPagedAttention);
}  // namespace ops
