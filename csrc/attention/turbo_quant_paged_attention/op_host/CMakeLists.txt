# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        turbo_quant_paged_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME TurboQuantPagedAttention
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE turbo_quant_paged_attention ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
