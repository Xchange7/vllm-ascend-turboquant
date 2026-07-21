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

#ifndef TURBOQUANT_PAGED_DEQUANT_TILING_H
#define TURBOQUANT_PAGED_DEQUANT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TurboQuantPagedDequantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, maxPages);
TILING_DATA_FIELD_DEF(uint32_t, activePages);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, slotSize);
TILING_DATA_FIELD_DEF(uint32_t, keyBits);
TILING_DATA_FIELD_DEF(uint32_t, keyDataBytes);
TILING_DATA_FIELD_DEF(uint32_t, keyPackedSize);
TILING_DATA_FIELD_DEF(uint32_t, valueBits);
TILING_DATA_FIELD_DEF(uint32_t, valueDataBytes);
TILING_DATA_FIELD_DEF(uint32_t, centroidCount);
TILING_DATA_FIELD_DEF(uint32_t, normCorrection);
TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantPagedDequant, TurboQuantPagedDequantTilingData)
}  // namespace optiling

#endif  // TURBOQUANT_PAGED_DEQUANT_TILING_H
