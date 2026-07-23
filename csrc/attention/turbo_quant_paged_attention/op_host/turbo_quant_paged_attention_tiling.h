/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef TURBO_QUANT_PAGED_ATTENTION_TILING_H
#define TURBO_QUANT_PAGED_ATTENTION_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TurboQuantPagedAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numQueryHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, numBlocks);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxPages);
TILING_DATA_FIELD_DEF(uint32_t, slotSize);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, numSplits);
TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, coreWorkspaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, partialAccumOffset);
TILING_DATA_FIELD_DEF(uint64_t, partialSumOffset);
TILING_DATA_FIELD_DEF(uint64_t, partialMaxOffset);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TurboQuantPagedAttention, TurboQuantPagedAttentionTilingData)
}  // namespace optiling

#endif  // TURBO_QUANT_PAGED_ATTENTION_TILING_H
