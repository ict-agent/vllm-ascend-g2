/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_scatter_update_v2_tiling.h
 * \brief Tiling header for CompressorScatterUpdateV2 on arch35
 *
 * Uses the same pattern as the standalone compressor: GetTilingData<T>()
 * for direct plain struct access, avoiding BEGIN_TILING_DATA_DEF layout
 * mismatch issues with the kernel-side plain struct.
 */

#ifndef COMPRESSOR_SCATTER_UPDATE_V2_TILING_H
#define COMPRESSOR_SCATTER_UPDATE_V2_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <sstream>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/compressor_scatter_update_v2_tiling_data.h"
#include "platform/platform_info.h"

// Reuse compressor's tiling key definitions and data structures
#include "../../../compressor/op_kernel/arch35/compressor_template_tiling_key.h"
#include "../../../compressor/op_kernel/arch35/compressor_tiling_data.h"
#include "../../../compressor/op_host/arch35/compressor_tiling.h"

#include "platform/platform_infos_def.h"

#ifdef ASCENDC_OP_TEST
#define CSU_EXTERN_C extern "C"
#else
#define CSU_EXTERN_C
#endif

namespace optiling {

// === Compressor input indices (reused from compressor_tiling.h) ===
// TOKEN_X_INPUT_INDEX, WEIGHT_KV_INPUT_INDEX, etc. are already defined in
// the included compressor_tiling.h. We only define the scatter-specific indices.

// === Scatter input indices ===
constexpr uint32_t SWA_KV_CACHE_INPUT_INDEX = 12;
constexpr uint32_t SCATTER_INDICES_INPUT_INDEX = 13;
constexpr uint32_t SCATTER_UPDATES_INPUT_INDEX = 14;

// === ATTR indices (compressor attrs reused from compressor_tiling.h, scatter attrs new) ===
constexpr uint32_t SCATTER_STRIDES_ATTR_INDEX = 7;
constexpr uint32_t SCATTER_USE_LOCKING_ATTR_INDEX = 8;

// === OUTPUT indices ===
constexpr uint32_t STATE_CACHE_OUTPUT_INDEX = 1;
constexpr uint32_t SWA_KV_CACHE_OUTPUT_INDEX = 2;

constexpr uint64_t CSU_ALIGNED_SIZE = 32;
constexpr uint64_t CSU_ALIGNED_NUM = 8;

CSU_EXTERN_C ge::graphStatus TilingCompressorScatterUpdateV2(gert::TilingContext *context);

class CompressorScatterUpdateV2Tiling {
public:
    explicit CompressorScatterUpdateV2Tiling(gert::TilingContext* context) : tilingContext_(context) {}
    ~CompressorScatterUpdateV2Tiling() = default;

    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus ComputeCompressorTiling();
    ge::graphStatus ComputeScatterTiling();
    ge::graphStatus CalcWorkSpace();

    CompressorScatterUpdateV2TilingData *tilingData_ = nullptr;
    gert::TilingContext* tilingContext_ = nullptr;

    // NPU info
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;

    // Scatter-specific params
    uint64_t scatterDataTypeSize_ = 0;
    uint64_t scatterTotalRow_ = 0;    // total number of rows in scatter_updates
    uint64_t scatterLength_ = 0;      // length per scatter row (kv_dim)
};

} // namespace optiling
#endif // COMPRESSOR_SCATTER_UPDATE_V2_TILING_H