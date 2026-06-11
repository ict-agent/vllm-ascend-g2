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
 * \brief Tiling header for CompressorScatterUpdateV2 on arch32
 *        Similar to arch35 but includes ROPE_DTYPE template parameter
 */

#ifndef COMPRESSOR_SCATTER_UPDATE_V2_TILING_H_ARCH32
#define COMPRESSOR_SCATTER_UPDATE_V2_TILING_H_ARCH32

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

// Reuse compressor's arch32 tiling structures
#include "../../../compressor/op_kernel/arch32/compressor_template_tiling_key.h"
#include "../../../compressor/op_kernel/arch32/compressor_tiling_data.h"
#include "../../../compressor/op_host/arch32/compressor_tiling.h"

#include "platform/platform_infos_def.h"

#ifdef ASCENDC_OP_TEST
#define CSU_EXTERN_C extern "C"
#else
#define CSU_EXTERN_C
#endif

namespace optiling {

// === Scatter input indices (same as arch35) ===
constexpr uint32_t SWA_KV_CACHE_INPUT_INDEX = 12;
constexpr uint32_t SCATTER_INDICES_INPUT_INDEX = 13;
constexpr uint32_t SCATTER_UPDATES_INPUT_INDEX = 14;

// === Scatter ATTR indices ===
constexpr uint32_t SCATTER_STRIDES_ATTR_INDEX = 7;
constexpr uint32_t SCATTER_USE_LOCKING_ATTR_INDEX = 8;

// === OUTPUT indices ===
constexpr uint32_t STATE_CACHE_OUTPUT_INDEX = 1;
constexpr uint32_t SWA_KV_CACHE_OUTPUT_INDEX = 2;

constexpr uint64_t CSU_ALIGNED_SIZE = 32;
constexpr uint64_t CSU_ALIGNED_NUM = 8;

CSU_EXTERN_C ge::graphStatus TilingCompressorScatterUpdateV2(gert::TilingContext *context);

// CompressorScatterUpdateV2CompileInfo is defined in compressor_scatter_update_v2_tiling_data.h
// (included via the arch35 tiling.h chain). No need to redefine here.

// The tiling class CompressorScatterUpdateV2Tiling is defined in the arch35 tiling.h,
// which is included when the arch32 tiling.cpp includes the arch35 tiling.cpp.

} // namespace optiling
#endif // COMPRESSOR_SCATTER_UPDATE_V2_TILING_H_ARCH32