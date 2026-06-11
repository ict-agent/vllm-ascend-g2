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
 * \file compressor_scatter_update_v2_tiling_data.h
 * \brief Kernel-side tiling data for CompressorScatterUpdateV2 operator
 *
 * This file uses plain C++ structs (not BEGIN_TILING_DATA_DEF macros)
 * because the AscendC kernel compiler does not support CANN host-side macros.
 * The host-side version (with BEGIN_TILING_DATA_DEF) is in
 * compressor_scatter_update_v2_tiling_data_host.h, included only by op_host files.
 *
 * Both versions must have the same field layout for the raw tiling buffer compatibility.
 * The splitCoreParam struct array is flattened into 6 separate uint32_t arrays
 * (matching the host-side BEGIN_TILING_DATA_DEF layout).
 */

#ifndef COMPRESSOR_SCATTER_UPDATE_V2_TILING_DATA_H
#define COMPRESSOR_SCATTER_UPDATE_V2_TILING_DATA_H
#include <cstdint>

namespace optiling {

constexpr uint32_t CMP_SU_MAX_AIC_CORE_NUM = 36;
constexpr uint64_t CSU_MAX_DIM_NUM = 8;

// === Compressor sub-structs (flattened splitCoreParam matching host-side layout) ===

struct CompressorScatterUpdateV2BaseParams {
    uint32_t batchSize = 0;
    uint32_t seqSize = 0;
    uint32_t hiddenSize = 0;
    uint32_t tokenSize = 0;
    uint32_t headDim = 0;
    uint32_t ropeHeadDim = 64;
    uint32_t csSize = 0;
    uint32_t cmpRatio = 4;
    uint32_t cgSize = 0;
    float normEps = 1e-6f;
    float reciprocalD = 0;
    uint32_t usedCoreNum = 0;
    uint32_t nSize = 0;
    uint64_t stateCacheStrideDim0 = 0;
    uint32_t kBaseNum = 0;
    uint32_t kBaseSize = 0;
    uint32_t coreGroupNum = 0;
    uint32_t mLoopNum = 0;
    // Flattened splitCoreParam: 6 uint32_t arrays matching host-side TILING_DATA_FIELD_DEF_ARR layout
    uint32_t splitCoreParamMStart[CMP_SU_MAX_AIC_CORE_NUM];
    uint32_t splitCoreParamMEnd[CMP_SU_MAX_AIC_CORE_NUM];
    uint32_t splitCoreParamNStart[CMP_SU_MAX_AIC_CORE_NUM];
    uint32_t splitCoreParamNEnd[CMP_SU_MAX_AIC_CORE_NUM];
    uint32_t splitCoreParamKStart[CMP_SU_MAX_AIC_CORE_NUM];
    uint32_t splitCoreParamKEnd[CMP_SU_MAX_AIC_CORE_NUM];
};

struct CompressorScatterUpdateV2PageAttentionParams {
    uint32_t blockNum = 0;
    uint32_t blockSize = 1;
    uint32_t maxBlockNumPerBatch = 1;
};

struct CompressorScatterUpdateV2InnerSplitParams {
    uint32_t mBaseSize;
    uint32_t dBaseSize;
};

struct CompressorScatterUpdateV2WorkspaceParams {
    uint32_t mm1KvResSize;
    uint32_t mm1ScoreResSize;
    uint32_t vec1ResSize;
    uint32_t vec1TailCacheSize;
    uint32_t dbWorkspaceRatio = 1;
};

// === Scatter tiling data ===

struct CompressorScatterUpdateV2ScatterTiling {
    uint64_t scatterTotalRow = 0;
    uint64_t scatterLength = 0;
    uint64_t scatterAlignLength = 0;
    uint64_t scatterFrontNum = 0;
    uint64_t scatterFrontRow = 0;
    uint64_t scatterTailRow = 0;
    uint64_t scatterTailNum = 0;
    uint64_t scatterCoreNum = 0;
    uint64_t scatterIndexDim = 0;
    uint64_t scatterStrides[CSU_MAX_DIM_NUM] = {0};
    uint64_t scatterDataTypeSize = 0;
    uint64_t scatterTileNum = 0;
    uint64_t scatterTileLength = 0;
    uint64_t scatterTileTail = 0;
    uint64_t scatterTileAlignLength = 0;
};

// === Main fused tiling data (plain struct for kernel side) ===

struct CompressorScatterUpdateV2TilingData {
    CompressorScatterUpdateV2BaseParams compressorBaseParams;
    CompressorScatterUpdateV2PageAttentionParams compressorPageAttentionParams;
    CompressorScatterUpdateV2InnerSplitParams compressorInnerSplitParams;
    CompressorScatterUpdateV2WorkspaceParams compressorWorkspaceParams;
    CompressorScatterUpdateV2ScatterTiling scatterTiling;
    uint32_t compressorTilingKey = 0;
    uint32_t blockDim = 0;
};

struct CompressorScatterUpdateV2CompileInfo {
    int64_t core_num;
    uint64_t ubSizePlatForm;
};

} // namespace optiling
#endif // COMPRESSOR_SCATTER_UPDATE_V2_TILING_DATA_H