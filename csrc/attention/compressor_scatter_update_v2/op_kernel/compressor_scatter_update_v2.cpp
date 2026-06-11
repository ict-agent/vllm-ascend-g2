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
 * \file compressor_scatter_update_v2.cpp
 * \brief Fused Compressor + ScatterUpdate kernel entry point
 *
 * This kernel combines:
 *   Phase 1: Scatter update (write KV into swa_kv_cache at positions from slot_mapping)
 *   Phase 2: Compressor computation (compress hidden_states → compressed_kv + state_cache update)
 *
 * The scatter writes' discrete memory access latency is masked by the compressor pipeline.
 * AIV cores start scatter writes; AIC cores begin compressor MM1 after initial scatter phase.
 */

// Include compressor kernel headers from the original compressor directory
// These must be available at compile time - for standalone builds, copy the compressor/op_kernel/ directory
#if (__CCE_AICORE__ == 220)
#include "../../compressor/op_kernel/arch32/compressor_kernel.h"
#include "../../compressor/op_kernel/arch32/compressor_kernel_perf.h"
#else
#include "../../compressor/op_kernel/arch35/compressor_kernel.h"
#include "../../compressor/op_kernel/arch35/compressor_kernel_full_load.h"
#endif

#include "scatter_update_phase.h"
#include "compressor_scatter_update_v2_tiling_data.h"

using namespace Compressor;
using namespace CompressorScatterUpdateV2;

// Helper: construct CompressorTilingData from the fused tiling data.
// CompressorScatterUpdateV2TilingData is a plain struct (kernel-side) that uses
// flattened splitCoreParam arrays. We copy from the fused data into CompressorTilingData
// which uses the struct array format that the compressor kernel expects.
// NOTE: arch32 CompressorBaseParams has fewer fields than arch35 (no kBaseNum,
// kBaseSize, coreGroupNum, mLoopNum, splitCoreParam), so the copy is conditional.
__aicore__ inline void BuildCompressorTilingData(
    const optiling::CompressorScatterUpdateV2TilingData& fusedData,
    optiling::CompressorTilingData& cmpData)
{
    // Copy CompressorBaseParams scalar fields (shared by both arch32 and arch35)
    auto& base = fusedData.compressorBaseParams;
    cmpData.baseParams.batchSize = base.batchSize;
    cmpData.baseParams.seqSize = base.seqSize;
    cmpData.baseParams.hiddenSize = base.hiddenSize;
    cmpData.baseParams.tokenSize = base.tokenSize;
    cmpData.baseParams.headDim = base.headDim;
    cmpData.baseParams.ropeHeadDim = base.ropeHeadDim;
    cmpData.baseParams.csSize = base.csSize;
    cmpData.baseParams.cmpRatio = base.cmpRatio;
    cmpData.baseParams.cgSize = base.cgSize;
    cmpData.baseParams.normEps = base.normEps;
    cmpData.baseParams.reciprocalD = base.reciprocalD;
    cmpData.baseParams.usedCoreNum = base.usedCoreNum;
    cmpData.baseParams.nSize = base.nSize;
    cmpData.baseParams.stateCacheStrideDim0 = base.stateCacheStrideDim0;

    // arch35-only fields: kBaseNum, kBaseSize, coreGroupNum, mLoopNum, splitCoreParam
    // (arch32 CompressorBaseParams does not have these fields)
#if (__CCE_AICORE__ != 220)
    cmpData.baseParams.kBaseNum = base.kBaseNum;
    cmpData.baseParams.kBaseSize = base.kBaseSize;
    cmpData.baseParams.coreGroupNum = base.coreGroupNum;
    cmpData.baseParams.mLoopNum = base.mLoopNum;

    // Copy flattened splitCoreParam arrays into struct array format
    for (uint32_t i = 0; i < CMP_MAX_AIC_CORE_NUM && i < optiling::CMP_SU_MAX_AIC_CORE_NUM; ++i) {
        cmpData.baseParams.splitCoreParam[i].mStart = base.splitCoreParamMStart[i];
        cmpData.baseParams.splitCoreParam[i].mEnd = base.splitCoreParamMEnd[i];
        cmpData.baseParams.splitCoreParam[i].nStart = base.splitCoreParamNStart[i];
        cmpData.baseParams.splitCoreParam[i].nEnd = base.splitCoreParamNEnd[i];
        cmpData.baseParams.splitCoreParam[i].kStart = base.splitCoreParamKStart[i];
        cmpData.baseParams.splitCoreParam[i].kEnd = base.splitCoreParamKEnd[i];
    }
#endif

    // Copy CompressorPageAttentionParams
    auto& pa = fusedData.compressorPageAttentionParams;
    cmpData.pageAttentionParams.blockNum = pa.blockNum;
    cmpData.pageAttentionParams.blockSize = pa.blockSize;
    cmpData.pageAttentionParams.maxBlockNumPerBatch = pa.maxBlockNumPerBatch;

    // Copy CompressorInnerSplitParams
    auto& is = fusedData.compressorInnerSplitParams;
    cmpData.innerSplitParams.mBaseSize = is.mBaseSize;
    cmpData.innerSplitParams.dBaseSize = is.dBaseSize;

    // Copy CompressorWorkspaceParams
    auto& ws = fusedData.compressorWorkspaceParams;
    cmpData.workspaceParams.mm1KvResSize = ws.mm1KvResSize;
    cmpData.workspaceParams.mm1ScoreResSize = ws.mm1ScoreResSize;
    cmpData.workspaceParams.vec1ResSize = ws.vec1ResSize;
    cmpData.workspaceParams.vec1TailCacheSize = ws.vec1TailCacheSize;
    cmpData.workspaceParams.dbWorkspaceRatio = ws.dbWorkspaceRatio;
}

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                               \
        optiling::CompressorTilingData cmpTilingData;                                                                  \
        BuildCompressorTilingData(*tilingData, cmpTilingData);                                                          \
        const optiling::CompressorTilingData *__restrict cmpTilingPtr = &cmpTilingData;                                \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, cmpTilingPtr);                                                 \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,  \
                cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);                                                    \
        op.Process();                                                                                                  \
    } while (0)

#if (__CCE_AICORE__ == 220)
template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId, uint8_t RopeDType>
#else
template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId>
#endif
__global__ __aicore__ void compressor_scatter_update_v2(
    // === Compressor inputs (0-11) ===
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    // === Scatter inputs (12-14) ===
    __gm__ uint8_t *swaKvCache,
    __gm__ uint8_t *scatterIndices,
    __gm__ uint8_t *scatterUpdates,
    // === Compressor outputs (0-1) ===
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    // === Scatter output (2) ===
    __gm__ uint8_t *swaKvCacheOut,
    // === Workspace + tiling ===
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorScatterUpdateV2TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorScatterUpdateV2TilingData, tilingDataIn, tiling);
    const optiling::CompressorScatterUpdateV2TilingData *__restrict tilingData = &tilingDataIn;

    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        // Even with empty x, still need to perform scatter update
        if ASCEND_IS_AIV {
            TPipe scatterPipe;
            constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
            using X_T = typename AscendC::Conditional<xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
            ScatterUpdatePhase<X_T> scatterOp(swaKvCache, scatterIndices, scatterUpdates, *tilingData, scatterPipe);
            scatterOp.Process();
        }
        return;
    }

    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
#if (__CCE_AICORE__ == 220)
    constexpr auto ropeDtype = static_cast<ROPE_DTYPE>(RopeDType);
#endif
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto rotaryMode = static_cast<ROTARY_MODE>(RotaryMode);
#if (__CCE_AICORE__ != 220)
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
#endif

    // ======== Phase 1: Scatter Update (AIV cores) ========
    // AIV cores perform scatter writes to swa_kv_cache.
    // AIC cores wait for initial scatter results to complete before starting compressor.
    // This overlap helps hide the scatter's discrete memory access latency.
    if ASCEND_IS_AIV {
        TPipe scatterPipe;
        using X_T = typename AscendC::Conditional<xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
        ScatterUpdatePhase<X_T> scatterOp(swaKvCache, scatterIndices, scatterUpdates, *tilingData, scatterPipe);
        scatterOp.Process();
    }

    // ======== Phase 2: Compressor Computation (AIC + AIV) ========
    // Run compressor computation using the original compressor kernel logic.
    // We construct CompressorTilingData from the fused tiling data's individual fields
    // We construct CompressorTilingData from the fused tiling data's individual fields
    // because the fused data uses flattened splitCoreParam arrays while the compressor
    // expects a struct array.
#if (__CCE_AICORE__ == 220)
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::PERF) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, xLayout, xDtype, ropeDtype, coff, rotaryMode);
    } else {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, ropeDtype, coff, rotaryMode);
    }
#else
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelFullLoad, xLayout, xDtype, coff, rotaryMode, cacheMode);
    } else {
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode, cacheMode);
    }
#endif
}