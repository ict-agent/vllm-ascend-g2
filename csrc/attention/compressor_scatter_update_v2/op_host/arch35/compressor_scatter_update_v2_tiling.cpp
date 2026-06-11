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
 * \file compressor_scatter_update_v2_tiling.cpp
 * \brief Tiling implementation for CompressorScatterUpdateV2 on arch35
 *
 * Uses GetTilingData<T>() pattern (like standalone compressor) for direct
 * plain struct access, avoiding BEGIN_TILING_DATA_DEF layout mismatch issues.
 */

#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "compressor_scatter_update_v2_tiling.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus CompressorScatterUpdateV2Tiling::GetNpuInfo()
{
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "num of core obtained is 0."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorScatterUpdateV2Tiling::ComputeCompressorTiling()
{
    // Reuse the compressor's tiling logic by creating a CompressorContext
    CompressorContext compressorContext;
    auto ret = CompressorTiling::ConvertContext(*tilingContext_, compressorContext);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Convert compressor context failed."),
                return ge::GRAPH_FAILED);

    // Create a CompressorTilingData and populate it
    CompressorTilingData compressorTilingData;
    CompressorTiling compressorTiling(&compressorContext);
    ret = compressorTiling.RunBigKernelTiling(&compressorTilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Compressor tiling failed."),
                return ge::GRAPH_FAILED);

    // Copy compressor tiling data into the fused plain struct
    // using direct member access (same pattern as standalone compressor)
    auto& base = tilingData_->compressorBaseParams;
    auto& cmpBase = compressorTilingData.baseParams;

    base.batchSize = cmpBase.batchSize;
    base.seqSize = cmpBase.seqSize;
    base.hiddenSize = cmpBase.hiddenSize;
    base.tokenSize = cmpBase.tokenSize;
    base.headDim = cmpBase.headDim;
    base.ropeHeadDim = cmpBase.ropeHeadDim;
    base.csSize = cmpBase.csSize;
    base.cmpRatio = cmpBase.cmpRatio;
    base.cgSize = cmpBase.cgSize;
    base.normEps = cmpBase.normEps;
    base.reciprocalD = cmpBase.reciprocalD;
    base.usedCoreNum = cmpBase.usedCoreNum;
    base.nSize = cmpBase.nSize;
    base.stateCacheStrideDim0 = cmpBase.stateCacheStrideDim0;
    base.kBaseNum = cmpBase.kBaseNum;
    base.kBaseSize = cmpBase.kBaseSize;
    base.coreGroupNum = cmpBase.coreGroupNum;
    base.mLoopNum = cmpBase.mLoopNum;

    // Flatten splitCoreParam struct array into 6 separate arrays
    for (uint32_t i = 0; i < CMP_SU_MAX_AIC_CORE_NUM && i < CMP_MAX_AIC_CORE_NUM; ++i) {
        base.splitCoreParamMStart[i] = cmpBase.splitCoreParam[i].mStart;
        base.splitCoreParamMEnd[i] = cmpBase.splitCoreParam[i].mEnd;
        base.splitCoreParamNStart[i] = cmpBase.splitCoreParam[i].nStart;
        base.splitCoreParamNEnd[i] = cmpBase.splitCoreParam[i].nEnd;
        base.splitCoreParamKStart[i] = cmpBase.splitCoreParam[i].kStart;
        base.splitCoreParamKEnd[i] = cmpBase.splitCoreParam[i].kEnd;
    }

    auto& pa = tilingData_->compressorPageAttentionParams;
    auto& cmpPA = compressorTilingData.pageAttentionParams;
    pa.blockNum = cmpPA.blockNum;
    pa.blockSize = cmpPA.blockSize;
    pa.maxBlockNumPerBatch = cmpPA.maxBlockNumPerBatch;

    auto& is = tilingData_->compressorInnerSplitParams;
    auto& cmpIS = compressorTilingData.innerSplitParams;
    is.mBaseSize = cmpIS.mBaseSize;
    is.dBaseSize = cmpIS.dBaseSize;

    auto& ws = tilingData_->compressorWorkspaceParams;
    auto& cmpWS = compressorTilingData.workspaceParams;
    ws.mm1KvResSize = cmpWS.mm1KvResSize;
    ws.mm1ScoreResSize = cmpWS.mm1ScoreResSize;
    ws.vec1ResSize = cmpWS.vec1ResSize;
    ws.vec1TailCacheSize = cmpWS.vec1TailCacheSize;
    ws.dbWorkspaceRatio = cmpWS.dbWorkspaceRatio;

    // Store compressor tiling key and block dim from compressor context
    tilingData_->compressorTilingKey = static_cast<uint32_t>(compressorContext.tilingKey);
    tilingData_->blockDim = compressorContext.blockDim;

    // Store workspace size from compressor
    if (compressorContext.workSpaces) {
        workspaceSize_ = compressorContext.workSpaces[0];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorScatterUpdateV2Tiling::ComputeScatterTiling()
{
    auto scatterIndicesShape = tilingContext_->GetInputShape(SCATTER_INDICES_INPUT_INDEX);
    auto scatterUpdatesShape = tilingContext_->GetInputShape(SCATTER_UPDATES_INPUT_INDEX);
    auto swaKvCacheShape = tilingContext_->GetInputShape(SWA_KV_CACHE_INPUT_INDEX);

    OP_CHECK_IF(scatterIndicesShape == nullptr || scatterUpdatesShape == nullptr || swaKvCacheShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Scatter input shapes are nullptr."),
                return ge::GRAPH_FAILED);

    // Get scatter data type size
    auto scatterDtype = tilingContext_->GetInputDesc(SCATTER_UPDATES_INPUT_INDEX)->GetDataType();
    switch (scatterDtype) {
        case ge::DT_BF16:   scatterDataTypeSize_ = 2; break;
        case ge::DT_FLOAT16: scatterDataTypeSize_ = 2; break;
        case ge::DT_FLOAT:  scatterDataTypeSize_ = 4; break;
        default:
            OP_LOGE("CompressorScatterUpdateV2", "Unsupported scatter data type: %d", scatterDtype);
            return ge::GRAPH_FAILED;
    }

    auto indicesStorageShape = scatterIndicesShape->GetStorageShape();
    auto updatesStorageShape = scatterUpdatesShape->GetStorageShape();
    auto varStorageShape = swaKvCacheShape->GetStorageShape();

    scatterTotalRow_ = indicesStorageShape.GetShapeSize();

    scatterLength_ = 1;
    if (updatesStorageShape.GetDimNum() > 1) {
        for (uint64_t i = 1; i < updatesStorageShape.GetDimNum(); ++i) {
            scatterLength_ *= updatesStorageShape.GetDim(i);
        }
    }

    auto attrs = tilingContext_->GetAttrs();
    auto stridesPtr = attrs->GetListInt(SCATTER_STRIDES_ATTR_INDEX);

    // Direct member access (plain struct)
    auto& scatter = tilingData_->scatterTiling;
    scatter.scatterTotalRow = scatterTotalRow_;
    scatter.scatterLength = scatterLength_;

    uint64_t scatterAlignNum = CSU_ALIGNED_SIZE / scatterDataTypeSize_;
    scatter.scatterAlignLength = (scatterLength_ + scatterAlignNum - 1) & ~(scatterAlignNum - 1);
    scatter.scatterDataTypeSize = scatterDataTypeSize_;

    scatter.scatterIndexDim = varStorageShape.GetDimNum();
    if (stridesPtr != nullptr) {
        for (uint64_t i = 0; i < scatter.scatterIndexDim && i < CSU_MAX_DIM_NUM; ++i) {
            scatter.scatterStrides[i] = static_cast<uint64_t>(stridesPtr->GetData()[i]);
        }
    }

    uint64_t ubAvailableForUpdates = ubSize_ / 2;
    uint64_t maxUpdateTileElements = ubAvailableForUpdates / scatterDataTypeSize_;
    scatter.scatterTileLength = std::min(scatterLength_, maxUpdateTileElements);
    if (scatter.scatterTileLength == 0) scatter.scatterTileLength = 1;
    scatter.scatterTileNum = (scatterLength_ + scatter.scatterTileLength - 1) / scatter.scatterTileLength;
    scatter.scatterTileTail = scatterLength_ - (scatter.scatterTileNum - 1) * scatter.scatterTileLength;
    scatter.scatterTileAlignLength = (scatter.scatterTileLength + scatterAlignNum - 1) & ~(scatterAlignNum - 1);

    uint64_t scatterCoreNum = std::min(static_cast<uint64_t>(aivNum_), scatterTotalRow_);
    scatterCoreNum = scatterCoreNum == 0 ? 1 : scatterCoreNum;
    scatter.scatterCoreNum = scatterCoreNum;

    uint64_t scatterTailRow = scatterTotalRow_ / scatterCoreNum;
    uint64_t scatterFrontRow = scatterTailRow + 1;
    uint64_t scatterFrontNum = scatterTotalRow_ % scatterCoreNum;
    uint64_t scatterTailNum = scatterTailRow == 0 ? 0 : scatterCoreNum - scatterFrontNum;

    scatter.scatterFrontNum = scatterFrontNum;
    scatter.scatterFrontRow = scatterFrontRow;
    scatter.scatterTailRow = scatterTailRow;
    scatter.scatterTailNum = scatterTailNum;

    OP_LOGD(tilingContext_, "Scatter tiling: totalRow=%lu, scatterLength=%lu, coreNum=%lu, tileNum=%lu",
            scatterTotalRow_, scatterLength_, scatterCoreNum, scatter.scatterTileNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorScatterUpdateV2Tiling::CalcWorkSpace()
{
    size_t totalWorkspace = workspaceSize_;

    auto* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkSpace == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Workspace sizes pointer is nullptr."),
                return ge::GRAPH_FAILED);
    currentWorkSpace[0] = totalWorkspace;

    OP_LOGI("CompressorScatterUpdateV2", "Total workspace: %zu", totalWorkspace);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorScatterUpdateV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling initing");
    auto compileInfo = static_cast<const CompressorScatterUpdateV2CompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Compile info is nullptr."),
                return ge::GRAPH_FAILED);

    // Get direct struct access to the tiling buffer (like standalone compressor)
    tilingData_ = tilingContext_->GetTilingData<CompressorScatterUpdateV2TilingData>();
    OP_CHECK_IF(tilingData_ == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "TilingData is nullptr."),
                return ge::GRAPH_FAILED);

    ge::graphStatus ret = GetNpuInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "GetNpuInfo failed."),
                return ge::GRAPH_FAILED);

    ret = ComputeCompressorTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "ComputeCompressorTiling failed."),
                return ge::GRAPH_FAILED);

    ret = ComputeScatterTiling();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "ComputeScatterTiling failed."),
                return ge::GRAPH_FAILED);

    ret = CalcWorkSpace();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "CalcWorkSpace failed."),
                return ge::GRAPH_FAILED);

    OP_LOGD(tilingContext_, "Tiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorScatterUpdateV2Tiling::SetKernelTiling()
{
    // Set block dim and tiling key directly (like standalone compressor)
    tilingContext_->SetBlockDim(tilingData_->blockDim);
    tilingContext_->SetTilingKey(tilingData_->compressorTilingKey);

    OP_LOGD(tilingContext_, "CompressorScatterUpdateV2 tiling: key=%u, blockDim=%u",
            tilingData_->compressorTilingKey, tilingData_->blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingCompressorScatterUpdateV2(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("CompressorScatterUpdateV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for CompressorScatterUpdateV2 start.");
    CompressorScatterUpdateV2Tiling tilingOp(context);
    if (tilingOp.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Tiling init fail");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Tiling for CompressorScatterUpdateV2 end.");
    return tilingOp.SetKernelTiling();
}

ge::graphStatus TilingPrepareCompressorScatterUpdateV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For CompressorScatterUpdateV2 start.");
    auto compileInfo = context->GetCompiledInfo<CompressorScatterUpdateV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAic();
    if (compileInfo->core_num == 0) {
        OP_LOGE(context, "aicCoreNum %ld", compileInfo->core_num);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ubSizePlatForm is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "Tiling Prepare For CompressorScatterUpdateV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompressorScatterUpdateV2)
    .Tiling(TilingCompressorScatterUpdateV2)
    .TilingParse<CompressorScatterUpdateV2CompileInfo>(TilingPrepareCompressorScatterUpdateV2);

} // namespace optiling