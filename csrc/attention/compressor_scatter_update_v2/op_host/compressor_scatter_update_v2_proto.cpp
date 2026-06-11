/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
    // === Compressor input indices (0-11) ===
    constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
    constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;
    constexpr uint32_t APE_INPUT_INDEX = 4;
    constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 5;
    constexpr uint32_t ROPE_SIN_INPUT_INDEX = 6;
    constexpr uint32_t ROPE_COS_INPUT_INDEX = 7;
    constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 8;
    constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 9;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 10;
    constexpr uint32_t START_POS_INPUT_INDEX = 11;

    // === Scatter input indices (12-14) ===
    constexpr uint32_t SWA_KV_CACHE_INPUT_INDEX = 12;
    constexpr uint32_t SCATTER_INDICES_INPUT_INDEX = 13;
    constexpr uint32_t SCATTER_UPDATES_INPUT_INDEX = 14;

    // === ATTR indices ===
    // Compressor attrs
    constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 0;
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
    constexpr uint32_t COFF_ATTR_INDEX = 2;
    constexpr uint32_t NORM_EPS_ATTR_INDEX = 3;
    constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 4;
    constexpr uint32_t CACHE_MODE_ATTR_INDEX = 5;
    constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 6;
    // Scatter attrs
    constexpr uint32_t SCATTER_STRIDES_ATTR_INDEX = 7;
    constexpr uint32_t SCATTER_USE_LOCKING_ATTR_INDEX = 8;

    // === OUTPUT indices ===
    constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;
    constexpr uint32_t STATE_CACHE_OUTPUT_INDEX = 1;
    constexpr uint32_t SWA_KV_CACHE_OUTPUT_INDEX = 2;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMP_RATIO_VALUE = 4;
    constexpr uint32_t COFF_VALUE = 1;

struct CompressorScatterUpdateV2ProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t Sr { 0 };
    int64_t H { 0 };
    int64_t D { 0 };
};

constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetCompressorScatterUpdateV2ShapeDim(const gert::InferShapeContext* context,
                                                       CompressorScatterUpdateV2ProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED)
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, wkvShape, return ge::GRAPH_FAILED)
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, wgateShape, return ge::GRAPH_FAILED)
    auto stateCacheShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, stateCacheShape, return ge::GRAPH_FAILED)
    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, apeShape, return ge::GRAPH_FAILED)
    auto normWeightShape = context->GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, normWeightShape, return ge::GRAPH_FAILED)
    auto ropeSinShape = context->GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, ropeSinShape, return ge::GRAPH_FAILED)
    auto ropeCosShape = context->GetRequiredInputShape(ROPE_COS_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, ropeCosShape, return ge::GRAPH_FAILED)
    auto stateBlockTableShape = context->GetRequiredInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, stateBlockTableShape, return ge::GRAPH_FAILED)
    auto cuSeqlensShape = context->GetRequiredInputShape(CU_SEQ_LEN_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, cuSeqlensShape, return ge::GRAPH_FAILED)
    auto seqUsedShape = context->GetRequiredInputShape(SEQ_USED_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, seqUsedShape, return ge::GRAPH_FAILED)
    auto startPosShape = context->GetRequiredInputShape(START_POS_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, startPosShape, return ge::GRAPH_FAILED)

    // Scatter inputs
    auto swaKvCacheShape = context->GetRequiredInputShape(SWA_KV_CACHE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, swaKvCacheShape, return ge::GRAPH_FAILED)
    auto scatterIndicesShape = context->GetRequiredInputShape(SCATTER_INDICES_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, scatterIndicesShape, return ge::GRAPH_FAILED)
    auto scatterUpdatesShape = context->GetRequiredInputShape(SCATTER_UPDATES_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, scatterUpdatesShape, return ge::GRAPH_FAILED)

    if (xShape->GetDimNum() == DIM_NUM_3) {
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {
        shapeParam.isBsMerge = true;
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = normWeightShape->GetDim(DIM_INDEX_0);
    shapeParam.Sr = ropeSinShape->GetDim(DIM_INDEX_1);

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorScatterUpdateV2ShapeDim(
    const CompressorScatterUpdateV2ProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    // === Compressor output: cmp_kv ===
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, cmpKvShape, return ge::GRAPH_FAILED)
    auto attr = context->GetAttrs();
    const uint32_t *cmpRatioPtr = attr->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);
    uint32_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const uint32_t *coffPtr = attr->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    uint32_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;

    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3);
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.H);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2);
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.H);
    }

    // === Compressor output: state_cache (same shape as input) ===
    auto stateCacheInShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    auto stateCacheOutShape = context->GetOutputShape(STATE_CACHE_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, stateCacheOutShape, return ge::GRAPH_FAILED)
    *stateCacheOutShape = *stateCacheInShape;

    // === Scatter output: swa_kv_cache (same shape as input, in-place) ===
    auto swaKvCacheInShape = context->GetRequiredInputShape(SWA_KV_CACHE_INPUT_INDEX);
    auto swaKvCacheOutShape = context->GetOutputShape(SWA_KV_CACHE_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, swaKvCacheOutShape, return ge::GRAPH_FAILED)
    *swaKvCacheOutShape = *swaKvCacheInShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressorScatterUpdateV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter CompressorScatterUpdateV2 inferDataType impl.");

    // cmp_kv output type follows x input type
    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));

    // state_cache output type follows state_cache input type
    context->SetOutputDataType(STATE_CACHE_OUTPUT_INDEX, context->GetRequiredInputDataType(STATE_CACHE_INPUT_INDEX));

    // swa_kv_cache output type follows swa_kv_cache input type (in-place)
    context->SetOutputDataType(SWA_KV_CACHE_OUTPUT_INDEX, context->GetRequiredInputDataType(SWA_KV_CACHE_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressorScatterUpdateV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("CompressorScatterUpdateV2", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter CompressorScatterUpdateV2 infershape impl.");

    CompressorScatterUpdateV2ProtoShapeParam shapeParam {};
    auto apiRet = GetCompressorScatterUpdateV2ShapeDim(context, shapeParam);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context get input shape failed");

    apiRet = SetCompressorScatterUpdateV2ShapeDim(shapeParam, context);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context set output shape failed");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompressorScatterUpdateV2)
    .InferShape(InferShapeCompressorScatterUpdateV2)
    .InferDataType(InferDataTypeCompressorScatterUpdateV2);
}  // namespace ops