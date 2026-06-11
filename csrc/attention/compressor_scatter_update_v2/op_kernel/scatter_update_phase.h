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
 * \file scatter_update_phase.h
 * \brief Simplified scatter update phase for CompressorScatterUpdateV2
 *        Supports 1D INT32 indices (slot_mapping style), BF16/FP16 updates, NoSort mode
 *        Designed to overlap scatter writes with compressor computation via pipeline
 */

#ifndef SCATTER_UPDATE_PHASE_H
#define SCATTER_UPDATE_PHASE_H

#include "kernel_operator.h"
#include "compressor_scatter_update_v2_tiling_data.h"

namespace CompressorScatterUpdateV2 {

using namespace AscendC;

constexpr uint64_t DOUBLE_BUFFER = 1;
constexpr uint64_t ALIGNED_SIZE = 32;
constexpr uint64_t ALIGN_NUM = 8;

__aicore__ inline void PipeMte2ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

__aicore__ inline void PipeMte3ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

__aicore__ inline void PipeVToMte3()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
}

__aicore__ inline void CalcBlockDistribution(
    uint64_t blockIdx, uint64_t frontNum, uint64_t frontRow, uint64_t tailRow,
    uint64_t& computeRow, uint64_t& start)
{
    if (blockIdx >= frontNum) {
        computeRow = tailRow;
        start = frontNum * frontRow + (blockIdx - frontNum) * computeRow;
    } else {
        computeRow = frontRow;
        start = blockIdx * computeRow;
    }
}

template<typename T>
class ScatterUpdatePhase {
public:
    __aicore__ inline ScatterUpdatePhase() = delete;
    __aicore__ inline ScatterUpdatePhase(
        GM_ADDR swaKvCache, GM_ADDR scatterIndices, GM_ADDR scatterUpdates,
        const optiling::CompressorScatterUpdateV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(swaKvCache, scatterIndices, scatterUpdates, tiling);
    }

    __aicore__ inline void InitParams(const optiling::CompressorScatterUpdateV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        auto& scatter = tiling.scatterTiling;
        CalcBlockDistribution(blockIdx_, scatter.scatterFrontNum, scatter.scatterFrontRow,
                              scatter.scatterTailRow, computeRow_, start_);
        end_ = start_ + computeRow_;

        scatterLength_ = scatter.scatterLength;
        scatterAlignLength_ = scatter.scatterAlignLength;
        scatterIndexDim_ = scatter.scatterIndexDim;
        dataTypeSize_ = scatter.scatterDataTypeSize;

        scatterTileNum_ = scatter.scatterTileNum;
        scatterTileLength_ = scatter.scatterTileLength;
        scatterTileTail_ = scatter.scatterTileTail;
        scatterTileAlignLength_ = scatter.scatterTileAlignLength;

        // Copy strides
        for (uint64_t i = 0; i < scatterIndexDim_; ++i) {
            scatterStrides_[i] = scatter.scatterStrides[i];
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        uint64_t updateBufBytes = scatterTileAlignLength_ * dataTypeSize_;
        pipe.InitBuffer(indexBuf_, DOUBLE_BUFFER * sizeof(int));
        pipe.InitBuffer(updateBuf_, updateBufBytes);
        pipe.InitBuffer(outputBuf_, updateBufBytes);

        indexLocal_ = indexBuf_.Get<int>();
        updateLocal_ = updateBuf_.Get<T>();
        outputLocal_ = outputBuf_.Get<T>();
    }

    __aicore__ inline void SetGmAddr(GM_ADDR swaKvCache, GM_ADDR scatterIndices, GM_ADDR scatterUpdates,
                                       const optiling::CompressorScatterUpdateV2TilingData& tiling)
    {
        indicesGm_.SetGlobalBuffer((__gm__ int*)scatterIndices);
        updatesGm_.SetGlobalBuffer((__gm__ T*)scatterUpdates);
        outputGm_.SetGlobalBuffer((__gm__ T*)swaKvCache);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t rowIdx = start_; rowIdx < end_; ++rowIdx) {
            // Read index
            int32_t linearIndex = indicesGm_.GetValue(rowIdx);
            if (linearIndex < 0) continue;  // Skip padding indices (-1)

            ProcessOneRow(rowIdx, static_cast<uint64_t>(linearIndex));
        }
    }

    __aicore__ inline void ProcessOneRow(uint64_t rowIdx, uint64_t linearIndex)
    {
        for (uint64_t tileIdx = 0; tileIdx < scatterTileNum_; ++tileIdx) {
            uint64_t tileLength = (tileIdx == scatterTileNum_ - 1) ? scatterTileTail_ : scatterTileLength_;

            // Read update data from GM
            uint64_t gmUpdateOffset = rowIdx * scatterLength_ + tileIdx * scatterTileLength_;
            DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * dataTypeSize_), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(updateLocal_, updatesGm_[gmUpdateOffset], updateCopyParams, padParams);
            PipeMte2ToS();

            // Write to output at computed offset
            uint64_t outOffset = linearIndex + tileIdx * scatterTileLength_;
            DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * dataTypeSize_), 0, 0, 0};
            DataCopyPad(outputGm_[outOffset], updateLocal_, outParams);
            PipeMte3ToS();
        }
    }

private:
    GlobalTensor<int> indicesGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> updateBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;

    LocalTensor<int> indexLocal_;
    LocalTensor<T> updateLocal_;
    LocalTensor<T> outputLocal_;

    uint64_t blockIdx_;
    uint64_t computeRow_;
    uint64_t start_;
    uint64_t end_;
    uint64_t scatterLength_;
    uint64_t scatterAlignLength_;
    uint64_t scatterIndexDim_;
    uint64_t dataTypeSize_;
    uint64_t scatterTileNum_;
    uint64_t scatterTileLength_;
    uint64_t scatterTileTail_;
    uint64_t scatterTileAlignLength_;
    uint64_t scatterStrides_[optiling::CSU_MAX_DIM_NUM] = {0};
};

} // namespace CompressorScatterUpdateV2

#endif // SCATTER_UPDATE_PHASE_H