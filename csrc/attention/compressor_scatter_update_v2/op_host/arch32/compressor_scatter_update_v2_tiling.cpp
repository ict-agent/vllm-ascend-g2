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
 * \file compressor_scatter_update_v2_tiling.cpp (arch32)
 * \brief Identical to arch35 tiling logic; the arch-specific differences are
 *        handled by reusing the arch32 CompressorTiling class.
 */

// This file reuses the same tiling logic as arch35.
// The only arch-specific difference is the compressor tiling, which is handled
// by including the arch32 compressor_tiling.h header in the .h file above.
// The scatter tiling logic is identical for both arch32 and arch35.

// Since the tiling code is identical (the scatter logic doesn't depend on arch,
// and the compressor tiling is handled via CompressorTiling::ConvertContext which
// adapts to the correct arch), we include the arch35 implementation directly.

#include "../arch35/compressor_scatter_update_v2_tiling.cpp"