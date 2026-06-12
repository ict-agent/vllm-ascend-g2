#!/bin/bash
# Merge CompressorScatterUpdateV2 into the existing vllm-ascend OPP structure
# This approach avoids the problem of the standalone proto library registering 0 ops
# by merging into the existing vllm-ascend proto library which already works.

set -e

VLLM_OPP="/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
FUSED_BUILD="/tmp/fused_build"
FUSED_OPP="/tmp/opp_fused/vendors/fused-op"
SOC="ascend910b"

echo "=== Merging CompressorScatterUpdateV2 into vllm-ascend OPP ==="

# 1. Copy kernel binaries (.o + .json)
echo "[1/5] Copying kernel binaries..."
mkdir -p "$VLLM_OPP/op_impl/ai_core/tbe/kernel/$SOC/compressor_scatter_update_v2"
cp "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2/"*.o \
   "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2/"*.json \
   "$VLLM_OPP/op_impl/ai_core/tbe/kernel/$SOC/compressor_scatter_update_v2/"

# 2. Copy kernel config JSON (variant selector)
echo "[2/5] Copying kernel config..."
cp "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2.json" \
   "$VLLM_OPP/op_impl/ai_core/tbe/kernel/config/$SOC/"

# 3. Merge ops-info.json
echo "[3/5] Merging ops-info.json..."
python3 << 'PYEOF'
import json

vllm_ops_info_path = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"
fused_ops_info_path = "/tmp/opp_fused/vendors/fused-op/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"

with open(vllm_ops_info_path) as f:
    existing = json.load(f)

with open(fused_ops_info_path) as f:
    new = json.load(f)

for key, value in new.items():
    existing[key] = value

with open(vllm_ops_info_path, 'w') as f:
    json.dump(existing, f, indent=4)

print(f"Merged ops-info. Now has {len(existing)} ops: {list(existing.keys())}")
PYEOF

# 4. Merge binary_info_config.json
echo "[4/5] Merging binary_info_config.json..."
python3 << 'PYEOF'
import json

vllm_bic_path = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json"
fused_bic_path = "/tmp/fused_build/binary/ascend910b/bin/binary_info_config.json"

with open(vllm_bic_path) as f:
    existing = json.load(f)

with open(fused_bic_path) as f:
    new = json.load(f)

for key, value in new.items():
    existing[key] = value

with open(vllm_bic_path, 'w') as f:
    json.dump(existing, f, indent=4)

print(f"Merged binary_info_config. Now has {len(existing)} ops: {list(existing.keys())}")
PYEOF

# 5. Add our proto + tiling library paths to LD_LIBRARY_PATH
# (the existing vllm-ascend proto library already works, so we just need to
# add our tiling library which contains the CompressorScatterUpdateV2 tiling)
echo "[5/5] Note: tiling library must be loadable"
echo "  The fused libcust_opmaster_rt2.0.so includes tiling for CompressorScatterUpdateV2"
echo "  This needs to be loadable alongside vllm-ascend's tiling library"
echo ""
echo "To use, set LD_LIBRARY_PATH to include the fused tiling lib:"
echo "  export LD_LIBRARY_PATH=$FUSED_OPP/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/:${LD_LIBRARY_PATH}"

echo ""
echo "=== Merge complete ==="
echo "Kernel files in vllm-ascend OPP:"
ls -la "$VLLM_OPP/op_impl/ai_core/tbe/kernel/$SOC/compressor_scatter_update_v2/" | head -20