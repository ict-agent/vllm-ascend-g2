#!/bin/bash
# ============================================================================
# register_fused_op_opp.sh — Install CompressorScatterUpdateV2 OPP artifacts
#                            into the CANN runtime for functional testing
#
# Usage:
#   Inside the vllm-ascend0130-g container:
#   bash register_fused_op_opp.sh [/path/to/fused_build]
#
#   Default fused_build path: /tmp/fused_build
#
# What this script does:
#   1. Creates a new OPP vendor directory at /tmp/opp_fused/
#   2. Copies proto lib, tiling lib, op_api lib from the fused build
#   3. Copies kernel binaries (.o + .json) for ascend910b
#   4. Generates aic-ascend910b-ops-info.json for CompressorScatterUpdateV2
#   5. Generates binary_info_config.json for the kernel variant selector
#   6. Sets ASCEND_CUSTOM_OPP_PATH so CANN can find the new op
#   7. Adds LD_LIBRARY_PATH for the op_api shared library
#
# After running this script, you can test the fused op with:
#   python3 test_fused_op_acl.py
# ============================================================================

set -e

FUSED_BUILD="${1:-/tmp/fused_build}"
OPP_DST="/tmp/opp_fused/vendors/fused-op"
SOC="ascend910b"

echo "=== Registering CompressorScatterUpdateV2 OPP ==="
echo "Fused build dir: $FUSED_BUILD"
echo "OPP destination: $OPP_DST"

# ---- 1. Create directory structure ----
echo "[1/7] Creating OPP directory structure..."
mkdir -p "$OPP_DST/op_proto/lib/linux/aarch64"
mkdir -p "$OPP_DST/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
mkdir -p "$OPP_DST/op_impl/ai_core/tbe/config/$SOC"
mkdir -p "$OPP_DST/op_impl/ai_core/tbe/kernel/$SOC/compressor_scatter_update_v2"
mkdir -p "$OPP_DST/op_impl/ai_core/tbe/kernel/config/$SOC"
mkdir -p "$OPP_DST/op_api/lib"
mkdir -p "$OPP_DST/op_api/include"
mkdir -p "$OPP_DST/bin"

# ---- 2. Copy proto library ----
echo "[2/7] Copying proto library..."
cp "$FUSED_BUILD/libcust_opsproto_rt2.0.so" "$OPP_DST/op_proto/lib/linux/aarch64/"

# ---- 3. Copy tiling library ----
echo "[3/7] Copying tiling library..."
cp "$FUSED_BUILD/libcust_opmaster_rt2.0.so" "$OPP_DST/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/"

# ---- 4. Copy op_api library ----
echo "[4/7] Copying op_api library..."
cp "$FUSED_BUILD/libcust_opapi.so" "$OPP_DST/op_api/lib/"

# ---- 5. Copy kernel binaries (.o + .json) ----
echo "[5/7] Copying kernel binaries..."
cp "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2/"*.o \
   "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2/"*.json \
   "$OPP_DST/op_impl/ai_core/tbe/kernel/$SOC/compressor_scatter_update_v2/"

# ---- 6. Generate ops-info.json ----
echo "[6/7] Generating ops-info.json..."
python3 -c '
import json

# This JSON tells CANN about the op signature — which inputs/outputs/attrs
# it expects. It must match the op definition in _def.cpp exactly.
ops_info = {
    "CompressorScatterUpdateV2": {
        "attr": {
            "list": "rope_head_dim,cmp_ratio,coff,norm_eps,rotary_mode,cache_mode,state_cache_stride_dim0,scatter_strides,scatter_use_locking"
        },
        "attr_rope_head_dim": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_cmp_ratio": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_coff": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_norm_eps": {
            "defaultValue": "0.0",
            "paramType": "required",
            "type": "float",
            "value": "all"
        },
        "attr_rotary_mode": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_cache_mode": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_state_cache_stride_dim0": {
            "defaultValue": "0",
            "paramType": "required",
            "type": "int",
            "value": "all"
        },
        "attr_scatter_strides": {
            "paramType": "optional",
            "type": "list_int",
            "value": "all"
        },
        "attr_scatter_use_locking": {
            "defaultValue": "false",
            "paramType": "optional",
            "type": "bool",
            "value": "all"
        },
        "coreType": {"value": "AiCore"},
        "dynamicCompileStatic": {"flag": "true"},
        "dynamicFormat": {"flag": "true"},
        "dynamicRankSupport": {"flag": "true"},
        "dynamicShapeSupport": {"flag": "true"},
        # 15 inputs (indices 0-14)
        "input0": {"dtype": "float16,bfloat16", "format": "ND,ND", "name": "x",       "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input1": {"dtype": "float16,bfloat16", "format": "ND,ND", "name": "wkv",      "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input2": {"dtype": "float16,bfloat16", "format": "ND,ND", "name": "wgate",     "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input3": {"dtype": "float32,float32",   "format": "ND,ND", "name": "state_cache","paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input4": {"dtype": "float32,float32",   "format": "ND,ND", "name": "ape",      "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input5": {"dtype": "float16,bfloat16,float32", "format": "ND,ND,ND", "name": "norm_weight", "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND,ND"},
        "input6": {"dtype": "float16,bfloat16,float32", "format": "ND,ND,ND", "name": "rope_sin", "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND,ND"},
        "input7": {"dtype": "float16,bfloat16,float32", "format": "ND,ND,ND", "name": "rope_cos", "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND,ND"},
        "input8":  {"dtype": "int32,int32",       "format": "ND,ND", "name": "state_block_table", "paramType": "optional", "shape": "all", "unknownshape_format": "ND,ND"},
        "input9":  {"dtype": "int32,int32",       "format": "ND,ND", "name": "cu_seqlens",       "paramType": "optional", "shape": "all", "unknownshape_format": "ND,ND"},
        "input10": {"dtype": "int32,int32",       "format": "ND,ND", "name": "seqused",          "paramType": "optional", "shape": "all", "unknownshape_format": "ND,ND"},
        "input11": {"dtype": "int32,int32",       "format": "ND,ND", "name": "start_pos",        "paramType": "optional", "shape": "all", "unknownshape_format": "ND,ND"},
        "input12": {"dtype": "float16,bfloat16",  "format": "ND,ND", "name": "swa_kv_cache",     "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input13": {"dtype": "int32,int32",        "format": "ND,ND", "name": "scatter_indices",  "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "input14": {"dtype": "float16,bfloat16",  "format": "ND,ND", "name": "scatter_updates",  "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        # 3 outputs (indices 0-2)
        "output0": {"dtype": "float16,bfloat16",  "format": "ND,ND", "name": "cmp_kv",          "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "output1": {"dtype": "float32,float32",    "format": "ND,ND", "name": "state_cache",     "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "output2": {"dtype": "float16,bfloat16",   "format": "ND,ND", "name": "swa_kv_cache",    "paramType": "required", "shape": "all", "unknownshape_format": "ND,ND"},
        "needCheckSupport": {"flag": "false"},
        "opFile":    {"value": "compressor_scatter_update_v2"},
        "opInterface":{"value": "CompressorScatterUpdateV2"},
        "prebuildPattern": {"value": "Opaque"},
        "precision_reduce": {"flag": "true"}
    }
}

with open("/tmp/opp_fused/vendors/fused-op/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json", "w") as f:
    json.dump(ops_info, f, indent=4)
print("Written ops-info.json")
'

# ---- 7. Generate binary_info_config.json ----
echo "[7/7] Generating binary_info_config.json..."
# Copy the pre-built binary config from the fused build
cp "$FUSED_BUILD/binary/$SOC/bin/compressor_scatter_update_v2.json" \
   "$OPP_DST/op_impl/ai_core/tbe/kernel/config/$SOC/"

# Copy binary_info_config.json
cp "$FUSED_BUILD/binary/$SOC/bin/binary_info_config.json" \
   "$OPP_DST/op_impl/ai_core/tbe/kernel/config/$SOC/"

# Copy relocatable_kernel_info_config.json if it exists
if [ -f "$FUSED_BUILD/binary/$SOC/bin/relocatable_kernel_info_config.json" ]; then
    cp "$FUSED_BUILD/binary/$SOC/bin/relocatable_kernel_info_config.json" \
       "$OPP_DST/op_impl/ai_core/tbe/kernel/config/$SOC/"
fi

# Copy version.info
cp "$FUSED_BUILD/version.info" "$OPP_DST/" 2>/dev/null || true

# ---- Create set_env.bash ----
cat > "$OPP_DST/bin/set_env.bash" << 'EOF'
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/opp_fused/vendors/fused-op:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/tmp/opp_fused/vendors/fused-op/op_api/lib/:${LD_LIBRARY_PATH}
EOF
chmod +x "$OPP_DST/bin/set_env.bash"

echo ""
echo "=== Registration complete ==="
echo "OPP vendor dir: $OPP_DST"
echo ""
echo "Directory structure:"
find "$OPP_DST" -type f | sort | head -30
echo ""
echo "To activate, run inside the container:"
echo "  source /tmp/opp_fused/vendors/fused-op/bin/set_env.bash"
echo ""
echo "Then test with:"
echo "  python3 test_fused_op_acl.py"