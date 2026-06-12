#!/bin/bash
# build_compressor.sh — 在 CANN 8.5.0 容器内编译 compressor / compressor_scatter_update_v2 算子
# 使用方法:
#   docker exec vllm-ascend0130-g bash /repos/vllm-ascend-g2/docs/logs-g/build_compressor.sh
#   或指定融合算子:
#   docker exec vllm-ascend0130-g bash /repos/vllm-ascend-g2/docs/logs-g/build_compressor.sh compressor_scatter_update_v2

set -e

# ─── 配置参数 ──────────────────────────────────────────────────
CANN_PATH="/usr/local/Ascend/cann-8.5.0"
SRC_PATH="/repos/vllm-ascend-g2/csrc"
PATCH_PATH="/tmp/csrc_patch"
BUILD_PATH="/tmp/compressor_build"
OP_NAME="${1:-compressor}"          # 默认编译 compressor，传参可改为 compressor_scatter_update_v2
SOC="ascend910b"
MAKE_JOBS=1                          # 使用 -j1 避免 abseil-cpp 并行竞争

echo "╔══════════════════════════════════════════════════════╗"
echo "║  CANN 8.5.0 算子编译脚本                            ║"
echo "║  目标算子: $OP_NAME                                 ║"
echo "║  目标 SOC:  $SOC                                    ║"
echo "╚══════════════════════════════════════════════════════╝"

# ─── Step 1: 复制源码 ─────────────────────────────────────────
echo "[1/5] 复制源码到 $PATCH_PATH ..."
rm -rf "$PATCH_PATH"
cp -r "$SRC_PATH" "$PATCH_PATH"
echo "  ✓ 完成"

# ─── Step 2: 应用 CANN 8.5.0 兼容补丁 ─────────────────────────
echo "[2/5] 应用 CANN 8.5.0 兼容补丁 ..."

# 补丁 A: 删除所有 _def.cpp 中的 ascend950 AddConfig
#   CANN 8.5.0 的 op_build 工具不认识 ascend950 SOC 版本
DEF_COUNT=$(find "$PATCH_PATH" -name '*_def.cpp' | wc -l)
find "$PATCH_PATH" -name '*_def.cpp' | xargs sed -i '/AddConfig("ascend950"/d'
echo "  ✓ 补丁 A: 已处理 $DEF_COUNT 个 _def.cpp，删除 ascend950 AddConfig"

# 补丁 B: 替换 SocVersion::ASCEND950 → ASCEND910B
#   CANN 8.5.0 头文件 platform_ascendc::SocVersion 枚举不含 ASCEND950
SV_COUNT=$(find "$PATCH_PATH" -name '*.cpp' -o -name '*.h' \
  | xargs grep -l 'SocVersion::ASCEND950' 2>/dev/null | wc -l)
find "$PATCH_PATH" -name '*.cpp' -o -name '*.h' \
  | xargs sed -i 's/platform_ascendc::SocVersion::ASCEND950/platform_ascendc::SocVersion::ASCEND910B/'
echo "  ✓ 补丁 B: 已处理 $SV_COUNT 个文件，替换 SocVersion::ASCEND950 → ASCEND910B"

# ─── Step 3: cmake configure ───────────────────────────────────
echo "[3/5] cmake configure ..."
source "$CANN_PATH/set_env.sh"
rm -rf "$BUILD_PATH"
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"

cmake \
  -DASCEND_COMPUTE_UNIT="$SOC" \
  -DASCEND_OP_NAME="$OP_NAME" \
  -DBUILD_OPEN_PROJECT=ON \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$CANN_PATH" \
  "$PATCH_PATH/" 2>&1 | tail -5

if [ ! -f "$BUILD_PATH/Makefile" ]; then
  echo "  ✗ cmake configure 失败！请检查上方的错误信息。"
  exit 1
fi
echo "  ✓ cmake configure 成功"

# ─── Step 4: 编译 ──────────────────────────────────────────────
echo "[4/5] 开始编译 (make -j$MAKE_JOBS) ..."
echo "  ⏱ 预计耗时 2-3 小时（每个 tiling variant 约 20-25 分钟）"
echo "  可用 'tail -f /tmp/compressor_build_log.txt' 查看进度"

make -j"$MAKE_JOBS" > /tmp/compressor_build_log.txt 2>&1
MAKE_EXIT=$?

if [ $MAKE_EXIT -ne 0 ]; then
  echo "  ✗ 编译失败！错误日志："
  tail -30 /tmp/compressor_build_log.txt
  exit 1
fi
echo "  ✓ 编译成功"

# ─── Step 5: 检查产物 ──────────────────────────────────────────
echo "[5/5] 检查产物 ..."
CUST="$BUILD_PATH/custom/vendors/custom"

echo ""
echo "产物目录: $CUST"
echo "──────────────────────────────────────"
echo "op_api   (aclnn 接口):"
ls "$CUST/op_api/" 2>/dev/null && echo "" || echo "  (不存在)"
echo "op_impl  (tiling + impl):"
find "$CUST/op_impl/" -name '*.so' -o -name '*.py' 2>/dev/null | head -10 && echo "" || echo "  (不存在)"
echo "op_proto (算子定义):"
ls "$CUST/op_proto/lib/" 2>/dev/null && echo "" || echo "  (不存在)"
echo "bin      (kernel binary):"
find "$CUST/bin/" -name '*.o' -o -name '*.bin' 2>/dev/null | head -5 && echo "" || echo "  (不存在)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  编译完成！                                          ║"
echo "║                                                      ║"
echo "║  如需安装到 OPP 目录，执行:                           ║"
echo "║  cp -r $CUST/* \$ASCEND_OPP_PATH/vendors/customize/  ║"
echo "║                                                      ║"
echo "║  注意: 当前容器无法运行推理（Driver 版本限制）         ║"
echo "║  运行测试需 CANN 9.0.0 + Driver ≥ V100R001C25 环境   ║"
echo "╚══════════════════════════════════════════════════════╝"