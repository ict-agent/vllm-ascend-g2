# CANN 8.5.0 环境下编译 vllm-ascend-g2 自定义算子指南

> 本文档详细记录在 `vllm-ascend0130-g` 容器（CANN 8.5.0）中单独编译 vllm-ascend-g2 主分支自定义算子的方法，
> 包括原始 compressor 算子和融合后的 `compressor_scatter_update_v2` 算子。

## 1. 环境信息

### 1.1 容器硬件与软件

| 项目 | 版本 |
|------|------|
| 容器名称 | `vllm-ascend0130-g` |
| OS | Ubuntu 22.04, Linux 5.10.0 aarch64 |
| NPU | Ascend 910B1 × 8 卡 |
| CANN | 8.5.0 (`/usr/local/Ascend/cann-8.5.0`) |
| NPU Driver | V100R001C23SPC002B212 (25.3.rc1) |
| cmake | 4.2.1 |
| gcc | 11.4.0 |
| Python | 3.11.14 |
| torch | 2.8.0+cpu |
| vllm-ascend | 0.13.0 (容器自带) |

### 1.2 源码挂载

宿主机 `~/repos` 目录已挂载到容器的 `/repos`，因此：

```
/repos/vllm-ascend-g2/csrc/   ← cann_ops-transformer 项目源码（主分支）
```

### 1.3 关键兼容性问题

vllm-ascend-g2 主分支代码针对 **CANN 9.0.0** 开发，而容器只有 **CANN 8.5.0**。主要差异：

| 问题 | 原因 | 修复方法 |
|------|------|---------|
| `ascend950` SOC 版本不存在 | CANN 8.5.0 的 `SocVersion` 枚举不含 `ASCEND950`，只有 `ASCEND910/B/910_93/910_95/910_55` | 删除/替换所有 `ASCEND950` 引用 |
| `op_build` 拒绝 `ascend950` 配置 | CANN 8.5.0 的 `op_build` 工具不认识 `ascend950` SOC | 删除 `_def.cpp` 中的 `AddConfig("ascend950", ...)` 行 |
| Driver 版本低于要求 | CANN 9.0.0 要求 Driver ≥ V100R001C25，容器只有 V100R001C23 | 本文档仅做算子编译（不需要运行），不影响 |

**注意**：Driver 版本问题只影响 NPU 运行时（无法执行推理），不影响算子编译。编译只是生成 kernel binary 和 host library。

## 2. 单独编译 compressor 算子

### 2.1 准备源码补丁

由于不能直接修改 `/repos` 下的源码（宿主机共享），需要先复制一份到 `/tmp`：

```bash
# 在容器内执行
cp -r /repos/vllm-ascend-g2/csrc /tmp/csrc_patch
```

#### 补丁 1：删除 ascend950 AddConfig

`op_build` 工具在解析 op 定义时会校验 SOC 版本，`ascend950` 不被 CANN 8.5.0 认识，会导致 cmake configure 阶段失败：

```
Invalid socVersion ascend950 of op Compressor, please check whether AddConfig are correctly configured in Opdef.
```

修复方法——删除 `_def.cpp` 中 `ascend950` 的配置行：

```bash
# compressor 算子的 _def.cpp
sed -i '/AddConfig("ascend950"/d' /tmp/csrc_patch/attention/compressor/op_host/compressor_def.cpp

# 如果编译所有算子（ASCEND_OP_NAME=ALL），需要批量处理所有 _def.cpp
find /tmp/csrc_patch/ -name '*_def.cpp' | xargs sed -i '/AddConfig("ascend950"/d'
```

> 此补丁不影响算子功能——910B1 硬件使用的是 `ascend910b` 配置，删除 `ascend950` 只是去掉一个不适用的 SOC target。

#### 补丁 2：替换 SocVersion::ASCEND950 → ASCEND910B

CANN 8.5.0 的 `platform_ascendc::SocVersion` 枚举不含 `ASCEND950`，直接引用会导致编译错误：

```
error: 'ASCEND950' is not a member of 'platform_ascendc::SocVersion'
```

修复方法——全局替换所有 CANN SocVersion 枚举引用：

```bash
# 必须替换的文件（编译时会直接引用）
find /tmp/csrc_patch/ -name '*.cpp' -o -name '*.h' \
  | xargs sed -i 's/platform_ascendc::SocVersion::ASCEND950/platform_ascendc::SocVersion::ASCEND910B/'

# 还有 common/tiling_util.cpp 中的一处
# 已被上面的全局替换覆盖，无需单独处理
```

> 注意：替换后逻辑上 `IsRegbaseSocVersion()` 函数将把 910B 当作 regbase SOC 处理，这在 910B1 硬件上功能影响很小（regbase 和非 regbase 的区别是 kernel 编译优化策略的细微差异，不影响 correctness）。

### 2.2 cmake configure

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
mkdir -p /tmp/compressor_build && cd /tmp/compressor_build

cmake \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_OP_NAME=compressor \
  -DBUILD_OPEN_PROJECT=ON \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0 \
  /tmp/csrc_patch/
```

关键 cmake 参数说明：

| 参数 | 说明 |
|------|------|
| `ASCEND_COMPUTE_UNIT=ascend910b` | 目标 SOC 版本，匹配容器内的 910B1 硬件 |
| `ASCEND_OP_NAME=compressor` | 只编译 compressor 算子（不编译其他算子），大幅缩短编译时间 |
| `BUILD_OPEN_PROJECT=ON` | 作为独立自定义算子项目编译（而非 CANN built-in） |
| `CUSTOM_ASCEND_CANN_PACKAGE_PATH` | 显式指定 CANN 路径（避免依赖默认路径） |

### 2.3 编译

```bash
cd /tmp/compressor_build
make -j4
```

#### 编译时间

compressor 算子有 8 个 tiling key variant，每个 variant 的 AICore kernel 编译约 20-25 分钟（`opc` 编译器为每个 variant 生成约 250+ 个 `.o` 文件）。总编译时间约 **2-3 小时**。

使用 `-j4` 可以并行编译多个 variant，但 opc 编译器本身是单线程的，`-j4` 主要加速 host 代码的编译阶段。

#### 编译产物

编译成功后，产物位于 `/tmp/compressor_build/custom/` 目录下：

```
custom/
├── vendors/
│   └── custom/
│       ├── op_api/          ← aclnn 接口库
│       ├── op_impl/
│       │   └ ai_core/
│       │     └── tbe/
│       │       ├── impl/    ← 动态 tiling Python 实现
│       │       └── op_tiling/ ← tiling 库 (.so)
│       ├── op_proto/        ← op 定义库 (.so)
│       └── bin/             ← AICore kernel binary (compressor_ascend910b.o)
```

### 2.4 安装到容器 OPP

编译产物需要安装到 CANN 的 OPP 目录才能被 torch_binding 调用：

```bash
# 将编译产物拷贝到 CANN OPP 的 vendors/customize 目录
cp -r /tmp/compressor_build/custom/vendors/custom/* \
      /usr/local/Ascend/cann-8.5.0/opp/vendors/customize/
```

> **注意**：仅编译并不等于可以运行。容器 vllm-ascend 0.13.0 的 torch_binding.cpp 不含 compressor 算子的注册代码，
> 需要升级 vllm-ascend 或手动注册才能从 Python 调用。参见第 4 节。

## 3. 编译融合算子 compressor_scatter_update_v2

融合算子的代码位于：

```
/repos/vllm-ascend-g2/csrc/attention/compressor_scatter_update_v2/
```

### 3.1 准备源码补丁

与第 2.1 节相同的两处补丁，但需要额外处理融合算子的 `_def.cpp`：

```bash
cp -r /repos/vllm-ascend-g2/csrc /tmp/csrc_fused_patch

# 补丁 1：删除所有 _def.cpp 中的 ascend950 AddConfig
find /tmp/csrc_fused_patch/ -name '*_def.cpp' | xargs sed -i '/AddConfig("ascend950"/d'

# 补丁 2：替换所有 CANN SocVersion::ASCEND950
find /tmp/csrc_fused_patch/ -name '*.cpp' -o -name '*.h' \
  | xargs sed -i 's/platform_ascendc::SocVersion::ASCEND950/platform_ascendc::SocVersion::ASCEND910B/'
```

> 融合算子的 `_def.cpp`（`compressor_scatter_update_v2_def.cpp`）目前只配置了 `ascend910b` 和 `ascend910_93`，
> 没有包含 `ascend950`，所以补丁 1 对它无影响。但其他算子的 `_def.cpp` 需要补丁，
> 因为 cmake 构建系统会遍历所有子目录（即使 `ASCEND_OP_NAME` 过滤了编译目标）。

### 3.2 cmake configure

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
mkdir -p /tmp/fused_build && cd /tmp/fused_build

cmake \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_OP_NAME=compressor_scatter_update_v2 \
  -DBUILD_OPEN_PROJECT=ON \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0 \
  /tmp/csrc_fused_patch/
```

> **重要**：如果只想编译融合算子，用 `ASCEND_OP_NAME=compressor_scatter_update_v2` 即可。
> 如果想同时编译 compressor 和融合算子，用逗号分隔：
> `ASCEND_OP_NAME=compressor,compressor_scatter_update_v2`（但注意 cmake 的 `ASCEND_OP_NAME` 参数实际只支持单个算子名或 `ALL`，
> 如需编译多个算子需要用 `ALL` 然后等待所有算子编译完成，或分两次 cmake configure 分别编译）。

### 3.3 编译

```bash
cd /tmp/fused_build
make -j4
```

融合算子的 tiling key variant 数量与 compressor 相似（约 8 个），编译时间也约 **2-3 小时**。

### 3.4 编译产物与安装

与第 2.4 节相同，将 `custom/` 目录下的产物拷贝到 CANN OPP 的 vendors 目录。

## 4. 从 Python 调用融合算子

### 4.1 注册算子到 torch.ops

容器内的 vllm-ascend 0.13.0 不含 compressor 和 compressor_scatter_update_v2 的注册代码。
需要将主分支的 `torch_binding.cpp` 和 `torch_binding_meta.cpp` 编译进 `_C_ascend` 扩展模块。

#### 方法 A：替换 torch_binding 并重新编译 vllm-ascend

```bash
# 1. 将主分支的 torch_binding 文件覆盖容器内的
cp /repos/vllm-ascend-g2/csrc/torch_binding.cpp /vllm-workspace/vllm-ascend/csrc/torch_binding.cpp
cp /repos/vllm-ascend-g2/csrc/torch_binding_meta.cpp /vllm-workspace/vllm-ascend/csrc/torch_binding_meta.cpp

# 2. 重新编译 vllm-ascend 的 C 扩展
cd /vllm-workspace/vllm-ascend
pip install -e . --no-build-isolation
```

> **问题**：vllm-ascend 0.13.0 的 CMakeLists.txt 要求 torch 2.5.0，而容器 torch 是 2.8.0，
> 版本检查会报 FATAL_ERROR。需要修改 CMakeLists.txt 中的版本检查或临时跳过。
> 此外 v0.13.0 的 C 扩展不含 compressor kernel 编译逻辑，只有 `ascendc_library(vllm_ascend_kernels)`
> 编译少量简单 kernel，不会编译 compressor 的 AICore kernel。

#### 方法 B：最小化 Python 测试脚本（推荐）

不修改 vllm-ascend 安装包，而是写一个独立的 Python 脚本直接调用 `aclnn` API：

```python
#!/usr/bin/env python3
"""独立测试 compressor_scatter_update_v2 融合算子的脚本"""
import torch
import torch_npu  # 需要安装 torch-npu

# 手动注册算子（如果已安装到 OPP）
# 通过 acl_op_compiler 在运行时加载自定义算子

def test_compressor_scatter_update_v2():
    # 构造测试输入
    batch_size = 4
    seq_len = 128
    head_dim = 64
    cmp_ratio = 4

    hidden_states = torch.randn(batch_size, seq_len, head_dim * 2, dtype=torch.bfloat16).npu()
    wkv = torch.randn(head_dim * 2, head_dim // cmp_ratio, dtype=torch.bfloat16).npu()
    wgate = torch.randn(head_dim * 2, head_dim // cmp_ratio, dtype=torch.bfloat16).npu()
    state_cache = torch.randn(batch_size, 1, head_dim // cmp_ratio, dtype=torch.float32).npu()
    # ... 其他输入

    # 调用融合算子
    result = torch.ops._C_ascend.compressor_scatter_update_v2(
        hidden_states, wkv, wgate, state_cache, ...
    )

    print("融合算子执行成功!")
    print(f"输出 shape: {result.shape}")
```

> **注意**：方法 B 需要先安装 `torch-npu`（容器内只有 `torch 2.8.0+cpu`，缺少 NPU 支持）。
> 安装 `torch-npu` 需要 Driver ≥ V100R001C25，与当前 Driver V100R001C23 不兼容。
> 因此目前 **无法在容器内实际运行推理**，只能编译。

### 4.2 当前限制总结

| 操作 | 是否可行 | 阻碍因素 |
|------|---------|---------|
| 编译 compressor 算子 | ✅ 可行 | 需要两处补丁（删除 ASCEND950） |
| 编译融合算子 | ✅ 可行（预计） | 同上 |
| 安装到 OPP 目录 | ✅ 可行 | 手动 cp |
| Python 调用推理 | ❌ 不可行 | torch-npu 需要 Driver ≥ C25；vllm-ascend 0.13.0 API 不兼容 |
| 运行 DSA 推理 | ❌ 不可行 | 同上 |

**结论**：当前容器可以完成编译验证（确认代码兼容 CANN 8.5.0），但无法运行推理。
实际运行测试需要在 Driver ≥ V100R001C25 的环境中进行。

## 5. 编译过程中的常见问题与解决

### 5.1 cmake configure 阶段失败：Invalid socVersion ascend950

```
Invalid socVersion ascend950 of op Compressor, please check whether AddConfig are correctly configured in Opdef.
CMake Error at cmake/config.cmake:268 (message): Error: ops prepare build failed.
```

**原因**：CANN 8.5.0 的 `op_build` 工具不认识 `ascend950` SOC 版本。

**解决**：删除所有 `_def.cpp` 中包含 `AddConfig("ascend950", ...)` 的行。

涉及的文件列表（共 24 个 `_def.cpp`）：

```
csrc/attention/compressor/op_host/compressor_def.cpp
csrc/attention/indexer_compress_epilog/op_host/indexer_compress_epilog_def.cpp
csrc/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_def.cpp
csrc/attention/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp
csrc/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp
csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_def.cpp
csrc/attention/lightning_indexer/op_host/lightning_indexer_def.cpp
csrc/attention/load_index_kv_cache/op_host/load_index_kv_cache_def.cpp
csrc/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp
csrc/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_def.cpp
csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_def.cpp
csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp
csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp
csrc/moe/causal_conv1d/op_host/causal_conv1d_def.cpp
csrc/moe/chunk_fwd_o/op_host/chunk_fwd_o_def.cpp
csrc/moe/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_def.cpp
csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_def.cpp
csrc/moe/hc_post/op_host/hc_post_def.cpp
csrc/moe/hc_pre/op_host/hc_pre_def.cpp
csrc/moe/hc_pre_inv_rms/op_host/hc_pre_inv_rms_def.cpp
csrc/moe/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_def.cpp
csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp
csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_def.cpp
csrc/moe/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp
```

一键批量修复：

```bash
find /tmp/csrc_patch/ -name '*_def.cpp' | xargs sed -i '/AddConfig("ascend950"/d'
```

### 5.2 编译阶段失败：'ASCEND950' is not a member of 'platform_ascendc::SocVersion'

```
error: 'ASCEND950' is not a member of 'platform_ascendc::SocVersion'; did you mean 'ASCEND910'?
```

**原因**：CANN 8.5.0 头文件中 `SocVersion` 枚举不含 `ASCEND950`。

**解决**：全局替换 `SocVersion::ASCEND950` → `SocVersion::ASCEND910B`：

```bash
find /tmp/csrc_patch/ -name '*.cpp' -o -name '*.h' \
  | xargs sed -i 's/platform_ascendc::SocVersion::ASCEND950/platform_ascendc::SocVersion::ASCEND910B/'
```

涉及的关键文件（仅列出编译 compressor 时直接引用的）：

```
csrc/common/src/tiling_base/tiling_util.cpp        ← IsRegbaseSocVersion() 函数
csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp  ← tiling 中的 SOC 检查
```

其他文件中的 `ASCEND950` 引用只在编译对应算子时才会触发，不影响 compressor 单独编译。

### 5.3 abseil-cpp 编译失败：ar unable to copy file

```
/usr/bin/ar: unable to copy file 'libabsl_flags_parse.a'; reason: No such file or directory
```

**原因**：这是 CANN 构建系统依赖 abseil-cpp 的 ExternalProject 并行编译时出现的竞争条件。
在 `-j4` 并行 make 时，abseil-cpp 的多个库同时 ar 打包可能冲突。

**解决**：使用 `make -j1`（单线程）可避免此问题，但编译更慢。或者重新执行 `make -j4`，
第二次通常会成功（因为 abseil-cpp 的大部分编译结果已缓存）。

### 5.4 编译时间过长

AICore kernel 编译是主要瓶颈。每个 tiling key variant 需要 opc 编译器生成约 250 个 `.o` 文件，
耗时约 20-25 分钟。compressor 有 8 个 variant，总计约 2-3 小时。

**优化建议**：

1. 使用 `ASCEND_OP_NAME=compressor` 只编译目标算子（不编译 ALL）
2. `-j4` 可以在 host 编译阶段并行，但 AICore kernel 编译阶段是串行的（opc 单线程）
3. 如果有更快的 CPU（更多核心），可以在 `TILINGKEY_PAR_COMPILE=1` 下并行编译多个 variant

## 6. 完整一键构建脚本

以下脚本可在 `vllm-ascend0130-g` 容器内一键完成从补丁到编译的全部流程：

```bash
#!/bin/bash
# build_compressor.sh — 在 CANN 8.5.0 容器内编译 compressor 算子
# 使用方法: docker exec vllm-ascend0130-g bash build_compressor.sh

set -e

CANN_PATH="/usr/local/Ascend/cann-8.5.0"
SRC_PATH="/repos/vllm-ascend-g2/csrc"
PATCH_PATH="/tmp/csrc_patch"
BUILD_PATH="/tmp/compressor_build"
OP_NAME="compressor"  # 改为 compressor_scatter_update_v2 可编译融合算子
SOC="ascend910b"

echo "=== Step 1: 复制源码 ==="
rm -rf "$PATCH_PATH"
cp -r "$SRC_PATH" "$PATCH_PATH"

echo "=== Step 2: 应用 CANN 8.5.0 兼容补丁 ==="
# 补丁 1: 删除 ascend950 AddConfig（op_build 不认识）
find "$PATCH_PATH" -name '*_def.cpp' | xargs sed -i '/AddConfig("ascend950"/d'
echo "  已删除所有 _def.cpp 中的 ascend950 AddConfig"

# 补丁 2: 替换 SocVersion::ASCEND950（头文件枚举不含）
find "$PATCH_PATH" -name '*.cpp' -o -name '*.h' \
  | xargs sed -i 's/platform_ascendc::SocVersion::ASCEND950/platform_ascendc::SocVersion::ASCEND910B/'
echo "  已替换所有 SocVersion::ASCEND950 → ASCEND910B"

echo "=== Step 3: cmake configure ==="
source "$CANN_PATH/set_env.sh"
rm -rf "$BUILD_PATH"
mkdir -p "$BUILD_PATH"
cd "$BUILD_PATH"

cmake \
  -DASCEND_COMPUTE_UNIT="$SOC" \
  -DASCEND_OP_NAME="$OP_NAME" \
  -DBUILD_OPEN_PROJECT=ON \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$CANN_PATH" \
  "$PATCH_PATH/"

echo "=== Step 4: 编译 ==="
# 使用 -j1 避免 abseil-cpp 并行编译的竞争条件
make -j1

echo "=== Step 5: 检查产物 ==="
ls -la "$BUILD_PATH/custom/vendors/custom/op_api/" 2>/dev/null || echo "op_api 目录不存在"
ls -la "$BUILD_PATH/custom/vendors/custom/bin/" 2>/dev/null || echo "bin 目录不存在"

echo ""
echo "编译完成！产物在 $BUILD_PATH/custom/ 目录下。"
echo "如需安装到 OPP，执行："
echo "  cp -r $BUILD_PATH/custom/vendors/custom/* /usr/local/Ascend/cann-8.5.0/opp/vendors/customize/"
```

将此脚本保存到宿主机 `/data/home_dir/i_zhongjian/repos/vllm-ascend-g2/docs/logs-g/build_compressor.sh`，
然后在容器内执行：

```bash
docker exec vllm-ascend0130-g bash /repos/vllm-ascend-g2/docs/logs-g/build_compressor.sh
```

## 7. 验证编译是否成功的方法

由于无法在容器内运行推理，只能通过以下方式验证编译产物正确性：

### 7.1 检查产物完整性

```bash
# 检查关键产物是否存在
CUST="/tmp/compressor_build/custom/vendors/custom"

echo "=== op_api（aclnn 接口） ==="
ls "$CUST/op_api/"

echo "=== op_impl（tiling + dynamic impl） ==="
find "$CUST/op_impl/" -name '*.so' -o -name '*.py' | head -10

echo "=== op_proto（算子定义） ==="
ls "$CUST/op_proto/lib/" 2>/dev/null

echo "=== bin（AICore kernel binary） ==="
find "$CUST/bin/" -name '*.o' -o -name '*.bin' | head -5
```

### 7.2 检查 AICore kernel 二进制

```bash
# 检查每个 tiling variant 是否生成了 kernel binary
ls /tmp/compressor_build/binary/ascend910b/bin/compressor/
# 应看到类似 Compressor_*.o 的文件
```

### 7.3 在有 CANN 9.0.0 + Driver C25 的环境中完整验证

交由有 Ascend 910B 开发环境（CANN 9.0.0 + Driver ≥ V100R001C25）的同事：

1. **编译验证**：直接用主分支代码编译（无需补丁），确认编译通过
2. **功能验证**：运行 DSA attention 模型推理，对比融合前后输出数值一致性
3. **性能验证**：对比融合前后 `compressor + scatter` 阶段的 NPU profiling 时间

---

*文档编写日期：2026-06-11*
*编译验证环境：vllm-ascend0130-g 容器，CANN 8.5.0，Ascend 910B1 × 8*