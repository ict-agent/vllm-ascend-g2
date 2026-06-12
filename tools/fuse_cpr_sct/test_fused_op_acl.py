#!/usr/bin/env python3
"""
test_fused_op_acl.py — Functional test for CompressorScatterUpdateV2 via ACL runtime API

This script tests the fused CompressorScatterUpdateV2 operator by:
1. Initializing ACL runtime and setting device
2. Creating input/output tensor descriptions matching the op's expected signature
3. Allocating device memory for inputs and outputs
4. Setting operator attributes
5. Calling acl.op.execute_v2 to run the operator
6. Copying outputs back to host and verifying results

Prerequisites:
  - Run register_fused_op_opp.sh first to install OPP artifacts
  - Run inside the vllm-ascend0130-g container

NOTE: This is a MINIMAL smoke test. It uses small shapes to verify the
operator can be dispatched and executed without crashing. Numerical accuracy
verification requires matching the exact compressor model weights and input
data, which is better done at the torch_binding level.
"""

import os
import sys
import ctypes
import numpy as np

# Ensure OPP path is set before importing acl
OPP_PATH = "/tmp/opp_fused/vendors/fused-op"
current_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
if OPP_PATH not in current_opp:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{OPP_PATH}:{current_opp}"
    print(f"Set ASCEND_CUSTOM_OPP_PATH to include {OPP_PATH}")

LD_PATH = "/tmp/opp_fused/vendors/fused-op/op_api/lib/"
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if LD_PATH not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{LD_PATH}:{current_ld}"
    print(f"Set LD_LIBRARY_PATH to include {LD_PATH}")

import acl

# ACL constants (C enum values)
ACL_DT_UNDEFINED = -1
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT32 = 3
ACL_BF16 = 27
ACL_FORMAT_ND = 2
ACL_SUCCESS = 0

# aclrtMemcpyKind enum values
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# aclrtMemMallocPolicy enum values
ACL_MEM_MALLOC_HUGE_FIRST = 0


def check_ret(ret, msg):
    if ret != ACL_SUCCESS:
        print(f"ERROR: {msg} returned {ret}")
        sys.exit(1)
    else:
        print(f"OK: {msg}")


def upload_to_device(dev_ptr, np_array):
    """Upload a numpy array to device memory via host pinned buffer."""
    size = np_array.nbytes
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_SUCCESS:
        print(f"ERROR: malloc_host({size}) returned {ret}")
        return ret
    # Write numpy data into host pinned memory via ctypes
    ctypes.memmove(host_ptr, np_array.tobytes(), size)
    # Copy host pinned to device
    ret = acl.rt.memcpy(dev_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE)
    acl.rt.free_host(host_ptr)
    return ret


def download_from_device(dev_ptr, size, dtype, shape):
    """Download data from device memory to a numpy array."""
    host_ptr, ret = acl.rt.malloc_host(size)
    if ret != ACL_SUCCESS:
        print(f"ERROR: malloc_host({size}) returned {ret}")
        return None
    ret = acl.rt.memcpy(host_ptr, size, dev_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != ACL_SUCCESS:
        print(f"ERROR: memcpy device->host returned {ret}")
        acl.rt.free_host(host_ptr)
        return None
    # Read host pinned memory via ctypes
    result_bytes = ctypes.string_at(host_ptr, size)
    result_arr = np.frombuffer(result_bytes, dtype=dtype).reshape(shape).copy()
    acl.rt.free_host(host_ptr)
    return result_arr


def main():
    print("=" * 60)
    print("CompressorScatterUpdateV2 ACL Functional Test")
    print("=" * 60)

    # ---- Initialize ACL ----
    ret = acl.init()
    check_ret(ret, "acl.init")

    ret = acl.rt.set_device(0)
    check_ret(ret, "acl.rt.set_device(0)")

    # ---- Define test dimensions (float16 variant) ----
    BATCH = 1
    HIDDEN_DIM = 512      # Must be multiple of 16 for alignment
    CMP_KV_DIM = 64       # compressed kv dim = hidden_dim / cmp_ratio
    CMP_RATIO = 8         # hidden_dim / cmp_kv_dim
    ROPE_DIM = 64         # rope_head_dim
    SEQ_LEN = 1           # number of tokens

    # ---- Create input data (numpy arrays) ----
    print("\n[1] Creating input data...")
    x_data = np.random.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float16) * 0.01
    wkv_data = np.random.randn(HIDDEN_DIM, CMP_KV_DIM).astype(np.float16) * 0.01
    wgate_data = np.random.randn(HIDDEN_DIM, CMP_KV_DIM).astype(np.float16) * 0.01
    state_cache_data = np.random.randn(BATCH, HIDDEN_DIM).astype(np.float32) * 0.01
    ape_data = np.zeros((BATCH, HIDDEN_DIM), dtype=np.float32)
    norm_weight_data = np.ones(HIDDEN_DIM, dtype=np.float16)
    rope_sin_data = np.random.randn(SEQ_LEN, ROPE_DIM).astype(np.float16) * 0.01
    rope_cos_data = np.random.randn(SEQ_LEN, ROPE_DIM).astype(np.float16) * 0.01
    swa_kv_cache_data = np.zeros((1, CMP_KV_DIM), dtype=np.float16)
    scatter_indices_data = np.array([0], dtype=np.int32)
    scatter_updates_data = np.random.randn(1, CMP_KV_DIM).astype(np.float16) * 0.01

    # ---- Create tensor descriptors ----
    print("\n[2] Creating tensor descriptors...")
    input_descs = [
        acl.create_tensor_desc(ACL_FLOAT16, [SEQ_LEN, HIDDEN_DIM], ACL_FORMAT_ND),        # 0: x
        acl.create_tensor_desc(ACL_FLOAT16, [HIDDEN_DIM, CMP_KV_DIM], ACL_FORMAT_ND),     # 1: wkv
        acl.create_tensor_desc(ACL_FLOAT16, [HIDDEN_DIM, CMP_KV_DIM], ACL_FORMAT_ND),     # 2: wgate
        acl.create_tensor_desc(ACL_FLOAT,   [BATCH, HIDDEN_DIM], ACL_FORMAT_ND),          # 3: state_cache
        acl.create_tensor_desc(ACL_FLOAT,   [BATCH, HIDDEN_DIM], ACL_FORMAT_ND),          # 4: ape
        acl.create_tensor_desc(ACL_FLOAT16, [HIDDEN_DIM], ACL_FORMAT_ND),                 # 5: norm_weight
        acl.create_tensor_desc(ACL_FLOAT16, [SEQ_LEN, ROPE_DIM], ACL_FORMAT_ND),          # 6: rope_sin
        acl.create_tensor_desc(ACL_FLOAT16, [SEQ_LEN, ROPE_DIM], ACL_FORMAT_ND),          # 7: rope_cos
        # Optional inputs (indices 8-11): DT_UNDEFINED for gen_placeholder mode
        # When optional inputs are not provided, CANN expects DT_UNDEFINED descriptors
        acl.create_tensor_desc(ACL_DT_UNDEFINED, [], ACL_FORMAT_ND),                       # 8: state_block_table (optional)
        acl.create_tensor_desc(ACL_DT_UNDEFINED, [], ACL_FORMAT_ND),                       # 9: cu_seqlens (optional)
        acl.create_tensor_desc(ACL_DT_UNDEFINED, [], ACL_FORMAT_ND),                       # 10: seqused (optional)
        acl.create_tensor_desc(ACL_DT_UNDEFINED, [], ACL_FORMAT_ND),                       # 11: start_pos (optional)
        acl.create_tensor_desc(ACL_FLOAT16, [1, CMP_KV_DIM], ACL_FORMAT_ND),              # 12: swa_kv_cache
        acl.create_tensor_desc(ACL_INT32, [1], ACL_FORMAT_ND),                             # 13: scatter_indices
        acl.create_tensor_desc(ACL_FLOAT16, [1, CMP_KV_DIM], ACL_FORMAT_ND),              # 14: scatter_updates
    ]

    output_descs = [
        acl.create_tensor_desc(ACL_FLOAT16, [SEQ_LEN, CMP_KV_DIM], ACL_FORMAT_ND),        # 0: cmp_kv
        acl.create_tensor_desc(ACL_FLOAT,   [BATCH, HIDDEN_DIM], ACL_FORMAT_ND),          # 1: state_cache
        acl.create_tensor_desc(ACL_FLOAT16, [1, CMP_KV_DIM], ACL_FORMAT_ND),              # 2: swa_kv_cache
    ]

    num_inputs = len(input_descs)
    num_outputs = len(output_descs)
    print(f"  num_inputs={num_inputs}, num_outputs={num_outputs}")

    # ---- Allocate device memory and create data buffers ----
    print("\n[3] Allocating device memory...")

    input_arrays = [
        x_data, wkv_data, wgate_data, state_cache_data, ape_data,
        norm_weight_data, rope_sin_data, rope_cos_data,
        None, None, None, None,  # optional inputs (not provided)
        swa_kv_cache_data, scatter_indices_data, scatter_updates_data,
    ]

    input_dev_ptrs = []
    input_buffers = []
    for i, arr in enumerate(input_arrays):
        if arr is None:
            # Optional empty input: allocate minimal placeholder buffer
            # For gen_placeholder mode, we need a buffer with 0 or minimal data
            size = 32  # minimum aligned size for CANN
            dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, f"malloc input[{i}] (optional placeholder, {size}B)")
            buf = acl.create_data_buffer(dev_ptr, size)
            input_dev_ptrs.append(dev_ptr)
        else:
            size = max(arr.nbytes, 32)  # minimum 32 bytes for alignment
            dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret(ret, f"malloc input[{i}] ({arr.shape}, {size}B)")
            buf = acl.create_data_buffer(dev_ptr, size)
            input_dev_ptrs.append(dev_ptr)
        input_buffers.append(buf)
        if arr is None:
            print(f"  Input[{i}]: optional placeholder, {size}B allocated")
        else:
            print(f"  Input[{i}]: {arr.shape} {arr.dtype}, {size}B allocated")

    # Output sizes (aligned to 32 bytes minimum)
    output_sizes = [
        max(SEQ_LEN * CMP_KV_DIM * 2, 32),    # cmp_kv (float16)
        max(BATCH * HIDDEN_DIM * 4, 32),       # state_cache (float32)
        max(1 * CMP_KV_DIM * 2, 32),           # swa_kv_cache (float16)
    ]

    output_dev_ptrs = []
    output_buffers = []
    for i, size in enumerate(output_sizes):
        dev_ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
        check_ret(ret, f"malloc output[{i}] ({size}B)")
        buf = acl.create_data_buffer(dev_ptr, size)
        output_dev_ptrs.append(dev_ptr)
        output_buffers.append(buf)
        print(f"  Output[{i}]: {size}B allocated")

    # ---- Upload input data to device ----
    print("\n[4] Uploading input data to device...")
    for i, arr in enumerate(input_arrays):
        if arr is not None and arr.nbytes > 0:
            ret = upload_to_device(input_dev_ptrs[i], arr)
            if ret != ACL_SUCCESS:
                print(f"  WARN: upload input[{i}] returned {ret}")
            else:
                print(f"  Input[{i}]: uploaded {arr.nbytes}B")
        elif arr is None:
            print(f"  Input[{i}]: optional, not uploaded (placeholder)")

    # ---- Set operator attributes ----
    print("\n[5] Setting operator attributes...")
    attr = acl.op.create_attr()

    acl.op.set_attr_int(attr, "rope_head_dim", ROPE_DIM)
    acl.op.set_attr_int(attr, "cmp_ratio", CMP_RATIO)
    acl.op.set_attr_int(attr, "coff", 2)
    acl.op.set_attr_float(attr, "norm_eps", 1e-6)
    acl.op.set_attr_int(attr, "rotary_mode", 2)
    acl.op.set_attr_int(attr, "cache_mode", 1)
    acl.op.set_attr_int(attr, "state_cache_stride_dim0", HIDDEN_DIM)
    acl.op.set_attr_list_int(attr, "scatter_strides", [CMP_KV_DIM])
    try:
        acl.op.set_attr_bool(attr, "scatter_use_locking", False)
    except TypeError:
        acl.op.set_attr_bool(attr, "scatter_use_locking", 0)

    print(f"  Attributes: rope_head_dim={ROPE_DIM}, cmp_ratio={CMP_RATIO}, coff=2, "
          f"norm_eps=1e-6, rotary_mode=2, cache_mode=1, "
          f"state_cache_stride_dim0={HIDDEN_DIM}, scatter_strides=[{CMP_KV_DIM}]")

    # ---- Create stream ----
    print("\n[6] Creating stream...")
    stream, ret = acl.rt.create_stream()
    check_ret(ret, "acl.rt.create_stream")
    print(f"  stream ptr: {stream}")

    # ---- Execute the operator ----
    print("\n[7] Executing CompressorScatterUpdateV2 via acl.op.execute_v2...")
    op_type = "CompressorScatterUpdateV2"

    try:
        # Python ACL binding expects 7 args (no explicit num_inputs/num_outputs):
        # (op_type, input_desc_list, input_buf_list, output_desc_list, output_buf_list, attr, stream)
        ret = acl.op.execute_v2(
            op_type,
            input_descs,
            input_buffers,
            output_descs,
            output_buffers,
            attr,
            stream
        )
    except Exception as e:
        print(f"ERROR: acl.op.execute_v2 raised exception: {e}")
        print("\nPossible causes:")
        print("  - OPP not registered (ASCEND_CUSTOM_OPP_PATH not set)")
        print("  - Op type name mismatch")
        print("  - Tensor descriptor dtype/shape mismatch with compiled kernel variants")
        acl.rt.reset_device(0)
        acl.finalize()
        sys.exit(1)

    if ret != ACL_SUCCESS:
        print(f"ERROR: acl.op.execute_v2 returned {ret}")
        err_msg = acl.get_recent_err_msg()
        if err_msg:
            print(f"  Error: {err_msg}")
        print("\nThis likely means the operator was not found or tiling failed.")
        print("Possible causes:")
        print("  - ASCEND_CUSTOM_OPP_PATH not set correctly")
        print("  - Tiling library (libcust_opmaster_rt2.0.so) missing or incompatible")
        print("  - Input shapes/dtypes not matching any compiled kernel variant")
        acl.rt.reset_device(0)
        acl.finalize()
        sys.exit(1)
    else:
        print(f"OK: Operator dispatched successfully (ret={ret})")

    # ---- Synchronize stream ----
    print("\n[8] Synchronizing stream...")
    ret = acl.rt.synchronize_stream(stream)
    check_ret(ret, "acl.rt.synchronize_stream")

    # ---- Download output data from device ----
    print("\n[9] Downloading output data from device...")
    output_dtypes = [np.float16, np.float32, np.float16]
    output_shapes = [(SEQ_LEN, CMP_KV_DIM), (BATCH, HIDDEN_DIM), (1, CMP_KV_DIM)]
    output_names = ["cmp_kv", "state_cache", "swa_kv_cache"]
    output_results = []

    for i in range(num_outputs):
        arr = download_from_device(output_dev_ptrs[i], output_sizes[i],
                                   output_dtypes[i], output_shapes[i])
        output_results.append(arr)
        if arr is not None:
            print(f"  Output[{i}] ({output_names[i]}): shape={arr.shape}, "
                  f"mean={arr.mean():.6f}, std={arr.std():.6f}, "
                  f"min={arr.min():.6f}, max={arr.max():.6f}")
        else:
            print(f"  Output[{i}] ({output_names[i]}): download FAILED")

    # ---- Basic sanity checks ----
    print("\n[10] Running sanity checks...")
    checks = []
    for name, arr in zip(output_names, output_results):
        if arr is None:
            print(f"  {name}: data unavailable")
            checks.append(False)
        elif np.all(arr == 0):
            print(f"  {name}: all zeros — kernel may not have computed correctly")
            checks.append(False)
        else:
            print(f"  {name}: has non-zero values ✓")
            checks.append(True)

    # ---- Cleanup ----
    print("\n[11] Cleanup...")
    for ptr in input_dev_ptrs:
        acl.rt.free(ptr)
    for ptr in output_dev_ptrs:
        acl.rt.free(ptr)
    for buf in input_buffers:
        acl.destroy_data_buffer(buf)
    for buf in output_buffers:
        acl.destroy_data_buffer(buf)
    for desc in input_descs:
        acl.destroy_tensor_desc(desc)
    for desc in output_descs:
        acl.destroy_tensor_desc(desc)
    acl.rt.destroy_stream(stream)
    acl.rt.reset_device(0)
    acl.finalize()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if any(checks):
        print("\n✓ Operator executed and produced non-zero output — SMOKE TEST PASSED")
    else:
        print("\n✗ All outputs are zero or unavailable — SMOKE TEST FAILED")
    print()
    print("NOTE: This is a smoke test only. Numerical accuracy requires proper")
    print("      model weights and reference data.")

    return 0 if any(checks) else 1


if __name__ == "__main__":
    sys.exit(main())