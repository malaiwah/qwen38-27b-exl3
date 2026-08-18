// SPDX-License-Identifier: Apache-2.0
// exl3_gemm_v8.cu — Use select_exl3_gemm_kernel exactly like existing code

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"

void exl3_gemm_v8_torch
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    at::Tensor& A_had,
    const c10::optional<at::Tensor>& svh,
    int bits,
    int cb,
    at::Tensor& locks,
    int num_sms
)
{
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;  // B shape is (K//16, N//16, 16*bits), so N = B.size(1)*16
    bool c_fp32 = (C.scalar_type() == at::ScalarType::Float);

    const half* A_ptr = (const half*)A.data_ptr<at::Half>();
    const uint16_t* B_ptr = (const uint16_t*)B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = suh ? (const half*)suh->data_ptr<at::Half>() : nullptr;
    half* A_had_ptr = (half*)A_had.data_ptr<at::Half>();
    const half* svh_ptr = svh ? (const half*)svh->data_ptr<at::Half>() : nullptr;
    int* locks_ptr = (int*)locks.data_ptr();

    // Use DevCtx for locks and num_sms (exactly like existing code)
    int device;
    cudaGetDevice(&device);
    int* dev_locks = DevCtx::instance().get_locks(device);
    int dev_num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);

    // Select kernel (exactly like existing code)
    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel(
        cc, size_m, size_k, size_n, bits, c_fp32,
        2,  // force_shape_idx = shape 2
        &block_dim, &shape_idx, &dev_num_sms, cb
    );

    if (!kernel) {
        TORCH_CHECK(false, "v8: select_exl3_gemm_kernel returned null");
    }

    // Set attribute (exactly like existing code)
    constexpr int SMEM_MAX = 90 * 1024;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);

    // Launch (exactly like existing code)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    void* kernelArgs[] = {
        (void*)&A_ptr, (void*)&B_ptr, (void*)&C_ptr, (void*)&size_m,
        (void*)&size_k, (void*)&size_n, (void*)&dev_locks,
        (void*)&suh_ptr, (void*)&A_had_ptr, (void*)&svh_ptr
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)kernel,
        dim3(dev_num_sms),
        dim3(block_dim),
        kernelArgs,
        SMEM_MAX,
        stream
    );
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "v8: cudaLaunchCooperativeKernel failed: ", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_gemm_prefill", &exl3_gemm_v8_torch, "EXL3 GEMM v8 (uses select_exl3_gemm_kernel)");
}
