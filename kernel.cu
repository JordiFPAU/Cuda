#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
__global__ void mat_vec_kernel(const float* A, const float* B,  float* c, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * B[col];
        }
        c[row] = sum;
    }
}

extern "C" void mat_vec_gpu(const float* d_A, const float* d_B,  float* d_c, int N){
    int thread_per_block = 256; 
    int blocks_per_grids = std::ceil(N* 1.0 / thread_per_block);
    mat_vec_kernel<<<blocks_per_grids, thread_per_block>>>(d_A, d_B, d_c, N);
    cudaDeviceSynchronize();
}