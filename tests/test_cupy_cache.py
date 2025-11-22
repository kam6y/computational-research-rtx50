import cupy as cp
import time
import os

print(f"CUDA_CACHE_PATH: {os.environ.get('CUDA_CACHE_PATH')}")
print(f"CUDA_CACHE_DISABLE: {os.environ.get('CUDA_CACHE_DISABLE')}")

code = r'''
extern "C" __global__
void my_kernel(float* x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = x[i] * 2.0;
    // Add some unique code to ensure it's a new kernel if needed
    // VERSION 2
}
'''

print("Compiling/Loading kernel...")
start = time.time()
kernel = cp.RawKernel(code, 'my_kernel')
end = time.time()
print(f"Compilation/Loading took: {end - start:.4f}s")

x = cp.ones((1024,), dtype=cp.float32)
kernel((1,), (1024,), (x,))
print(f"Result: {x[0]}")
