"""
Kernels launching and managing other kernels without any interaction or input on
behalf of the host.

Launch N threads print a msg and for each thread recursively call the N-1 threads
till N = 1 .
"""
from __future__ import division
import numpy as np
# We import the dynamicsourceModule than our usual SourceModule
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit



DynamicParallelismCode="""
__global__ void dynamic_hello_ker(int depth)
{
    printf("Hello from thread %d, recursion depth %d!\\n", threadIdx.x, depth);
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockDim.x > 1)
        {
            printf("Launching a new kernel from depth %d .\\n", depth);
            printf("-----------------------------------------\\n");
            dynamic_hello_ker<<< 1, blockDim.x - 1 >>>(depth + 1);
        }
}"""

dp_mod = DynamicSourceModule(DynamicParallelismCode)
our_kernel = dp_mod.get_function("dynamic_hello_ker")
our_kernel(np.int32(0), grid=(1, 1, 1), block=(4, 1, 1))