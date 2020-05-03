"""
* Atomic operations are very simple, thread-safe operations that output to a
single global array element or shared memory variable, which would normally
lead to race conditions otherwise.

* It should be noted that while Atomics are indeed thread-safe, they by no
means guarantee that all threads will access them at the same time, and
they may be executed at different times by different threads.
We need to do a syncthread to avoid this.

"""

from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

AtomicCode="""
__global__ void atomic_ker(int *add_out, int *max_out)
{
int tid = blockIdx.x*blockDim.x + threadIdx.x;
atomicExch(add_out, 0);
__syncthreads();
atomicAdd(add_out, 1);
atomicMax(max_out, tid);
}
"""

atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function('atomic_ker')
add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)
atomic_ker(add_out, max_out, grid=(1,1,1), block=(100,1,1))
print('Atomic operations test:')
print('add_out: %s' % add_out.get()[0])
print('max_out: %s' % max_out.get()[0])