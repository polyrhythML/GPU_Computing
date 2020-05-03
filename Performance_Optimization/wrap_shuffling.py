"""
* A feature in CUDA that allows threads that exist within the same CUDA Warp
concurrently to communicate by directly reading and writing to each other's
registers (that is, their local stack-space variables), without the use of
shared ​ variables or global device memory. Warp shuffling is actually much
faster and easier to use than the other two options. This almost sounds too
good to be true, so there must be a ​ catch— ​ indeed, the ​ catch ​ is that this only
works between threads that exist on the same CUDA Warp, which limits
shuffling operations to groups of threads of size 32 or less.

* Another catch is that we can only use datatypes that are 32 bits or less. This means that we
can't shuffle 64-bit ​ long long ​ integers or ​ double ​ floating point values across
a Warp.

* Just as a Grid consists of blocks, blocks similarly consist of one or more Warps, depending on the
number of threads the Block uses – if a Block consists of 32 threads, then it
will use one Warp, and if it uses 96 threads, it will consist of three Warps.

* Even if a Warp is of a size less than 32, it is also considered a full Warp: this
means that a Block with only one single thread will use 32 cores. This also
implies that a block of 33 threads will consist of two Warps and 31 cores.

* A Warp has what is known as the Lockstep Property. This means that every thread in a warp will iterate through every instruction,
perfectly in parallel with every other thread in the Warp.

* Every thread in a single Warp will step through the same exact instructions
simultaneously, ​ ignoring ​ any instructions that are not applicable to a
particular thread – this is why any divergence among threads within a
single Warp is to be avoided as much as possible. NVIDIA calls this
execution model ​ Single Instruction Multiple Thread, or SIMT​.

* That is why we try to use Blocks of 32 threads consistently.

* A lane​ in a Warp is a unique identifier for a particular thread within the warp,
which will be between 0 and 31. Sometimes, this is also called the Lane ID.
"""

# Write a program to shuffle even and odd lane ID variables across threads in a warp.

from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

#__shufl_xor
"""
It performs an XOR​ operation on the binary Lane ID of the current thread with 1 ​ ​ , which will be either its left neighbor (if the
least significant digit of this thread's Lane is "1" in binary), or its right
neighbor (if the least significant digit is "0" in binary). It then sends the
current thread's temp value to its neighbor, while retrieving the neighbor's
temp value, which is __shfl_xor.This will be returned as output right back
into temp.
"""

ShflCode='''
__global__ void shfl_xor_ker(int *input, int * output) {
int temp = input[threadIdx.x];
temp = __shfl_xor(temp, 1, blockDim.x);
output[threadIdx.x] = temp;
}'''

shfl_mod = SourceModule(ShflCode)
shfl_ker = shfl_mod.get_function('shfl_xor_ker')
dinput = gpuarray.to_gpu(np.int32(range(32)))
doutout = gpuarray.empty_like(dinput)
shfl_ker(dinput, doutout, grid=(1,1,1), block=(32,1,1))
print('input array: %s' % dinput.get())
print('array after __shfl_xor: %s' % doutout.get())

