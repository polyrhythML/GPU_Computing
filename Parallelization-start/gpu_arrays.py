# Very much analogous to the Numpy array, There  are gpu arrays as a data structure which we will be using to perform
# operations on the gpu tensors.

"""

GPU Data transfer

* GPU's own memory is called it's device memory.

* This memory is different from the cache memory, shared memory, and register memory and is sometimes called global
device memory.

* For the most part, we treat (global) device memory on the GPU as we do dynamically allocated heap memory in C
(with the malloc and free functions) or C++ (as with the new and delete operators); in CUDA C, this is complicated
further with the additional task of transferring data back and forth between the CPU to the GPU (with commands such as
cudaMemcpyHostToDevice and cudaMemcpyDeviceToHost),
all while keeping track of multiple pointers in both the CPU and GPU space and performing proper memory allocations
(cudaMalloc) and deallocations (cudaFree).

* PyCUDA covers all of the overhead of memory allocation, deallocation, and data transfers with the gpuarray class.
As stated, this class acts similarly to NumPy arrays, using vector/ matrix/tensor shape structure information for the
data. gpuarray objects even perform automatic cleanup based on the lifetime, so we do not have to worry about freeing 
any GPU memory stored in a gpuarray object when we are done with it. 

"""

import numpy as np
import sys
import pycuda.autoinit
from pycuda import gpuarray

# Create a numpy array
cpu_data = np.array([2, 3, 5, 6, 7], dtype=np.float32)

# transfer the numpy array to the GPU
gpu_data = gpuarray.to_gpu(cpu_data)

# Perform an operation on the gpu array
gpu_data_1 = gpu_data**2

# Fetch the computed gpu array
print(gpu_data_1.get())

"""
Note:Generally speaking, it's a good idea to specifically set data types with NumPy when we are sending data to the GPU. 
The reason for this is twofold: first, since we are using a GPU for increasing the performance of our application, 
we don't want any unnecessary overhead of using an unnecessary type that will possibly take up more computational 
time or memory, 
Secondly, Since we will be writing some code in C and since C is a statically type language, we will have to specify 
with types or our code won't work correctly.
"""

# Basic point wise arithmetic operations

a_gpu = gpuarray.to_gpu(np.random.random((1,10)))
b_gpu = gpuarray.to_gpu(np.random.random((1,10)))
c_gpu = gpuarray.to_gpu(np.random.random((1,10)))

print((a_gpu + b_gpu).get())
print((a_gpu + c_gpu).get())
print((b_gpu + c_gpu).get())
print((a_gpu/2).get())
print((b_gpu-1).get())
print((c_gpu+2).get())

