# INTRODUCTION

"""
* We saw one level of concurrency i.e. synchronization of threads.
* There is another level of concurrency available multiple kernels and GPU memory operations. We can launch multiple
memory operations and kernel operation at once, without waiting for each operation to finish.
* We should not launch a kernel until all it's input is copied to the device memory or shouldn't copy the output of a
launched kernel to the host until the kernel has finished execution.

* CUDA STREAM : a stream is a sequence of operations that are run in order on the GPU. By itself, a single stream isn't
of any use—the point is to gain concurrency over GPU operations issued by the host by using multiple streams.
This means that we should interleave launches of GPU operations that correspond to different streams, in order to
exploit this notion.

* EVENTS : feature of streams that are used to precisely time kernels and indicate to the host as to what operations
have been completed within a given stream.

* CONTEXT : A context can be thought of as analogous to a process in your operating system, in that the GPU
keeps each context's data and kernel code walled off and encapsulated away from the other contexts currently
existing on the GPU.

"""

"""
CUDA DEVICE SYNCHRONIZATION

* This is an operation where the host blocks any further execution until all operations issued to the GPU 
(memory transfers and kernel executions) have completed.This is required to ensure that operations dependent on prior 
operations are not executed out-of-order—for example, to ensure that a CUDA kernel launch is completed before the 
host tries to read its output.This function effectively blocks further execution on the host until all GPU operations 
have completed.

* If we are synchronizing a single device operation in a run then calling cudaDeviceSynchronize() sequentially makes
sense. But if we want to run multiple kernels and want to have multiple to and fro host to device memory transfers
concurrently, we synchronize such operation across different streams.
 
"""

# Example of using streams
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

num_arrays = 200
array_len = 1024**2

ker = SourceModule(
    """ __global__ void mult_ker(float * array, int array_len)
    {     
        int thd = blockIdx.x*blockDim.x + threadIdx.x;     
        int num_iters = array_len / blockDim.x;     
        for(int j=0; j < num_iters; j++)     
            {   
                int i = j * blockDim.x + thd;         
                for(int k = 0; k < 50; k++)         
                {   
                    array[i] *= 2.0;             
                    array[i] /= 2.0;    
                }     
               
            }
        }""")


# Instantiate the Kernel
mult_ker = ker.get_function('mult_ker')

data = []
data_gpu = []
gpu_out = []

# generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))


def without_stream():

    t_start = time()

    # copy arrays to GPU
    for k in range(num_arrays):
        data_gpu.append(gpuarray.to_gpu(data[k]))

    # process arrays.
    for k in range(num_arrays):
        mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))

    # copy arrays from GPU
    for k in range(num_arrays):
        gpu_out.append(data_gpu[k].get())

    t_end = time()

    for k in range(num_arrays):
        assert (np.allclose(gpu_out[k], data[k]))

    print('Total time: %f' % (t_end - t_start))


def with_stream():
    streams = []
    t_start = time()

    # Generate series of streams that we will use to organize the kernel launches. We can get a stream object from the
    # pycuda.driver submodule with the Stream class.
    for _ in range(num_arrays):
        streams.append(drv.Stream())

    # First we want to synchronously parallelize host to device transfer of the data array.
    for k in range(num_arrays):
        data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

    # Now launch kernel for each of the stream.
    for k in range(num_arrays):
        mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1), stream=streams[k])

    # Finally pull out the processed data array back to the host
    for k in range(num_arrays):
        gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

    t_end = time()

    print('Total time: %f' % (t_end - t_start))


if __name__ == "__main__":

    without_stream()
    with_stream()
    """
    Output :
    Total time: 3.137281
    Total time: 0.410192
    7.8x decrease in the runtime using streams
    Since operations in a single stream are blocked until only all necessary prior operations are competed, 
    We will gain concurrency among distinct GPU operations and make full use of our device.
    
    """






