# Events

"""
* They are a progress marker for a stream of operations.Events are generally used to provide measure time duration on
the device side to precisely time operations; the measurements we have been doing so far have been with host-based
Python profilers and standard Python library functions such as time.

* Additionally, events they can also be used to provide a status update for the host as to the state of a stream and
what operations it has already completed, as well as for explicit stream-based synchronization.

* If there were a high degree of deviation, then that would mean that we were making highly uneven usage of the GPU in
our kernel executions, and we would have to re-tune parameters to gain a greater level of concurrency.
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

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


def no_stream():

    array_len = 100 * 1024 ** 2
    data = np.random.randn(array_len).astype('float32')
    data_gpu = gpuarray.to_gpu(data)

    # Create start and end event objects
    start_event = drv.Event()
    end_event = drv.Event()

    # Record the kernel timing
    start_event.record()
    mult_ker(data_gpu, np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))
    end_event.record()
    # Kernels in PyCUDA have launched asynchronously, we have to have to ensure that our host code is properly
    # synchronized with the GPU.
    end_event.synchronize()

    # Events have a binary value that indicates whether they were reached or not yet,
    # which is given by the function query.
    # We can block further host code execution until the kernel completes by this event object's synchronize function
    # this will ensure that the kernel has completed before any further lines of host code are executed.
    print('Has the kernel started yet? {}'.format(start_event.query()))
    print('Has the kernel ended yet? {}'.format(end_event.query()))
    print('Kernel execution time in milliseconds: %f ' % start_event.time_till(end_event))


def with_stream():
    # We will time progress of each of the streams using the events
    data = []
    data_gpu = []
    gpu_out = []
    streams = []
    start_events = []
    end_events = []
    num_arrays = 200
    array_len = 1024 ** 2

    for _ in range(num_arrays):
        streams.append(drv.Stream())
        start_events.append(drv.Event())
        end_events.append(drv.Event())

    # generate random arrays.
    for _ in range(num_arrays):
        data.append(np.random.randn(array_len).astype('float32'))

    t_start = time()

    # copy arrays to GPU.
    for k in range(num_arrays):
        data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

    # process arrays.
    for k in range(num_arrays):
        start_events[k].record(streams[k])
        mult_ker(data_gpu[k], np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1), stream=streams[k])
    for k in range(num_arrays):
        end_events[k].record(streams[k])

    # copy arrays from GPU.
    for k in range(num_arrays):
        gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

    t_end = time()

    for k in range(num_arrays):
        assert (np.allclose(gpu_out[k], data[k]))

    kernel_times = []

    for k in range(num_arrays):
        kernel_times.append(start_events[k].time_till(end_events[k]))

    print('Total time: %f' % (t_end - t_start))
    print('Mean kernel duration (milliseconds): %f' % np.mean(kernel_times))
    print('Mean kernel standard deviation (milliseconds): %f' % np.std(kernel_times))


if __name__ == "__main__":

    #no_stream()

    for i in range(20):
        with_stream()






