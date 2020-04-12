# CONCEPTUAL REVISINO OF HOST SIDE MULTITHREADING AND MULTIPROCESSING

"""
*  A process has its own address space, as it runs concurrently with, and independently of, all other processes.
A process is, generally speaking, blind to the actions of other processes, although multiple processes can communicate
through sockets or pipes. In Linux and Unix, new processes are spawned with the fork system call.

* Host-side thread exists within a single process, and multiple threads can also exist within a single process.
Multiple threads in a single process run concurrently. All threads in the same process share the same address
space within the process and have access to the same shared variables and data.
Generally, resource locks are used for accessing data among multiple threads, so as to avoid race conditions.

* Threads are much more lightweight than processes, and it is far faster for an operating system kernel to switch
tasks between multiple threads in a single process, than to switch tasks between multiple processes.
Normally, an operating system kernel will automatically execute different threads and processes on different
CPU cores to establish true concurrency.

* A peculiarity of Python is that while it supports multi-threading through the threading module, all threads will
execute on the same CPU core. This is due to technicalities of Python being an interpreted scripting language,
and is related to Python's Global Identifier Lock (GIL). To achieve true multi-core concurrency on the host
through Python, we, unfortunately, must spawn multiple processes with the multiprocessing module.

* By the nature of PyCUDA, however, we will have to create a new CUDA context for every thread or process that
we will use that will make use of the GPU.

"""
import pycuda
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import threading

num_arrays = 10
array_len = 1024 ** 2

kernel_code = """       
__global__ void mult_ker(float * array, int array_len)
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

}
"""


class KernelLauncherThread(threading.Thread):
    def __init__(self, input_array):
        threading.Thread.__init__(self)
        self.input_array = input_array
        self.output_array = None

    def run(self):
        self.dev = drv.Device(0)
        self.context = self.dev.make_context()

        self.ker = SourceModule(kernel_code)

        self.mult_ker = self.ker.get_function('mult_ker')

        self.array_gpu = gpuarray.to_gpu(self.input_array)

        self.mult_ker(self.array_gpu, np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))

        self.output_array = self.array_gpu.get()

        self.context.pop()

    def join(self):
        threading.Thread.join(self)
        return self.output_array


drv.init()

data = []
gpu_out = []
threads = []

# generate random arrays and thread objects.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

for k in range(num_arrays):
    # create a thread that uses data we just generated
    threads.append(KernelLauncherThread(data[k]))

# launch threads to process arrays.
for k in range(num_arrays):
    threads[k].start()

# get data from launched threads.
for k in range(num_arrays):
    gpu_out.append(threads[k].join())

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))