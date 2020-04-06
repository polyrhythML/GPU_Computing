import numpy as np
import sys
sys.path.insert(0, "home/amit/anaconda3/envs/pycuda/lib/python3.7/site-packages/pycuda")
print(sys.path)
import pycuda.autoinit
from pycuda.elementwise import Elementwisekernel
from pycuda import gpuarray
from time import time


# Time multiplication on CPU
cpu_data = np.float32(np.random.random(50000000))


def CPU_multiplication():

    t1 = time()
    cpu_data_2 = cpu_data * np.float32(2)
    t2 = time()
    return t2-t1


def GPU_multiplication():

    gpu_data = gpuarray.to_gpu(np.random.random(50000000))
    t1 = time()
    gpu_data_2 = gpu_data * np.float32(2)
    t2 = time()
    return t2-t1


# define a elementwise kernel

gpu_2div_ker = Elementwisekernel("float *in, float *out", "out[i] = 2/in[i];", "gpu_2div_ker")


def single_operation_speed_test():

    cpu_time_array = []
    gpu_time_array = []
    for i in range(10):
        print("\nIteration : {}".format(i))
        cpu_time = CPU_multiplication()
        print("Total time to compute on CPU : {}".format(cpu_time))
        gpu_time = GPU_multiplication()
        print("Total time to compute on GPU : {}".format(gpu_time))
        cpu_time_array.append(cpu_time)
        gpu_time_array.append(gpu_time)

    print("Average CPU time : {}".format(np.mean(cpu_time_array[1:])))
    print("Average GPU time : {}".format(np.mean(gpu_time_array[1:])))


def kernel_speed_test():

    t1 = time()
    host_data_2x =  cpu_data * np.float32(2)
    t2 = time()
    print('total time to compute on CPU: %f' % (t2 - t1))
    device_data = gpuarray.to_gpu(cpu_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)
    t1 = time()
    gpu_2div_ker(device_data, device_data_2x)
    t2 = time()
    from_device = device_data_2x.get()
    print('total time to compute on GPU: %f' % (t2 - t1))
    print('Is the host computation the same as the GPU computation? : {}'.
          format(np.allclose(from_device, host_data_2x)))




if __name__ == "__main__":

    cpu_time_array = []
    gpu_time_array = []
    for i in range(10):
        print("\nIteration : {}".format(i))
        cpu_time = CPU_multiplication()
        print("Total time to compute on CPU : {}".format(cpu_time))
        gpu_time = GPU_multiplication()
        print("Total time to compute on GPU : {}".format(gpu_time))
        cpu_time_array.append(cpu_time)
        gpu_time_array.append(gpu_time)

    print("Average CPU time : {}".format(np.mean(cpu_time_array[1:])))
    print("Average GPU time : {}".format(np.mean(gpu_time_array[1:])))

"""
In PyCUDA, GPU code is often compiled at runtime with the NVIDIA nvcc compiler and then subsequently called from PyCUDA.
This can lead to an unexpected slowdown, usually the first time a program or GPU operation is run in a given 
Python session.
"""