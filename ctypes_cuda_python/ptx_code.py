"""
* Previous example, we had host and device code compiled into a single file.

* We can use pure gpu compiled code and then launch it using writing a C wrapper
each and everytime.

* The NVCC compiler compiles CUDA-C into ​ PTX​ (​ Parallel Thread
Execution​ ), which is an interpreted pseudo-assembly language that is
compatible across NVIDIA 's various GPU architectures.

* Whenever you compile a program that uses a CUDA kernel with NVCC into an executable
EXE, DLL, .so​, or ELF file, there will be PTX code for that kernel contained
within the file. We can also directly compile a file with the extension PTX,
which will contain only the compiled GPU kernels from a compiled CUDA
.cu file.

* PyCUDA includes an interface to load a CUDA kernel directly from a PTX, freeing us from the shackles of just-in-time
compilation while still allowing us to use all of the other nice features from
PyCUDA.

"""
from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pycuda
from pycuda import gpuarray
import pycuda.autoinit

# load the ptx file
mandel_mod = pycuda.driver.module_from_file('./kernel_cuda.ptx')
# Refer to the kernel function
mandel_ker = mandel_mod.get_function('mandelbrot_ker')


def mandelbrot(breadth, low, high, max_iters, upper_bound):
    lattice = gpuarray.to_gpu(np.linspace(low, high, breadth, dtype=np.float32))
    out_gpu = gpuarray.empty(shape=(lattice.size, lattice.size), dtype=np.float32)

    gridsize = int(np.ceil(lattice.size ** 2 / 32))

    mandel_ker(lattice, out_gpu, np.int32(256), np.float32(upper_bound ** 2), np.int32(lattice.size),
               grid=(gridsize, 1, 1), block=(32, 1, 1))

    out = out_gpu.get()

    return out


if __name__ == '__main__':
    t1 = time()
    mandel = mandelbrot(512, -2, 2, 256, 2)
    t2 = time()

    mandel_time = t2 - t1

    print('It took %s seconds to calculate the Mandelbrot graph.' % mandel_time)

    plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.show()
