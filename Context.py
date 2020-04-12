# CONTEXTS
"""
* A CUDA context is usually described as being analogous to a process in an operating system.a process is an instance
of a single program running on a computer; all programs outside of the operating system kernel run in a process.
Each process has its own set of instructions, variables, and allocated memory, and is, generally speaking,
blind to the actions and memory of other processes. When a process ends, the operating system kernel performs a
cleanup, ensuring that all memory that the process allocated has been de-allocated, and closing any files,
network connections, or other resources the process has made use of.

* A context is associated with a single host program that is using the GPU. A context holds in memory all CUDA kernels
and allocated memory that is making use of and is blind to the kernels and memory of other currently existing contexts.
When a context is destroyed (at the end of a GPU based program, for example), the GPU performs a cleanup of all code
and allocated memory within the context, freeing resources up for other current and future contexts.

* We can access the current context object with pycuda.autoinit.context, and we can synchronize in our current
context by calling the pycuda.autoinit.context.synchronize() function.

* Memory allocation in CUDA is always synchronized!


"""
from time import time
import matplotlib
#this will prevent the figure from popping up
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel


mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
    """
    mandelbrot_graph[i] = 1;
    pycuda::complex<float> c = lattice[i]; 
    pycuda::complex<float> z(0,0);
    for (int j = 0; j < max_iters; j++)
        {
    
         z = z*z + c;
    
         if(abs(z) > upper_bound)
             {
              mandelbrot_graph[i] = 0;
              break;
             }
        }
    
    """,
    "mandel_ker")


def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    # we set up our complex lattice as such
    real_vals = np.matrix(np.linspace(real_low, real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace(imag_high, imag_low, height), dtype=np.complex64) * 1j
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)

    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu_async(mandelbrot_lattice)

    # synchronize in current context
    pycuda.autoinit.context.synchronize()

    # allocate an empty array on the GPU
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper_bound))

    pycuda.autoinit.context.synchronize()

    mandelbrot_graph = mandelbrot_graph_gpu.get_async()

    pycuda.autoinit.context.synchronize()

    return mandelbrot_graph


def manual_context():

    # Initialize CUDA with pycuda.driver.init
    pycuda.driver.init()

    # Pick a device to instantiate a context, in-case there are multiple devices
    dev = pycuda.driver.Device(0)

    # Create a context
    ctx = dev.make_context()

    # Within that context add an array to the GPU memory
    x = gpuarray.to_gpu(np.float32([1, 2, 3]))

    # fetch the value from the GPU memory
    print(x.get())

    # Destroy the Context
    ctx.pop()


if __name__ == '__main__':
    t1 = time()
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2)
    t2 = time()

    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))

    manual_context()