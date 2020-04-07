# KERNEL
"""
A parallel funciton that can be launched directly from the CPU onto the device(the GPU). It is interchangeably used with
terms such as CUDA kernel or Kernel function.
"""

# DEVICE FUNCTION
"""
A device function is a function that can only be called from a kernel function or another device function. Device 
functions look and act like normal series C/C++ functions, only they are running on the GPU and are called in parallel 
from the kernels.
"""

# SourceModule
"""
* SourceModule complies code into a CUDA module, this is like a Python module, contains complied CUDA code.
"""
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule


#################   Writing a Inhousekernel   ###############

"""
* Below code writes a C function where we use __global__ keyword in the declaration. This distinguishes the function as 
a kernel to the compiler.

* Void function to be used as return type since we will always get our output values by passing a pointer to some empty
chunk of the memory that we pass in as a parameter.

* outvec - output array pointer

* vec - input array pointer

* scalar - float value, not a pointer. We can pass a singleton value to our kernel. We can do so without using pointers.

* threadIX - index of a thread, used to denote the identity to each of the thread. 
             What index to be processed on the input and output arrays.
             This can also be used to assign particular threads some task other than with standard C control flow.

"""

ker = SourceModule("""
__global__ void scalar_multiply_kernel(float *outvec, float scalar, float *vec)
{
int i = threadIdx.x;
outvec[i] = scalar*vec[i];
}
""")

##################### Understanding Thread Blocks and Grids ####################################

"""
* A thread is a sequence of instructions that are exectued on a single core of the GPU cores and threads should not be
thought of as synonymous.

* It is possible to launch kernels that use many more threads than there are cores on the GPU. This is because not all
not all the threads are run simulanteously. Rather there is system scheduling of the threads. Thus much like CPUs, where
a few cores can run hundreds of processes and each of these processes can run thousands of threads.

* GPU handles threads in a similar way, allowing for seamless computation over tens of thousands of cores.

* BLOCKS - multiple thread executed on the GPU in abstract unit called blocks.

threadIDX.x - 1st block dimension
threadIDX.y - 2nd block dimension
threadIDX.z - 3rd block dimension

We can index block over 3 dimensions, suppose we are doing 3D point arithmetic, we can distribute each computation over 
a 3D block of GPU threads.

* GRIDS - Blocks are further executed in abstract batches known as grids. Its kind of a group of blocks. Similarly like 
block ,we can index grid in 3 dimensions.

blockIDX.x - 1st Grid dimension
blockIDX.y - 2nd Grid dimension
blockIDX.z - 3rd Grid dimension 



"""



def test_kernel():

    scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")

    # Create some arrays for computation and move them to the GPU
    test_vec = np.random.randn(512).astype(np.float32)
    test_vec_gpu = gpuarray.to_gpu(test_vec)
    # Allocate some gpu memory to the output vector
    out_vec_gpu = gpuarray.empty_like(test_vec_gpu)

    # Let's feed the arrays to our inhouse-kernel
    scalar_multiply_gpu(out_vec_gpu, np.float32(2), test_vec_gpu, block=(512,1,1), grid=(1,1,1))
    print("Does our kernel work correctly? : {}".format(np.allclose(out_vec_gpu.get(), 2*test_vec)))


if __name__ == "__main__":

    test_kernel()

