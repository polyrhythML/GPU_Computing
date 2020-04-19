# THREAD SYNCHRONIZATION AND SHARED MEMORY

"""
* We need to ensure that every single thread has reached the same exact line in the code before we continue with any
further computation; we call this thread synchronization. Synchronization works hand-in-hand with thread
intercommunication, that is, different threads passing and reading input from each other; in this case, we'll usually
want to make sure that all of the threads are aligned at the same step in computation before any data is passed around.

* RACE CONDITION :
The issue of multiple threads reading and writing to the same memory address and the problems that may arise from that.
block level synchronization barrier—this means that every thread that is executing within a block will stop when it
reaches a __syncthreads() instance and wait until each and every other thread within the same block reaches that same
invocation of __syncthreads() before the the threads continue to execute the subsequent lines of code.

__syncthreads() can only synchronize threads within a single CUDA block, not all threads within a CUDA grid!

* SHARED MEMORY is a type of memory meant specifically for intercommunication of threads within a single CUDA block;
the advantage of using this over global memory is that it is much faster for pure inter-thread communication.
In contrast to global memory, though, memory stored in shared memory cannot directly be accessed by the
host—shared memory must be copied back into global memory by the kernel itself first.

* Local thread arrays (for example, a declaration of int a[10]; within the kernel) and pointers to global GPU memory
(for example, a value passed as a kernel parameter of the form int * b) may look and act similarly,
but are very different. For every thread in the kernel, there will be a separate a array that the other threads
cannot read, yet there is a single b that will hold the same values and be equally accessible for all of the threads.

"""


import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt


ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )
#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )
#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )
#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )
// return the number of living neighbors for a given cell                
__device__ int nbrs(int x, int y, int * in)
{
     return ( in[ _INDEX(x -1, y+1) ] + in[ _INDEX(x-1, y) ] + in[ _INDEX(x-1, y-1) ] \
                   + in[ _INDEX(x, y+1)] + in[_INDEX(x, y - 1)] \
                   + in[ _INDEX(x+1, y+1) ] + in[ _INDEX(x+1, y) ] + in[ _INDEX(x+1, y-1) ] );
}
__global__ void conway_ker(int * lattice, int iters)
{
   // x, y are the appropriate values for the cell covered by this thread
   int x = _X, y = _Y;

   for (int i = 0; i < iters; i++)
   {

       // count the number of neighbors around the current cell
       int n = nbrs(x, y, lattice);

       int cell_value;


        // if the current cell is alive, then determine if it lives or dies for the next generation.
        if ( lattice[_INDEX(x,y)] == 1)
           switch(n)
           {
              // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
              case 2:
              case 3: cell_value = 1;
                      break;
              default: cell_value = 0;                   
           }
        else if( lattice[_INDEX(x,y)] == 0 )
             switch(n)
             {
                // a dead cell comes to life only if it has 3 neighbors that are alive.
                case 3: cell_value = 1;
                        break;
                default: cell_value = 0;         
             }

        __syncthreads();
        lattice[_INDEX(x,y)] = cell_value;
        __syncthreads(); 
    }

}
""")

conway_ker = ker.get_function("conway_ker")

if __name__ == '__main__':
    # set lattice size
    N = 32

    lattice = np.int32(np.random.choice([1, 0], N * N, p=[0.25, 0.75]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)
    conway_ker(lattice_gpu, np.int32(100000), grid=(1, 1, 1), block=(32, 32, 1))
    fig = plt.figure(1)
    plt.imshow(lattice_gpu.get())
    plt.show()


