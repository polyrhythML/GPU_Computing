# Debugging CUDA Kernels

"""
* We can use printf statement inside the C kernel code to get the output
across every single thread to check where exactly is the problem with the
kernel code.

* We will take up the example of the dot product in a matrix to illustrate
debugging.

"""
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# Let's write a kernel to print the thread and the block dimensions.


def hello_world_example():
    ker = SourceModule("""
    
    __global__ void dimensions()
    {
        printf("Hello world from thread %d in block %d! \\n", threadIdx.x, blockIdx.x);
        // synchronize threads after printf function execution
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x ==0)
        {
            printf("----------------------------------\\n");
            printf("This Kernel was launched over a grid consisting of %d blocks, \\n", gridDim.x);
            printf("Where each block has %d threads. \\n", blockDim.x);
        }
    }
    """)

    # Instantiate the kernel

    ker_dim = ker.get_function("dimensions")
    ker_dim(block=(5, 1, 1), grid=(2, 1, 1))


def dot_prod_debug():

    # Dot product kernel

    ker = SourceModule("""
    
    __device__ float rowcol_dot(float *matrix_a, float *matrix_b, int row, int col, int N)
    {
        float val = 0;
        
        for (int k=0; k < N; k++)
        {
            val += matrix_a[ row*N + k ] * matrix_b[ col + k*N];
            if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            {
                printf("Dot-product loop: k value is %d, matrix_a value is %f, matrix_b is %f.\\n", k,
                matrix_a[ row*N + k], matrix_b[ col + k*N]);
            }
        }
        
        return(val);
    }    
        
    __global__ void matrix_mul_ker(float * matrix_a, float * matrix_b, float * output_matrix, int N)
    {
        int row = blockIdx.x*blockDim.x + threadIdx.x;
        int col = blockIdx.y*blockDim.y + threadIdx.y;
        printf("threadIdx.x,y: %d,%d blockIdx.x,y: %d,%d -- row is %d, col is %d, N is %d.\\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, N);
        __syncthreads();
        output_matrix[col + row*N] = rowcol_dot(matrix_a, matrix_b, row, col, N);

    }
    """)

    dot_ker = ker.get_function("matrix_mul_ker")

    matrix_a = np.float32([range(1, 5)]*4)
    matrix_b = np.float32([range(14, 10, -1)]*4)

    output_mat  = np.matmul(matrix_a, matrix_b)

    matrix_a_gpu = gpuarray.to_gpu(matrix_a)
    matrix_b_gpu = gpuarray.to_gpu(matrix_b)

    output_mat_gpu = gpuarray.empty_like(matrix_a_gpu)

    # call the kernel for dot product

    dot_ker(matrix_a_gpu, matrix_b_gpu, output_mat_gpu, np.int32(4), block=(2,2,1), grid=(2,2,1))

    assert(np.allclose(output_mat_gpu.get(), output_mat))








if __name__ == "__main__":

    #hello_world_example()
    dot_prod_debug()




