import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cublas
from time import time

# STANDARD CUDA LIBRARIES FOR MATHEMATICAL OPERATIONS

"""
* cuBLAS - cuda Basic Linear Algebra Subprograms
* cuFFT - cuda Fast fourier transform - > convolution filters
* cuSolver - wrapper over cuBLAS - > Cholesky factorization
* scikit-CUDA - a user-friendly wrapper interface to these libraries.

"""


######################## cuBLAS ########################

"""
* BLAS is divided into several levels of operations
    Level 1 - vector addition , dot prod, norms(ax + y -> AXPY)
    Level 2 - matrix vector operations(GEMV)
    Level 3 - matrix-matrix operations(GEMM)

* Scikit-CUDA provides wrappers for cuBLAS that are compatible with
PyCUDA gpuarray​ objects, as well as with PyCUDA streams. This means that
we can couple and interface these functions with our own custom CUDA-C
kernels by way of PyCUDA, as well as synchronize these operations over
multiple streams.

* Always remember that BLAS and CuBLAS functions act in-place to save
time and memory from a new allocation call. This means that an input
array will also be used as an output!

* Single precision - 32 Bit floating point number - cublasSaxpy() 
* Double precision - 64 Bit floating point number - cublasDaxpy()
* Complex64 - 64 bit single precision complex values - cublasCaxpy()
* Complex128 - Double precision complex values - cublasZaxpy()

* If output of a function is a single value as opposed to an array, the 
function will directly output this value to the host rather than within an
array of memory that has to be pulled from the GPU.


"""

# Level-1 operations illustration
def level_1():

    # cublas.cublasSdot - for level 1 dot product
    # cublas.cublasSnrm2 - for level 1 L2 norm

    # Peform ax + b operation
    a = np.float32(10)
    x = np.float32([1, 2, 3])
    y = np.float32([-3.45, 8.15, -15.867])

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)


    # Create a cuBLAS context, only used to handle cuBLAS session

    cublas_context_h = cublas.cublasCreate()

    # Use cublas.Saxpy to operation the level 1 operation
    cublas.cublasSaxpy(cublas_context_h, x_gpu.size, a, x_gpu.gpudata,
                       1, y_gpu.gpudata, 1)
    cublas.cublasDestroy(cublas_context_h)
    print('This is close to the NumPy approximation: %s' % np.allclose(a*x + y , y_gpu.get()))


# Level-2 operations illustration GEMV
def level_2():

    # y = a1*X*Y + b1*Z
    # a1 - constant  scalar, b1 - constant scalar
    # X  - Matrix, Y - Vector, Z - vector
    """
    cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    * handle - cuda context objet
    * trans - specify whether we want to use the original matrix, a direct
    transpose, or a conjugate transpose (for complex matrices). This is
    important to keep in mind because this function will expect that the
    matrix A is stored in column-major​ format.
    * m, n - the number of rows and columns of the matrix A
    * alpha - a1 - floating point value for a.
    * A - Matrix - X
    * lda - lda​ indicates the leading dimension of the matrix, where the total size
            of the matrix is actually l ​ da​ x ​ n ​ . This is important in the column-major
            format because if lda​ is larger than m, this can cause problems for
            cuBLAS when it tries to access the values of A since its underlying
            structure of this matrix is a one-dimensional array.
    * x - is Y from the above equation, should be of size n
    * incx - stride of the x
    * beta - is b1 from the above equation, floating point scalar
    * y - Z vector from the above equation, size of m
    * y - output - is an in-place operation.

    """
    m = 10
    n = 1000
    alpha = 1
    beta = 0
    A = np.random.rand(m, n).astype("float32")
    x = np.random.rand(n).astype("float32")
    y = np.zeros(m).astype("float32")

    # Numpy default is row major storage
    A_columnwise = A.T.copy()
    A_gpu = gpuarray.to_gpu(A_columnwise)
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    # Since the matrix A is already stored in the column-wise fashion
    # we disable the trans variable of the cuBLAS
    trans = cublas._CUBLAS_OP["N"]

    lda = m
    incx = 1
    incy = 1

    handle = cublas.cublasCreate()
    cublas.cublasSgemv(handle, trans, m, n, alpha, A_gpu.gpudata, lda, x_gpu.gpudata, incx,
                       beta, y_gpu.gpudata, incy)
    cublas.cublasDestroy(handle)
    print('cuBLAS returned the correct value: %s' % np.allclose(np.dot(A, x), y_gpu.get()))


# Level - 3 illustration GEMM
def level_3(precision="S"):

    if precision =="S":
        float_type ="float32"
    elif precision == "D":
        float_type = "float64"
    else:
        return -1

    # C <- alpha*A*B + beta*C
    """
    A - mxn
    B - nxk
    C - mxn
    Total FLOPS = 2mn(k+1)
    FLOPs/second -> FLOPs/(total time to run the matrix-matrix operation)


    """
    m = 5000
    n = 10000
    k = 10000

    A = np.random.rand(m, k).astype(float_type)
    B = np.random.rand(k, n).astype(float_type)
    C = A = np.random.rand(m, n).astype(float_type)
    A_cm = A.T.copy()
    B_cm = B.T.copy()
    C_cm = C.T.copy()
    A_gpu = gpuarray.to_gpu(A_cm)
    B_gpu = gpuarray.to_gpu(B_cm)
    C_gpu = gpuarray.to_gpu(C_cm)
    alpha = np.random.randn()
    beta = np.random.randn()
    transa = cublas._CUBLAS_OP["N"]
    transb = cublas._CUBLAS_OP["N"]
    lda = m
    ldb = k
    ldc = m
    t = time()

    handle = cublas.cublasCreate()
    exec('cublas.cublas%sgemm(handle, transa, transb, m, n, k, alpha, A_gpu.gpudata, lda,B_gpu.gpudata, ldb, beta, '
         'C_gpu.gpudata, ldc)' % precision)
    cublas.cublasDestroy(handle)
    t = time() - t
    gflops = 2 * m * n * (k + 1) * (10 ** -9) / t

    return gflops


if __name__ =="__main__":

    level_1()
    level_2()
    print('Single-precision performance: %s GFLOPS' % level_3('S'))
    print('Double-precision performance: %s GFLOPS' % level_3('D'))




