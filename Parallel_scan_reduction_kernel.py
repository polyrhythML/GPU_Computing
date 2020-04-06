import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel




def example1():
    seq = np.array([1,2,3,4],dtype=np.int32)
    seq_gpu = gpuarray.to_gpu(seq)
    sum_gpu = InclusiveScanKernel(np.int32, "a+b")
    print(sum_gpu(seq_gpu).get())


def example2():
    seq = np.array([1, 100, -3, -10000, 4, 10000, 66, 14, 21], dtype=np.int32)
    seq_gpu = gpuarray.to_gpu(seq)
    max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
    print(max_gpu(seq_gpu).get()[-1])
    print(np.max(seq))

if __name__ == "__main__":

    example1()
    example2()

