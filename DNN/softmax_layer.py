from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

# Softmax layer
"""
* A softmax layer​ is used when you only want to assign a single class
to a sample by inference—this is done by computing a probability
for each possible class (with probabilities over all classes, of course,
summing to 100%). We can then select the class with the highest probability
to give the final classification.


"""


# threads: at least "num"
SoftmaxExpCode = '''
__global__ void softmax_exp( int num, float *x, float *y, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num)
    {
        for (int k=0; k < batch_size; k++)
        {
            y[num*k + i] = expf(x[num*k+i]);

        }
    }
}
'''
exp_mod = SourceModule(SoftmaxExpCode)
exp_ker = exp_mod.get_function('softmax_exp')

# threads: at least batch size
SoftmaxMeanCode = '''
__global__ void softmax_mean( int num, float *x, float *y, int batch_size)
{
    // parallelize over
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < batch_size)
    {
        float temp = 0.0f;

        for(int k=0; k < num; k++)
            temp += x[i*num + k];


        for(int k=0; k < num; k++)
            y[i*num+k] = x[i*num+k] / temp;

    }

    return;
}'''

mean_mod = SourceModule(SoftmaxMeanCode)
mean_ker = mean_mod.get_function('softmax_mean')


class SoftmaxLayer:

    def __init__(self, num=None, stream=None):
        self.num = np.int32(num)
        self.stream = stream

    def eval_(self, x, y=None, batch_size=None, stream=None):

        if stream is None:
            stream = self.stream

        if type(x) != pycuda.gpuarray.GPUArray:
            temp = np.array(x, dtype=np.float32)
            x = gpuarray.to_gpu_async(temp, stream=stream)

        if batch_size == None:
            if len(x.shape) == 2:
                batch_size = np.int32(x.shape[0])
            else:
                batch_size = np.int32(1)
        else:
            batch_size = np.int32(batch_size)

        if y is None:
            if batch_size == 1:
                y = gpuarray.empty((self.num,), dtype=np.float32)
            else:
                y = gpuarray.empty((batch_size, self.num), dtype=np.float32)

        exp_ker(self.num, x, y, batch_size, block=(32, 1, 1), grid=(int(np.ceil(self.num / 32)), 1, 1), stream=stream)

        mean_ker(self.num, y, y, batch_size, block=(32, 1, 1), grid=(int(np.ceil(batch_size / 32)), 1, 1),
                 stream=stream)

        return y

