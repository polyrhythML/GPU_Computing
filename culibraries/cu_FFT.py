# FAST FOURIER TRANSFORMS

"""
* Fast fourier transform is widely used not only in signal processing but also in image
analysis e.g. edge detection, image filtering, image reconstruction and image compression.

* Normal matrix-vector operation computational complexity - > N^2
* Due to symmetries in the DFT matrix, this can be reduced to O(NlogN) by using FFT


"""
from __future__ import division
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft
from skcuda import linalg
from matplotlib import pyplot as plt


def vector_fft():
    # Create a 1000 dimensional vector
    x = np.asarray(np.random.rand(1000), dtype=np.float32)
    x_gpu = gpuarray.to_gpu(x)
    x_hat = gpuarray.empty_like(x_gpu, dtype=(np.complex64))

    # plan object used to determine the shape, as well as input and output data types of the transform
    plan = fft.Plan(x_gpu.shape, np.float32, np.complex64)

    # Setup inverse plan object

    inverse_plan = fft.Plan(x.shape, np.complex64, np.float32)

    # Perform fft
    fft.fft(x_gpu, x_hat, plan)
    # Perform inverse fft
    fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)

    y = np.fft.fft(x)

    # compare the result from numpy operation and cufft operation

    print('cuFFT matches NumPy FFT: %s' % np.allclose(x_hat.get(), y, atol=1e-6))
    print('cuFFT inverse matches original: %s' % np.allclose(x_gpu.get(), x, atol=1e-6))
    """
    output 1 : False
    output 2 : True
    Reason : 
    While the NumPy FFT fully computes these values anyway, cuFFT saves time by 
    only computing the first half of the outputs when it sees that the input is 
    real, and it sets the remaining outputs to 0.
    """


def conv_fft(x, y):

    """
    * We apply a filter to a continuous signal to smooth out the signal, or
    extract relevant features from the signal.

    * We apply a circular convolution i.e. we are dealing with two length
    n-vectors whose indices below 0 or above n-1 will wrap around to the other end;
    that is to say, x[-1] = x[n-1],x[-2] = x[n-2], x [n] = x[0], x[n+1] = x [1],
    and so on.

    * Pefrom circular FFT and then perform inverse FFT on the final result.

    * We will create a gaussian smoothening filter here.

    """
    x = x.astype(np.complex64)
    y = y.astype(np.complex64)

    if(x.shape!=y.shape):
        return -1
    plan = fft.Plan(x.shape, np.complex64, np.complex64)
    inverse_plan = fft.Plan(x.shape, np.complex64, np.complex64)

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)
    x_fft = gpuarray.empty_like(x_gpu, dtype=np.complex64)
    y_fft = gpuarray.empty_like(y_gpu, dtype=np.complex64)
    out_gpu = gpuarray.empty_like(x_gpu, dtype=np.complex64)
    fft.fft(x_gpu, x_fft, plan)
    fft.fft(y_gpu, y_fft, plan)
    # hadamard product, element-wise
    linalg.multiply(x_fft, y_fft, overwrite=True)
    fft.ifft(y_fft, out_gpu, inverse_plan, scale=True)
    conv_out = out_gpu.get()
    return conv_out


def conv_2d(ker, img):

    padded_ker = np.zeros((img.shape[0] + 2 * ker.shape[0], img.shape[1] + 2 * ker.shape[1])).astype(np.float32)
    padded_ker[:ker.shape[0], :ker.shape[1]] = ker
    # Roll move the kernel back into the centre
    # // is floor division
    padded_ker = np.roll(padded_ker, shift=-ker.shape[0] // 2, axis=0)
    padded_ker = np.roll(padded_ker, shift=-ker.shape[1] // 2, axis=1)
    padded_img = np.zeros_like(padded_ker).astype(np.float32)
    padded_img[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]] = img

    out_ = conv_fft(padded_ker, padded_img)
    output = out_[ker.shape[0]:-ker.shape[0], ker.shape[1]:-ker.shape[1]]

    return output

# inline function to create gaussian filter operation
gaussian_filter = lambda x, y, sigma: (1 / np.sqrt(2*np.pi*(sigma**2)))*np.exp(-(x**2 + y**2) / (2 * (sigma**2)))


def gaussian_ker(sigma):
    ker_ = np.zeros((2*sigma+1, 2*sigma+1))
    for i in range(2*sigma + 1):
        for j in range(2*sigma + 1):
            ker_[i, j] = gaussian_filter(i - sigma, j - sigma, sigma)
    total_ = np.sum(ker_.ravel())
    ker_ = ker_ / total_
    return ker_


if __name__ == "__main__":

    kohli = np.float32(plt.imread('kohli.jpg')) / 255
    kohli_blurred = np.zeros_like(kohli)
    ker = gaussian_ker(15)
    for k in range(3):
        kohli_blurred[:, :, k] = conv_2d(ker, kohli[:, :, k])

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle('Gaussian Filtering', fontsize=20)
    ax0.set_title('Before')
    ax0.axis('off')
    ax0.imshow(kohli)
    ax1.set_title('After')
    ax1.axis('off')
    ax1.imshow(kohli)
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    plt.show()
