"""
* The Driver API is slightly different and a little more technical than the ​ CUDA
Runtime API​ , the latter being what we have been working within this text
from CUDA-C. The Driver API is designed to be used with a regular C/C++
compiler rather than with NVCC, with some different conventions.

*

"""

from ctypes import *
import sys
if 'linux' in sys.platform:
    cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
    cuda = CDLL('nvcuda.dll')

# Initialize cuda
cuInit = cuda.cuInit
# input argument types
cuInit.argtypes = [c_uint]
# return type
cuInit.restype = int

# Count the number of CUDA devices
cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

# wrapper for cuDeviceGet
cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

"""
Let's remember that every CUDA session will require at least one CUDA
Context, which can be thought of as analogous to a process running on the
CPU. Since this is handled automatically with the Runtime API, here we
will have to create a context manually on a device (using a device handle)
before we can use it, and we will have to destroy this context when our
CUDA session is over.
"""
"""
* The first input is a pointer to a type called CUcontext, which is actually 
itself a pointer to a particular C structure used internally by CUDA. 
Since our only interaction with CUcontext​from Python will be to hold onto 
its value to pass between other functions, we can just store CUcontext​as a C void *​type, 
which is used to store a generic pointer address for any type. 
* Since this is actually a pointer to a CU context(again, which is 
itself a pointer to an internal data structure—this is another pass-by-reference return value), 
we can set the type to be just a plain void *, which is a c_void_p​type in Ctypes.

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev)
"""
# create context object
cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
cuCtxCreate.restype = int

#You can always use the void * type in C/C++ ( c_void_p in Ctypes) to point to
#any arbitrary data or variable—even structures and objects whose
#definition may not be available.

# create module to load PTX file for us.
cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype = int

# Synchronize all launched operations over the current CUDA context.
cuCtxSynchronize = cuda.cuCtxSynchronize
cuCtxSynchronize.argtypes = []
cuCtxSynchronize.restype = int

# launch function, to get a kernel function handle from a loaded module so that
# we launch it onto the GPU, similar to what get_function did in CUDA.

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p]
cuModuleGetFunction.restype = int

# Let's write dynamic memory allocation operations wrapper

cuMemAlloc = cuda.cuMemAlloc
cuMemAlloc.argtypes = [c_void_p, c_size_t]
cuMemAlloc.restype = int

# Host to device transfer
cuMemcpyHtoD = cuda.cuMemcpyHtoD
cuMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemAlloc.restype = int

# Device to host transfer
cuMemcpyDtoH = cuda.cuMemcpyDtoH
cuMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]
cuMemcpyDtoH.restype = int

# Free memory
cuMemFree = cuda.cuMemFree
cuMemFree.argtypes = [c_void_p]
cuMemFree.restype = int

"""
* This is what we will use to launch a CUDA kernel onto the GPU, provided that
we have already initialized the CUDA Driver API, set up a context, loaded a
module, allocated memory and configured inputs, and have extracted the
kernel function handle from the loaded module.


CUresult cuLaunchKernel ( CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int
blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void**
extra )
"""


cuLaunchKernel = cuda.cuLaunchKernel
cuLaunchKernel.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
c_uint, c_void_p, c_void_p, c_void_p]
cuLaunchKernel.restype = int

# Destroy cuda session
cuCtxDestroy = cuda.cuCtxDestroy
cuCtxDestroy.argtypes = [c_void_p]
cuCtxDestroy.restype = int

## Driver API wrapper is complete!!
