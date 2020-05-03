#cuRAND

"""
* Used for generating pseud-random numbers.
* Why pseudo-random ? -> since the digital
hardware is always deterministic and never random or arbitrary, we use
algorithms to generate a sequence of apparently random values from an
initial seed value.

* Initial seed(deterministic) is used to generate a random value.

* Different Seeds may have a correlation between the random values generated.

* Values generated by cuRAND will only be truly mathematically
mutually random if they are generated by the same seed value.

* For N thread to have different value of random number , we will use the same seed to
assign sequence of random values for that thread.

* For a sequence of random numbers, sub-sequences are assigned to each thread
and a pointer is used to refer to which pseudo-random number is getting referred in this
thread.

*


"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from sympy import Rational

ker = SourceModule(no_extern_c=True, source='''
#include <curand_kernel.h>
#define _PYTHAG(a,b)  (a*a + b*b)
#define ULL  unsigned long long
extern "C" {
__global__ void estimate_pi(ULL iters, ULL * hits)
{
	curandState cr_state;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init( (ULL)  clock() + (ULL) tid, (ULL) 0, \
	(ULL) 0, &cr_state);
	float x, y;

	for(ULL i=0; i < iters; i++)
	{ 
		 x = curand_uniform(&cr_state);
		 y = curand_uniform(&cr_state);


		 if(_PYTHAG(x,y) <= 1.0f)
			 hits[tid]++;
	}

 return;
}
}// (End of 'extern "C"' here)
''')

pi_ker = ker.get_function("estimate_pi")

threads_per_block = 32
blocks_per_grid = 512

total_threads = threads_per_block * blocks_per_grid

hits_d = gpuarray.zeros((total_threads,), dtype=np.uint64)

iters = 2 ** 24

pi_ker(np.uint64(iters), hits_d, grid=(blocks_per_grid, 1, 1), block=(threads_per_block, 1, 1))

total_hits = np.sum(hits_d.get())
total = np.uint64(total_threads) * np.uint64(iters)

est_pi_symbolic = Rational(4) * Rational(int(total_hits), int(total))

est_pi = np.float(est_pi_symbolic.evalf())

print("Our Monte Carlo estimate of Pi is : %s" % est_pi)
print("NumPy's Pi constant is: %s " % np.pi)

print("Our estimate passes NumPy's 'allclose' : %s" % np.allclose(est_pi, np.pi))