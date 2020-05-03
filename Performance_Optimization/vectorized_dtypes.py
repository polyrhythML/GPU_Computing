from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

# To load int4 variables from an array of integers, and doubles from an array of
# doubles - uae reinterpret_cast operation
VecCode='''
__global__ void vec_ker(int *ints, double *doubles) {
int4 f1, f2;
f1 = *reinterpret_cast<int4*>(ints);
f2 = *reinterpret_cast<int4*>(&ints[4]);
printf("First int4: %d, %d, %d, %d\\n", f1.x, f1.y, f1.z, f1.w);
printf("Second int4: %d, %d, %d, %d\\n", f2.x, f2.y, f2.z, f2.w);
double2 d1, d2;
d1 = *reinterpret_cast<double2*>(doubles);
d2 = *reinterpret_cast<double2*>(&doubles[2]);
printf("First double2: %f, %f\\n", d1.x, d1.y);
printf("Second double2: %f, %f\\n", d2.x, d2.y);
}'''

