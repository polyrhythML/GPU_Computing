
"""
* The Game of Life (often called LIFE for short) is a cellular automata simulation that was invented by the
British mathematician John Conway back in 1970.

* This sounds complex, but it's really quite simple—LIFE is a zero-player game that consists of a two-dimensional
binary lattice of cells that are either considered live or dead.

* The lattice is iteratively updated by the following set of rules:
1.Any live cell with fewer than two live neighbors dies
2.Any live cell with two or three neighbors lives
3.Any live cell with more than three neighbors dies
4.Any dead cell with exactly three neighbors comes to life

* This is parallelizable, as it is clear that each cell in the lattice can be managed by a single CUDA thread.

A CUDA device function is a serial C function that is called by an individual CUDA thread from within a kernel.
While these functions are serial in themselves, they can be run in parallel by multiple GPU threads.
Device functions cannot by themselves by launched by a host computer onto a GPU, only kernels.

Need to refer to the code_base at :
https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA/blob/master/Chapter04/conway_gpu.py
"""