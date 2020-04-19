# Why GPU programming
"""
General purpose GPU programming can help one get along with the capabilities of the GPU's parallelism power to solve
problems efficiently. It opens up people from different domains such as genetics, biologists use GPU for DNA analysis,
physicists and mathematician use GPUs for large scale simulations.

"""

# Overview

"""
*  An individual GPU core is actually quite simplistic, and at a disadvantage when compared to a modern individual CPU 
core, which use many fancy engineering tricks, such as branch prediction to reduce the latency of computations. 
Latency refers to the beginning-to-end duration of performing a single computation.

* The power of the GPU derives from the fact that there are many, many more cores than in a CPU, which means a huge step
 forward in throughput. Throughput here refers to the number of computations that can be performed simultaneously. 
 Let's use an analogy to get a better understanding of what this means.
 
* The average Intel or AMD CPU has only two to eight cores—while an entry-level, consumer-grade NVIDIA GTX 1050 GPU has 
640 cores, and a new top-of-the-line NVIDIA RTX 2080 Ti has 4,352 cores.

* We can exploit this massive throughput, provided we know how properly to parallelize any program or algorithm we wish 
to speed up. By parallelize, we mean to rewrite a program or algorithm so that we can split up our workload to run in 
parallel on multiple processors simultaneously. Let's think about an analogy from real-life.

"""

# Amdahl's Law

"""
* Out of the total code if p is the proportion of the code that is parallelizable then 1-p is the part of the code that
is not parallelizable. Then the total theoretical speedup possible with parallelized code is given by : 

                                   Speedup = 1/((1-p) + p/N),
                                         
where N is the number of cores over which you are going to parallelize your parallelizable code.
"""

# Summary

"""

* The main advantage of using a GPU over a CPU is its increased throughput, which means that we can execute more 
parallel code simultaneously on GPU than on a CPU; a GPU cannot make recursive algorithms or nonparallelizable 
algorithms somewhat faster.

* cProfiler can be used in general to find out the bottleneck areas in your code and whether these bottleneck areas can
be optimized or not.


"""