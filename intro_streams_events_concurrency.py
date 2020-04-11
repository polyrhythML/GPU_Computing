# INTRODUCTION

"""
* We saw one level of concurrency i.e. synchronization of threads.
* There is another level of concurrency available multiple kernels and GPU memory operations. We can launch multiple
memory operations and kernel operation at once, without waiting for each operation to finish.
* We should not launch a kernel until all it's input is copied to the device memory or shouldn't copy the output of a
launched kernel to the host until the kernel has finished execution.

* CUDA STREAM : a stream is a sequence of operations that are run in order on the GPU. By itself, a single stream isn't
of any use—the point is to gain concurrency over GPU operations issued by the host by using multiple streams.
This means that we should interleave launches of GPU operations that correspond to different streams, in order to
exploit this notion.

* EVENTS : feature of streams that are used to precisely time kernels and indicate to the host as to what operations
have been completed within a given stream.

* CONTEXT : A context can be thought of as analogous to a process in your operating system, in that the GPU
keeps each context's data and kernel code walled off and encapsulated away from the other contexts currently
existing on the GPU.

"""