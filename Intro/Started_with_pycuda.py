import pycuda
import pycuda.driver as drv

drv.init()

print('CUDA device query (PyCUDA version) \n')

print('Detected {} CUDA Capable device(s) \n'.format(drv.Device.count()))

for i in range(drv.Device.count()):

    gpu_device = drv.Device(i)
    print('Device {}: {}'.format(i, gpu_device.name()))
    compute_capability = float('%d.%d' % gpu_device.compute_capability())
    print('\t Compute Capability: {}'.format(compute_capability))
    print('\t Total Memory: {} megabytes'.format(gpu_device.total_memory() // (1024 ** 2)))

    # The following will give us all remaining device attributes as seen
    # in the original deviceQuery.
    # We set up a dictionary as such so that we can easily index
    # the values using a string descriptor.

    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}

    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v

    num_mp = device_attributes['MULTIPROCESSOR_COUNT']

    # Cores per multiprocessor is not reported by the GPU!
    # We must use a lookup table based on compute capability.
    # See the following:
    # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

    """
    A GPU divides its individual cores up into larger units known as Streaming Multiprocessors (SMs); a GPU device will 
    have several SMs, which will each individually have a particular number of CUDA cores, depending on the 
    compute capability of the device. To be clear: the number of cores per multiprocessor is not indicated directly by 
    the GPU—this is given to us implicitly by the compute capability.
    
    The compute capability of a device is represented by a version number, also sometimes called its "SM version". 
    This version number identifies the features supported by the GPU hardware and is used by applications at runtime to 
    determine which hardware features and/or instructions are available on the present GPU.
    
    The compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y.
    Devices with the same major revision number are of the same core architecture. The major revision number is 7 for 
    devices based on the Volta architecture, 6 for devices based on the Pascal architecture, 5 for devices based on the 
    Maxwell architecture, 3 for devices based on the Kepler architecture, 2 for devices based on the Fermi architecture,
    and 1 for devices based on the Tesla architecture.
    The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new
    features.
    
    Turing is the architecture for devices of compute capability 7.5, and is an incremental update based on the Volta 
    architecture.
    
    """

    cuda_cores_per_mp = {5.0: 128, 5.1: 128, 5.2: 128, 6.0: 64, 6.1: 128, 6.2: 128}[compute_capability]

    print('\t ({}) Multiprocessors, ({}) CUDA Cores / Multiprocessor: {} CUDA Cores'.format(num_mp, cuda_cores_per_mp,
                                                                                      num_mp * cuda_cores_per_mp))

    device_attributes.pop('MULTIPROCESSOR_COUNT')

    for k in device_attributes.keys():
        print('\t {}: {}'.format(k, device_attributes[k]))