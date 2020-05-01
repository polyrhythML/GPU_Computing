"""
Let's first start by writing the constructor—we
will be able to set the max batch size here, which will affect how much
memory is allocated for the use of this network – we'll store some allocated
memory used for weights and input/output for each layer in the list
variable, network_mem​ . We will also store the D
enseLayer​ and S ​ oftmaxLayer​ objects
in the list network, and information about each layer in the NN in
network_summary​ . Notice how we can also set up some training parameters
here, including the delta, how many streams to use for gradient descent,
as well as the number of training epochs.

"""
import numpy as np
from cross_entropy import cross_entropy
from dense_layer import DenseLayer
from softmax_layer import SoftmaxLayer
from queue import Queue
import pycuda.driver as drv
from pycuda import gpuarray


class SequentialNetwork:

    def __init__(self, layers=None, delta=None, stream=None, max_batch_size=32, max_streams=10, epochs=10):

        self.network = []
        self.network_summary = []
        self.network_mem = []

        if stream is not None:
            self.stream = stream
        else:
            self.stream = drv.Stream()

        if delta is None:
            delta = 0.0001

        self.delta = delta
        self.max_batch_size = max_batch_size

        self.max_streams = max_streams

        self.epochs = epochs

        if layers is not None:
            for layer in layers:
                self.add_layer(self, layer)

    def add_layer(self, layer):

        if layer['type'] == 'dense':
            if len(self.network) == 0:
                num_inputs = layer['num_inputs']
            else:
                num_inputs = self.network_summary[-1][2]

            num_outputs = layer['num_outputs']
            sigmoid = layer['sigmoid']
            relu = layer['relu']

            weights = layer['weights']

            b = layer['bias']

            self.network.append(
                DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs, sigmoid=sigmoid, relu=relu, weights=weights,
                           b=b))
            self.network_summary.append(('dense', num_inputs, num_outputs))

            if self.max_batch_size > 1:
                if len(self.network_mem) == 0:
                    self.network_mem.append(
                        gpuarray.empty((self.max_batch_size, self.network_summary[-1][1]), dtype=np.float32))
                self.network_mem.append(
                    gpuarray.empty((self.max_batch_size, self.network_summary[-1][2]), dtype=np.float32))
            else:
                if len(self.network_mem) == 0:
                    self.network_mem.append(gpuarray.empty((self.network_summary[-1][1],), dtype=np.float32))
                self.network_mem.append(gpuarray.empty((self.network_summary[-1][2],), dtype=np.float32))

        elif layer['type'] == 'softmax':

            if len(self.network) == 0:
                raise Exception("Error!  Softmax layer can't be first!")

            if self.network_summary[-1][0] != 'dense':
                raise Exception("Error!  Need a dense layer before a softmax layer!")

            num = self.network_summary[-1][2]

            self.network.append(SoftmaxLayer(num=num))

            self.network_summary.append(('softmax', num, num))

            if self.max_batch_size > 1:
                self.network_mem.append(
                    gpuarray.empty((self.max_batch_size, self.network_summary[-1][2]), dtype=np.float32))
            else:
                self.network_mem.append(gpuarray.empty((self.network_summary[-1][2],), dtype=np.float32))

    def predict(self, x, stream=None):

        if stream is None:
            stream = self.stream

        if type(x) != np.ndarray:
            temp = np.array(x, dtype=np.float32)
            x = temp

        if (x.size == self.network_mem[0].size):
            self.network_mem[0].set_async(x, stream=stream)
        else:

            if x.size > self.network_mem[0].size:
                raise Exception("Error: batch size too large for input.")

            x0 = np.zeros((self.network_mem[0].size,), dtype=np.float32)
            x0[0:x.size] = x.ravel()
            self.network_mem[0].set_async(x0.reshape(self.network_mem[0].shape), stream=stream)

        if (len(x.shape) == 2):
            batch_size = x.shape[0]
        else:
            batch_size = 1

        for i in range(len(self.network)):
            self.network[i].eval_(x=self.network_mem[i], y=self.network_mem[i + 1], batch_size=batch_size,
                                  stream=stream)

        y = self.network_mem[-1].get_async(stream=stream)

        if len(y.shape) == 2:
            y = y[0:batch_size, :]

        return y

    def partial_predict(self, layer_index=None, w_t=None, b_t=None, partial_mem=None, stream=None, batch_size=None,
                        delta=None):

        self.network[layer_index].eval_(x=self.network_mem[layer_index], y=partial_mem[layer_index + 1],
                                        batch_size=batch_size, stream=stream, w_t=w_t, b_t=b_t, delta=delta)

        for i in range(layer_index + 1, len(self.network)):
            self.network[i].eval_(x=partial_mem[i], y=partial_mem[i + 1], batch_size=batch_size, stream=stream)

    def bsgd(self, training=None, labels=None, delta=None, max_streams=None, batch_size=None, epochs=1,
             training_rate=0.01):

        training_rate = np.float32(training_rate)

        training = np.float32(training)
        labels = np.float32(labels)

        if (training.shape[0] != labels.shape[0]):
            raise Exception("Number of training data points should be same as labels!")

        if max_streams is None:
            max_streams = self.max_streams

        if epochs is None:
            epochs = self.epochs

        if delta is None:
            delta = self.delta

        streams = []
        bgd_mem = []

        # create the streams needed for training

        for _ in range(max_streams):
            streams.append(drv.Stream())
            bgd_mem.append([])

        # allocate memory for each stream

        for i in range(len(bgd_mem)):
            for mem_bank in self.network_mem:
                bgd_mem[i].append(gpuarray.empty_like(mem_bank))

        # begin training!

        num_points = training.shape[0]

        if batch_size is None:
            batch_size = self.max_batch_size

        index = list(range(training.shape[0]))

        for k in range(epochs):

            print('-----------------------------------------------------------')
            print('Starting training epoch: %s' % k)
            print('Batch size: %s , Total number of training samples: %s' % (batch_size, num_points))
            print('-----------------------------------------------------------')

            all_grad = []

            np.random.shuffle(index)

            for r in range(int(np.floor(training.shape[0] / batch_size))):

                batch_index = index[r * batch_size:(r + 1) * batch_size]

                batch_training = training[batch_index, :]
                batch_labels = labels[batch_index, :]

                batch_predictions = self.predict(batch_training)

                cur_entropy = cross_entropy(predictions=batch_predictions, ground_truth=batch_labels)

                print('entropy: %s' % cur_entropy)

                # need to iterate over each weight / bias , check entropy

                for i in range(len(self.network)):

                    if self.network_summary[i][0] != 'dense':
                        continue

                    all_weights = Queue()

                    grad_w = np.zeros((self.network[i].weights.size,), dtype=np.float32)
                    grad_b = np.zeros((self.network[i].b.size,), dtype=np.float32)

                    for w in range(self.network[i].weights.size):
                        all_weights.put(('w', np.int32(w)))

                    for b in range(self.network[i].b.size):
                        all_weights.put(('b', np.int32(b)))

                    while not all_weights.empty():

                        stream_weights = Queue()

                        for j in range(max_streams):

                            if all_weights.empty():
                                break

                            wb = all_weights.get()

                            if wb[0] == 'w':
                                w_t = wb[1]
                                b_t = None
                            elif wb[0] == 'b':
                                b_t = wb[1]
                                w_t = None

                            stream_weights.put(wb)

                            self.partial_predict(layer_index=i, w_t=w_t, b_t=b_t, partial_mem=bgd_mem[j],
                                                 stream=streams[j], batch_size=batch_size, delta=delta)

                        for j in range(max_streams):

                            if stream_weights.empty():
                                break

                            wb = stream_weights.get()

                            w_predictions = bgd_mem[j][-1].get_async(stream=streams[j])

                            w_entropy = cross_entropy(predictions=w_predictions[:batch_size, :],
                                                      ground_truth=batch_labels)

                            if wb[0] == 'w':
                                w_t = wb[1]
                                grad_w[w_t] = -(w_entropy - cur_entropy) / delta

                            elif wb[0] == 'b':
                                b_t = wb[1]
                                grad_b[b_t] = -(w_entropy - cur_entropy) / delta

                    all_grad.append([np.reshape(grad_w, self.network[i].weights.shape), grad_b])

            for i in range(len(self.network)):

                if self.network_summary[i][0] == 'dense':
                    new_weights = self.network[i].weights.get()
                    new_weights += training_rate * all_grad[i][0]
                    new_bias = self.network[i].b.get()
                    new_bias += training_rate * all_grad[i][1]
                    self.network[i].weights.set(new_weights)
                    self.network[i].b.set(new_bias)