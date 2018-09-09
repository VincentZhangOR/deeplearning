"""
He Zhang
"""

from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np
import math

hidden_units = 10
learning_rate = 1
momentum = 0
l2_penalty = 0

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	# DEFINE __init function
        self.W = W
        self.b = b

    def forward(self, x):
	# DEFINE forward function
        return np.dot(self.W, x) + self.b

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
	# DEFINE backward function
        return grad_output

# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        return np.maximum(x, 0)

    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
    # DEFINE backward function
        if grad_output > 0:
            return np.vectorize(1)
        elif grad_output == 0:
            return np.vectorize(0.5)
        else:
            return np.vectorize(0)

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

	def forward(self, x):
		# DEFINE forward function
        z = 1/(1+np.exp(-x))
        return z
        # output_sigmoid_c2 = 1/(1+np.exp(x))
        # return output_sigmoid_c1 if output_sigmoid_c1 >= output_sigmoid_c2 else output_sigmoid_c2

	def backward(self, y, z, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
		# DEFINE backward function
        return (y/z - (1-y)/(1-z)) * (z*(1-z))



# ADD other operations and data entries in SigmoidCrossEntropy if needed
    # def calculate_loss(self, z):
    #     return self.y * log(z) + (1-self.y) * log(1-z)


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.w1 = np.random.rand(input_dims，hidden_units)
        self.b1 = np.random.rand(1, hidden_units)
        self.w2 = np.random.rand(hidden_units，1)
        self.b2 = np.random.rand(1, 1)
        self.delta_w1 = np.zeros(input_dims，hidden_units)
        self.delta_b1 = np.zeros(1，hidden_units)
        self.delta_w2 = np.zeros(hidden_units，1)
        self.delta_b2 = np.zeros(1,1)

    def clear(self):
        self.delta_w1 = np.zeros(input_dims，hidden_units)
        self.delta_b1 = np.zeros(1，hidden_units)
        self.delta_w2 = np.zeros(hidden_units，1)
        self.delta_b2 = np.zeros(1,1)

    def normalize(self, x):
        return (x-np.mean(x)) / np.std(x)

    def bound(self, x):
        if x >= 1:
            return 0.99999
        elif x <= 0:
            return 0.00001
        return np.vectorize(x)

    def calculate_loss(self, y, z):
        return y * np.log(z) + (1-y) * np.log(1-z)

    def train(self, x_batch, y_batch, learning_rate, momentum, l2_penalty):
	# INSERT CODE for training the network
        f1_x = LinearTransform(self.w1, self.b1)
        g_x = ReLU()
        f2_x = LinearTransform(self.w2, self.b2)
        h_x = SigmoidCrossEntropy()

        x_batch = normalize(x_batch)
        hidden_i = f1_x.forward(x_batch)
        hidden_o = g_x.forward(hidden_i)
        output_i = f2_x.forward(hidden_o)
        output_o = h_x.forward(output_i)

        output_o = bound(output_o)

        loss = calculate_loss(y_batch[0], output_o[0])

        h_x_backward = h_x.backward(y_batch[0], output_o[0], learning_rate, momentum, l2_penalty)
        self.delta_w2 += h_x_backward * hidden_o[0]
        self.delta_b2 += h_x_backward
        g_x_backward = g_x.backward(hidden_i, learning_rate, momentum, l2_penalty)
        self.delta_w1 += h_x_backward * self.w2 * g_x_backward * x_batch
        self.delta_b1 += h_x_backward * self.w2 * g_x_backward

        return loss

    def update(self, num_batches):
        self.w1 = momentum * self.w1 - learning_rate * self.delta_w1 / num_batches
        self.b1 = momentum * self.b1 - learning_rate * self.delta_b1 / num_batches
        self.w2 = momentum * self.w2 - learning_rate * self.delta_w2 / num_batches
        self.b2 = momentum * self.b2 - learning_rate * self.delta_b2 / num_batches
        mlp.clear()

    def evaluate(self, x, y):
	# INSERT CODE for testing the network
        f1_x = LinearTransform(self.w1, self.b1)
        g_x = ReLU()
        f2_x = LinearTransform(self.w2, self.b2)
        h_x = SigmoidCrossEntropy()
        size = len(y)
        total_loss = 0
        correct = 0

        for i in range(len(x)):
            x_batch = normalize(x[i])
            hidden_i = f1_x.forward(x_batch)
            hidden_o = g_x.forward(hidden_i)
            output_i = f2_x.forward(hidden_o)
            output_o = h_x.forward(output_i)
            output_o = bound(output_o)
            loss = calculate_loss(y[i][0], output_o[0])
            total_loss += loss
            if abs(y[i][0] - output_o[0]) < 0.5:
                correct += 1

        return total_loss / size, correct / size


# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    num_examples, input_dims = train_x.shape
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	num_epochs = 10
	num_batches = 1000
    mlp = MLP(input_dims, hidden_units)

    for epoch in xrange(num_epochs):
	# INSERT YOUR CODE FOR EACH EPOCH HERE
        sample = np.random.randint(num_examples, size = num_batches)
        x_batch = train_x[sample,:]
        y_batch = train_y[sample,:]

        for b in xrange(num_batches):
			total_loss = 0.0
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            total_loss = mlp.train(x_batch[b], y_batch[b], learning_rate, momentum, l2_penalty)
			# MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        mlp.update(num_batches)
        train_loss, train_accuracy = mlp.evaluate(train_x, train_y)
        test_loss, test_accuracy = mlp.evaluate(test_x, test_y)
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
