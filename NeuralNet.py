import time
import random
import numpy as np 

from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from activation_functions import sigmoid_function, tanh_function, linear_function,\
								 LReLU_function, ReLU_function
from tools import add_bias

# Data specific
image_length = 86*86
image_width = 86
image_height = 86


default_settings = {
	# Default optional settings
	"weights_low"				: -0.1, 		# Lower bound on initial weight range
	"weights_high"				: 0.1,		# Upper bound on initial weight range

	"batch-size"				: 1 		# 1 := stochastic gradient descent, 0 := gradient descent
}

class NeuralNetwork(object):
	def __init__(self, settings):
		self.__dict__.update( default_settings )
		self.__dict__.update( settings )

		self.weight_layers = self.setWeights( self.weights_low, self.weights_high )
	
	# end init

	def setWeights(self, low, high):
		if (self.n_hidden_layers == 0):
			weight_layers = [ np.random.uniform(low, high, size=(self.n_inputs+1,self.n_outputs)) ]
		else:
			weight_layers = [ np.random.uniform(low, high, size=(self.n_inputs+1,self.n_hiddens)) ]
			for i in xrange(self.n_hidden_layers-1):
				weight_layers += [ np.random.uniform(low, high, size=(self.n_hiddens+1,self.n_hiddens)) ]
			weight_layers += [ np.random.uniform(low, high, size=(self.n_hiddens+1,self.n_outputs)) ]
		return weight_layers
	# end setWeights

	def forward(self, X):
		# Forward propogate inputs through the network 
		(N, d) = X.shape

		X_bias = add_bias(X)

		# Calculate first activation
		self.Z2 = np.dot(X_bias, self.weight_layers[0])

		if (self.n_hidden_layers == 0):
			# Return activation on Z2
			return self.activation_functions[0]( self.Z2 , False)
		else: 

			self.A2 = self.activation_functions[0]( self.Z2 , False)
			A2_bias = add_bias(self.A2)

			self.Z3 = np.dot(A2_bias, self.weight_layers[1])
			yHat = self.activation_functions[1]( self.Z3, False)

			# Final layer output
			return yHat

	def backprop(self, X, y):
		yHat = self.forward(X)
		error = yHat - y

		delta = error
		MSE = np.mean( np.power(error, 2))

		X_bias = add_bias(X)

		if (self.n_hidden_layers == 0):
			delta2 = np.multiply(delta, self.activation_functions[0]( self.Z2 , True))

			# Compute final gradient
			dJdW1 = np.dot(X_bias.T, delta_final)
			return [ dJdW1 , MSE]
		else:
			delta3 = np.multiply(delta, self.activation_functions[1]( self.Z3 , True))
			dJdW2 = np.dot(add_bias(self.A2).T, delta3)

			# Pass backward
			delta2 = np.multiply(np.dot(delta3, self.weight_layers[1][0:self.n_hiddens].T), 
							self.activation_functions[0]( self.Z2 , True))
			dJdW1 = np.dot(X_bias.T, delta2)

			return [(dJdW1, dJdW2), MSE]

	def train(self, X, y, ERROR_LIMIT = 1e-3, learning_rate=0.3):
		X_train, X_val, y_train, y_val = train_test_split(
				X, y, test_size=0.2, random_state=42)

		MSE 			= float('inf')
		batch_size 		= self.batch_size if self.batch_size != 0 else X_train.shape[0]
		epoch 			= 0

		while MSE > ERROR_LIMIT:
			epoch += 1

			time = 0
			for start in xrange( 0, X_train.shape[0], batch_size ):
				time+=1
				# Calculate gradient and current MSE
				grad_list, MSE = self.backprop(X_train[start:start+batch_size], y_train[start:start+batch_size])

				# Store grad list for image alteration
				self.grad_list = grad_list

				# Update parameter weights
				for i in xrange(len(self.weight_layers)):
					self.weight_layers[i] -= learning_rate*grad_list[i]

		# Show the current training status
		print "* current epoch error (MSE):", MSE

	def alter_image(self, image, label, ERROR_LIMIT = 1e-3):

		grad_list = self.backprop(image, label)
		epoch = 0

		while MSE > ERROR_LIMIT:

			epoch +=1
			delta = 1
			for i in xrange(0, len(grad_list), -1):
				delta = np.dot(grad_list[i], delta)
			image -= np.swapaxes(delta, 0, 1)

			if epoch%10 == 0:
				img = np.reshape(image, (image_width, image_height))
				plt.imshow(img, cmap=cm.Greys_r)
				plt.show()



