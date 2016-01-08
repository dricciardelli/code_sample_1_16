from activation_functions import sigmoid_function, tanh_function, linear_function,\
								 LReLU_function, ReLU_function

from NeuralNet import NeuralNetwork
from tools import load_data
import numpy as np

# Load data from Hot_or_Not website scrape
male_images, male_scores, fem_images, fem_scores = load_data()
image_length = male_images.shape[1]

settings = {

	# Preset Parameters
	"n_inputs" 				:  image_length, 		# Number of input signals
	"n_outputs"				:  1, 					# Number of output signals from the network
	"n_hidden_layers"		:  1,					# Number of hidden layers in the network (0 or 1 for now)
	"n_hiddens"				:  100,   				# Number of nodes per hidden layer
	"activation_functions"	:  [ LReLU_function, sigmoid_function ],		# Activation functions by layer

	# Optional parameters

	"weights_low"			: -0.1,		# Lower bound on initial weight range
	"weights_high"			: 0.1,  	# Upper bound on initial weight range
	"save_trained_network"  : False,	# Save trained weights or not.

	"batch_size"			: 1, 		# 1 for stochastic gradient descent, 0 for gradient descent
}

# Initialization
network = NeuralNetwork( settings )


# Train
network.train( 				fem_images, fem_scores, 	# Trainingset
							ERROR_LIMIT = 1e-3,			# Acceptable error bounds
							learning_rate	= 1e-5,		# Learning Rate
						)

# Alter image

network.alter_image(		fem_images[0], 				# Image to alter
							fem_scores[0]				# Label for initial backprop
						)