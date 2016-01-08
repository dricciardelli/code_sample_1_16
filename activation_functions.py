from scipy.special import expit
import numpy as np 

######### Activation Functions ######

def sigmoid_function( signal, derivative=False ):
	# Calculate sigmoid function activation

	# Enforce numerical stability
	signal = np.clip( signal, -500, 500 )

	# Calculate activation
	signal = expit( signal )

	if derivative:
		# Return partial derivative, y*(1-y)
		return np.multiply(signal, 1.0 -signal)
	else:
		# Return the activation signal
		return signal
# end sigmoid function

def ReLU_function( signal, derivative=False):
	# Calculate Rectified Linear Unit activation
	if derivative:
		# Enforce linear constraint
		derivate = np.maximum( 0, signal )
		# Calculate derivative
		derivate[ derivate != 0 ] = 1.0
		return derivate
	else:
		# Return the activation signal
		return np.maximum( 0, signal )
# end ReLU activation function

def LReLU_function( signal, derivative=False):
	# Calculate Leaky Rectified Linear Unit activation
	if derivative:
		# Enforce linear constraint
		derivate = np.maximum( 0, signal )
		# Calculate derivative
		derivate[ derivate < 0 ] = 0.01
		derivate[ derivate > 0 ] = 1.0
		return derivate
	else:
		# Return Activation signal
		output = np.copy( signal )
		output[ output < 0 ] *= 0.01
		return output
# end Leaky Rectified Linear Unit activation

def step_function( signal, derivative=False):
	# Calculate step function activation (only used to check basic functionality)

	if derivative:
		# Derivative of step function is always zero
		zeroed_signal = signal.fill(0)
		return zeroed_signal
	else:
		signal = np.maximum(0, signal)
		signal[ signal != 0 ] = 1
		return signal
# end step activation function

def tanh_function( signal, derivative=False):
	# Calculate good old tanh activation function
	signal = np.tanh( signal )

	if derivative:
		# Return the partial derivative of the activation function
		return 1-np.power(signal,2)
	else:
		# Return the activation signal
		return signal
# end tanh activation function

def linear_function( signal, derivative=False ):
	if derivative:
		# Return the partial derivation of the activation function
		return 1
	else:
		# Return the activation signal
		return signal
# end linear activation function

