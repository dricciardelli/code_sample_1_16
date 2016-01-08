from scipy import ndimage, misc
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm

from scipy.io import loadmat
from numpy.matlib import repmat
from scipy import sparse
from scipy.sparse.linalg import svds

from sklearn.decomposition import PCA


# Load initial data
image_length = 86*86
image_width = 86
image_height = 86

def load_data():
	male_data 	= loadmat('./attractiveness_data/male_picts')
	fem_data 	= loadmat('./attractiveness_data/fem_picts')

	male_images = male_data['rectimgs'][0]
	image_length = male_images[0].shape[0]*male_images[0].shape[1]

	flat_images = np.zeros((male_images.shape[0], image_length))
	for i in xrange(male_images.shape[0]):
		flat_images[i] = np.reshape(rgb2gray(male_images[i]), (image_length))
	male_images = flat_images
	male_scores = male_data['score'][0]

	fem_images	= fem_data['rectimgs'][0]
	flat_images = np.zeros((fem_images.shape[0], image_length))
	for i in xrange(fem_images.shape[0]):
		flat_images[i] = np.reshape(rgb2gray(fem_images[i]), (image_length))
	fem_images = flat_images

	fem_scores	= fem_data['score'][0]

	return [male_images, male_scores, fem_images, fem_scores]

def rgb2gray(rgb):
	# Why is this not included in a numpy package? ...
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def image_tour( images, scores, n): 
	# Basic tour of images with user attractiveness ratings.
	for i in xrange(n):
		print "Image rating: " + str(scores[i])
		image = np.reshape(images[i], (image_width, image_height))
		plt.imshow(image,  cmap=cm.Greys_r)
		plt.show()
# end image_tour

#image_tour(male_images, male_scores)

def image_averages( images, scores):
	# Output attractive and unattractive image averages
	mean = np.mean(scores)
	std = np.std(scores)

	# Low scoring users
	low_images = images[ scores < mean - std]

	# High scoring users
	high_images = images[ scores > mean + std]

	# Output mean (hah?) face
	print("Average Face")
	mean = array_average(images)
	plt.imshow(mean, cmap=cm.Greys_r)
	plt.show()

	# Output average low scoring face
	print("Low scoring Face")
	low_mean = array_average(low_images)
	plt.imshow(low_mean, cmap=cm.Greys_r)
	plt.show()

	# Output average high scoring face
	print("High Scoring Face")
	high_mean = array_average(high_images)
	plt.imshow(high_mean, cmap=cm.Greys_r)
	plt.show()

	# Difference between low scoring mean and high scoring mean
	print("High_Mean minus Mean")
	plt.imshow(low_mean - high_mean, cmap=cm.Greys_r)
	plt.show()
# end image_averages

def array_average( images ):
	mean = (images[0])
	for i in xrange(images.shape[0]-1):
		mean = np.add(mean, images[i+1])
	mean = np.divide(mean,images.shape[0])
	return mean
# end array_average


def mean_difference( images, scores ):
	mean = np.mean(scores)
	std = np.std(scores)

	low_images = images[ scores < mean - std]
	high_images = images[ scores > mean + std]
	
	low_mean = array_average(low_images)
	high_mean = array_average(high_images)

	return high_mean - low_mean
# end mean_difference


def make_pretty( image, images, scores):
	mean_diff = mean_difference(images, scores)
	return rgb2gray(image) + 2*mean_diff;

def altered_tour(images, scores):
	# Basic tour of images with user attractiveness ratings, with altered images
	for i in xrange(images.shape[0]):
		print scores[i]
		image = images[i]
		plt.imshow(rgb2gray(image), cmap=cm.Greys_r)
		plt.show()
		plt.imshow(make_pretty(image, images, scores), cmap=cm.Greys_r)
		plt.show()


def pca_plot( images, scores):
	# Plot pca dimension comparisons
	mean = np.mean(scores)
	std = np.std(scores)

	# For plotting, k = 2
	k = 2

	pca = PCA(n_components=k)

	reduced_imgs = pca.fit_transform(images)

	# Find coordinates for low images
	low_flats = reduced_imgs[ scores < mean - std ]
	low_x, low_y = zip(*low_flats)

	# Find coordinates for high images
	high_flats = reduced_imgs[ scores > mean + std ]
	high_x, high_y = zip(*high_flats)

	# 0.91 variance maintained with 100 components
	plt.plot(low_x, low_y, 'g^', high_x, high_y, 'bs')
	plt.show()

def pca_tour( X, d=5 ):
	print "Examining top %s Eigenfaces" %d
	n, p = X.shape
	X = np.nan_to_num(X)

	X = X - np.tile(np.mean(X,0), (n,1))

	sData = sparse.csr_matrix(X)
	sData.astype('double')
	ud, sd, vdt = svds(sData, d)

	for i in range(vdt.shape[0]):
		image = np.reshape(vdt[i], (image_width, image_height))
		plt.imshow(image, cmap=cm.Greys_r)
		plt.show()



def add_bias(Z):
	(N, d) = Z.shape
	bias_col = np.ones((N,1))

	Z_bias = np.concatenate((Z, bias_col), axis=1)
	return Z_bias
# end add_bias


#image_averages( fem_images, fem_scores)
#altered_tour(fem_images, fem_scores)
[male_images, male_scores, fem_images, fem_scores] = load_data()
#image_tour(fem_images, fem_scores, 5)
pca_plot( male_images, male_scores )

#pca_tour( fem_images, 10)

	