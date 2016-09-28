# Scipy
# Numpy provides a high-performance multidimensional array and
# basic tools to compute with and manipulate these arrays.
# Scipy builds on this, and provides a large number of functions
# that operate on numpy arrays and are useful for different types
# of scientific and engineering applications.

# Image operations
# Scipy provides some basic functions to work with images.
# For example, it has functions to read images from disk
# into numpy arrays, to write numpy arrays to disk as images,
# and to resize images. Here is a simple example that
# showcases these functions:

from scipy.misc import imread, imsave, imresize
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print img.dtype, img.shape # Prints "uint8 (400, 248, 3)"


# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)

# MATLAB files
# The functions scipy.io.loadmat and scipy.io.saveamat allow you
# to read and write MATLAB files.

# Distance between points
# Scipy defines some useful functions for computing distance
# between sets of points

# The function scipy.spatial.distance.pdist computes the distance
# between all pairs of points in a given set:

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[0.  1.414   2.236]
#  [1.414 0.    1.   ]
#  [2.236 1.    0.   ]]
d = squareform(pdist(x, 'euclidean'))
print d

# Matplotlib
# Matplotlib is plotting library. In this section give a brief
# introduction to the matplotlib.pyplot module, which provides
# a plotting system similar to that of MATLAB.

# Plotting
# The most important function in matplotlib is plot, which
# allows you to plot 2D data. Here is a simple example:

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.figure(1)
plt.plot(x, y)
# plt.show() # You must call plt.shoe() to make graphics appear.

# With just a little bit of extra work we can easily plot
# multiple lines at once, and add a title, legend, and axis labels:

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.figure(2)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosien'])
# plt.show()


# Subplots
# You can plot different things in the same figure using the subplot
# functions. Here is an example:

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.figure(3)
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as acitve, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure
# plt.show()

# Images
# You can use the imshow functions to show images.
# Here is an example:

img = imread('assets/cat.jpg')
img_tinted = img * [1.5, 1.5, 1.5]

# Show the original image
plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(img)
print img_tinted.dtype

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might five strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
