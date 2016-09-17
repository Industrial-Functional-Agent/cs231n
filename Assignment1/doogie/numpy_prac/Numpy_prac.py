import numpy as np

# Arrays
# A numpy array is a grid of values, all of the same type, and is indexed by
# a tuple of nonnegative integers. The number of dimensions is the rank
# of the array; the shape of an array is a tuple of integers giving the size of the
# array along each dimension.

a = np.array([1, 2, 3])  # Create a rank 1 array
print type(a)            # Prints "<type 'numpy.ndarray'>"
print a.shape            # Prints "(3,)"
print a[0], a[1], a[2]   # Prints "1 2 3"
a[0] = 5                 # Change an element of the array
print a                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print b.shape                     # Prints "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # Prints "1 2 4"
print b

# Numpy also provides many functions to create arrays:

a = np.zeros((2,2))
print a

b = np.zeros((2,2))
print b
c = np.full((2,2), 7)
print c

d = np.eye(2)
print d

e = np.random.random((2,2)) # Create an array filled with random valuesR
print e

# Array indexing
# Numpy offers several ways to index into arrays.
# Slicing: Similar to Python lists, numpy arrays can be sliced.
# Since arrays may be multidimensional, you must specify a slice
# for each dimension of the array:

# Create the following rank 2 array with shape (3, 4)
# [[ 1 2 3 4]
#  [ 5 6 7 8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]])

# Use slicing to pull out the subarray consisting of the first 2
# rows and columns 1 and 2; b is the following array of shape(2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print a[0, 1]
b[0, 0] = 77
print a[0, 1]

# You can also mix integer indexing with slice indexing. However, doing so will
# yield an array of lower rank than the original array. Note the this is quite different
# from the way that MATLAB handles array slicing:

# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]])

# Two ways of accessing the data in the middle row the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :] # Rank 1 view of the second row of a
row_r2 = a[1:2, :] # Rank 2 view of the second row of a
print row_r1, row_r1.shape # Prints "[5 6 7 8] (4,)"
print row_r2, row_r2.shape # Prints "[[5 6 7 8] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape # Prints "[2 6 10] (3,)"
print col_r2, col_r2.shape # Prints "[[ 2]
                           #          [ 6]
                           #          [ 10]] (3, 1)"

# Integer array indexing: When you index into numpy using slicing, the
# resulting array view will always be a subarray of the original array.
# In contrast, integer array indexing allows you to construct arbitrary arrays
# using the data from another array. Here is an example:
a = np.array([[1, 2], [3, 4], [5, 6]])
print a

# An example of integer array indexing.
# The returned array will have shape (3, ) and
print a[[0, 1, 2], [0, 1, 0]] # Prints "[1 4 5]" row 0, 1, 2 and column 0, 1, 0

# When using integer array indexing, you can reuse the same
# element from the source array:
print a[[0, 0], [1, 1]] # Prints "[2 2]" row 0, 0 and column 1, 1

# Equivalent to the previous integer array indexing example
print np.array([a[0, 1], a[0, 1]])

# One useful trick with integer array indexing is selecting or mutating one element
# from each row of a matrix:

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print a

b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print a[np.arange(4), b]

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print a

# Boolean array indexing: Boolean array indexing lets you pick out arbitrary
# element of an array. Frequently this type of indexing is used to select
# the elements of an array that satisfy some condition. Here is an example:
