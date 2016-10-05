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
print np.argsort(b, axis=-1, kind='quicksort', order=None)
print np.argsort(b[0, :], axis=0, kind='quicksort', order=None) < 2
# Numpy also provides many functions to create arrays:
np.argmax(a)

a = np.zeros((2,2))
print a

b = np.zeros((2,2))
print b
c = np.full((2,2), 7)
print c

d = np.eye(2)
print d

e = np.random.random((2,3)) # Create an array filled with random valuesR
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
a = np.array([[1,2], [3,4], [5,6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print bool_idx

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print a[bool_idx]

# We can do all of the above in a single concise statement:
print a[a > 2]

# Datatypes
# Every numpy array is a grid of elements of the same type.
# Numpy provides a large set of numeric datatypes that you can
# use to construct arrays. Numpy tries to guess a datatype when you
# create an array, but functions that construct arrays usually
# also include an optional argument to explicitly specify the datatype.
# Here is an example:

x = np.array([1, 2])
print x.dtype

x = np.array([1.0, 2.0])
print x.dtype

x = np.array([1, 2], dtype = np.int64)
print x.dtype

# Array math
# Basic mathematical functions operate elementwise on arrays, and are
# available both as operator overloads and as functions is the numpy
# module:

x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[5,6], [7,8]], dtype=np.float64)

# Elementwise sum;
print x + y
print np.add(x, y)

# Elementwise difference;
print x - y
print np.subtract(x, y)

# Elementwise product;
print x * y
print np.multiply(x, y)

# Elementwise division;
print x / y
print np.divide(x, y)

# Elementwise square root;
print np.sqrt(x)

# Note that unlike MATLAB, * is elementwise multiplication, not matrix multiplication
# We instead use the dot function to compute inner products of vectors, to multiply
# a vector by a matrix, and to multiply matrices. dot is available both as a funtion
# in th numpy module and as an instance method of array objects:

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors
print v.dot(w)
print np.dot(v, w)

# Matrix / vector product;
print x.dot(v)
print np.dot(x, v)
print np.dot(v, x)

# Matrix / matrix product
print x.dot(y)
print np.dot(x, y)

# Numpy provides many useful functions for performing computations on arrays;
# one of the most useful is sum:

x = np.array([[1,2], [3,4]])

print np.sum(x)
print np.sum(x, axis=0)
print np.sum(x, axis=1)

# Apart from computing mathematical functions using arrays, we frequently
# need to reshape or otherwise manipulate data in arrays.
# The simplest example of this type of operation is transposing a matrix;
# to transpose a matrix, simply use the T attribute of an object:

x = np.array([[1,2], [3,4]])
print x
print x.T

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print v
print v.T

# Broadcasting
# Broadcasting is a powerful mechanism that allows numpy to work with
# arrays of different shapes when performing arithmetic operations.
# Frequently we have a smaller array and a larger array, and we
# want to use the smaller array multiple times to perform some
# operation on the larger array.

# For example, suppose that we want to add a constant vector to
# each row of a matrix. We could do it like this:

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
y = np.empty_like(x) # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print y

# This works; however when the matrix is very large, computing
# an explicit loop in Python could be slow. Note that adding the
# vector v to each row of the matrix x is equivalent to forming a matrix
# vv by stacking multiple copies of v vertically, then performing elementwise
# summation of x and vv. We could implement this approach like this:

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))
print vv

y = x + vv
print y

# Numpy broadcasting allows us to perform this computation
# without actually creating multiple copies of v.
# Consider this version, using broadcasting:

# We will add the vector v to each row of the matrix x,
# storing the result in matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1,0,1])
y = x + v
print y

# This line y = x + v works even though x has shape (4, 3) and v shape (3, ) due
# to broadcasting: this line works as if v actually had shape (4, 3), where each row
# was a copy of v, and the sum was performed elementwise.

# Broadcasting two arrays together follows these rules:
#   1. If the arrays do not have the same rank, prepend the shape of the lower rank
#       with 1s until both shapes have the same length.
#   2. The two arrays are said to be compatible ina dimension if they have the same size
#       in the dimension, or if one of the arrays has size 1 in the dimension.
#   3. The arrays can be broadcast together if they are compatible in all dimensions.
#   4. After broadcasting, each array behaves as if it had shape equal to the
#       elementwise maximum of shapes of the two input arrays.
#   5. In any dimension where one array had size 1 and the other array had size greater
#       than 1, the first array behaves as if it were copied along that dimension

# Here are some applications of broadcasting:

# Compute outer product of vectors
v = np.array([1,2,3])
w = np.array([4,5])
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[4 5]
#  [8 10]
#  [12 15]
print np.reshape(v, (3, 1)) * w

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[5 6 7]
#  [9 10 11]]
print (x.T + w).T
# Another solution is to reshape w to be a row vector of shape (2,1);
# we can then broadcast it directly against x to produce the same output.
print x + np.reshape(w, (2, 1))
print x +w.reshape(2, 1)

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[2 4 6]
#  [8 10 12]]
print x * 2

# Broadcasting typically makes your code more concise and faster,
# so you should strive to use it where possible.