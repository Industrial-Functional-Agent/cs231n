# Python is a high-level, dynamically typed multiparadigm programing language.
# Python code is often said to be almost like pseudocode, since it allows you
# to express very powerful ideas in very few lines of code while being very
# readable. As an example, here is an implementation of the classic quicksort
# algorithm in Python:
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) / 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
print quicksort([3,6,8,10,1,2,1])

# Strings: Python has great support for strings:
hello = 'hello'
world = "world" # String literal can use single quotes and double quotes;
print hello
print len(hello)

hw = hello + ' ' + world # String concatenation
print hw
hw12 = '%s %s %d' % (hello, world, 12)
print hw12

# String objects have bunch of useful methods; for example:
s = "hello"
print s.capitalize()
print s.upper()
print s.rjust(7)
print s.center(7)
print s.replace('1', '(ell)')

print ' world '.strip()

# Containers
# Python includes several built=in container types: lists, dictionaries, sets, and tuples.

# List
xs = [3, 1, 2]
print xs, xs[2]
print xs[-1]
xs[2] = 'foo'
print xs
xs.append('bar')
print xs
x = xs.pop()
print x, xs

# Slicing: In addition to accessing list elements one at a time,
# Python provides concise syntax to access sublists; this is
# known as slicing
nums = range(5) # range is a built-in function that creates a list of integers
print nums
print nums[2:4] # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print nums[2:]  # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print nums[:2]
print nums[:]
print nums[:-1] # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]
print nums

# We will see slicing again in the context of numpy arrays.
# Loops: You can loop over the elements of a list like this:
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print animal

# If you want access to the index of each element within the body of a loop,
# use the built-in enumerate function:
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)

# List comprehensions: When programming, frequently we want to transform one
# type of data into another. As a simple example, consider the following code
# that computes square numbers:
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print squares

# You can make this code simpler using a list comprehension:
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print squares

# List comprehensions can also contain conditions:
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares

# Dictionaries
# A dictionary stores (key, value) pairs, similar to a Map in Java
# or an object in Javascript. You can use it like this:
d = {'cat': 'cute', 'dog': 'furry'}
print d['cat']
print 'cat' in d
d['fish'] = 'wet'
print d['fish']
print d.get('monkey', 'N/A') # Get an element with a default; prints "N/A"
print d.get('fish', 'N/A')  # Get an element with a default; prints "wet"
del d['fish']
print d.get('fish', 'N/A')

# Loops: It is easy to iterate over the keys in a dictionary
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print 'A %s has %d legs' % (animal, legs)

# If you want access to keys and their corresponding values, use the iteritems method:
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.iteritems():
    print 'A %s has %d legs' % (animal, legs)

# Dictionary comprehensions: These are similar to list comprehensions,
# but allow you to easily construct  dictionaries. For example:
nums = [0, 1, 2, 4, 5]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print even_num_to_square

# Sets
# A set in an unordered collection of distinct elements. As a simple example,
# consider the following:
animals = {'cat', 'dog'}
print 'cat' in animals
print 'fish' in animals
animals.add('fish')
print 'fish' in animals
print len(animals)
animals.add('cat')
print len(animals)
animals.remove('cat')
print len(animals)

# Loops: Iterating over a set has the same syntax as iterating over a list;
# however since sets are unordered, you cannot make assumptions about the order
# in which you visit the elements of the set:
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)

# Set comprehensions: Like lists and dictionaries, we can easily
# construct sets using set comprehensions:
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print nums

# Tuples
# A tuple is an (immutable) ordered list of values. A tuple is in many
# ways similar to a list; one of the most important differences is that
# tuples can be used as keys in dictionaries and as elements of sets,
# while lists cannot. Here is a trivial example:
d = {(x, x + 1): x for x in range(10)}
t = (5, 6)
print d
print type(t)
print d[t]
print d[(1, 2)]

# Functions
# Python functions are defined using the def keyword. For example:
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print sign(x)

# We will often define functions to take optional keyword arguments, like this:
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s!' % name.upper()
    else:
        print 'Hello, %s' % name

hello('Bob')
hello('Fred', loud=True)

# Classes
# The syntax for defining classes in Python is straightforward
class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name    # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')
g.greet()
g.greet(loud=True)