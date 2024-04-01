#!/usr/bin/python3

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik 2017-2020

import numpy as np
from util import *

# Instructions: you don`t need to code anything, just read the code and see what it does.
# Turn on/off the sections by changing "if False" conditions.


##################################################
if True:
    print('=== 1) Creating vectors and matrices ===')

    # If you would want to create a vector in numpy you can do so as follows:
    a = np.array([2, 4, 5, 6])
    print(a, '\n')

    # To create a matrix you can use same syntax as you would for 2D arrays in Python
    A = np.array([[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12]])
    print(A, '\n')

    # All objects like these (vector, matrix...) have shape attribute which describes, as the name
    # suggests, their shape. Or in other words, what do the dimensions of a particular object look
    # like. This attribute can be accessed by writing .shape after the name of the object as
    # follows:
    print('Shape of vector a is {}'.format(a.shape))
    print('Shape of matrix A is {}\n'.format(A.shape))

    # Shape is always s tuple. Note that shape of numpy vector is "(4,)". It is not "(4,1)" -
    # column vector, nor "(1,4)" - row vector! This may simplify life in some situations, but
    # cause problems in others. If you wish, you can change any numpy vector (of shape (n,) ) to
    # explicitly column vector (i.e. matrix of shape (n,1) ) by calling function vector(x), that
    # we prepared for you.
    column_a = vector(a)
    print(column_a)
    print('Shape of vector column_a is {}\n\n\n'.format(column_a.shape))



##################################################
if False:
    print('=== 2) Obtaining values from vectors/matrices ===')

    # You can get any value from a vector simply by indexing it (indexes are zero-based):
    a = np.array([2, 4, 5, 6])
    print('a = {}'.format(a))
    print(a[0])
    print(a[3], '\n')

    # To get an element of a matrix, you specify two indexes - row and column:
    A = np.arange(100).reshape((10,10)) # some matrix
    print('A =\n{}'.format(A))
    print(A[1,3], '\n')

    # You can select whole row or column of an array, just use ':' instead of index (it means
    # 'all'). Note that the result is a numpy vector, and its shape is (n,).
    A_col0 = A[:,0]     # "give me all rows, 0-th column"
    print('0-th column of A is:\n{}\nand its shape is {}\n'.format(A_col0, A_col0.shape))

    A_row3 = A[3,:]     # "give me 3rd row, all columns"
    print('3rd row of A is:\n{}\nand its shape is {}\n'.format(A_row3, A_row3.shape))

    # Finally, you can select a whole sub-matrix by specifying which columns and rows you want:
    A_sub = A[0:3, 3:8] # "give me rows 0 to 2, columns 3 to 7"
    print(A_sub, '\n')

    A_sub = A[:-1, :] # "give me all columns except last one, all rows"
    print(A_sub, '\n\n\n')



##################################################
if False:
    print('=== 3) Reshape and transpose matrices ===')

    # We already told that all vectors/matrices have shape. You can change the shape (with
    # preserving the elements order) by calling reshape:
    A = np.array([[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12]])
    print('A =\n{}\n'.format(A))

    B = A.reshape((2,6))    # "change to 2 rows, 6 columns"
    print(B, '\n')

    C = A.reshape((4, -1))  # "change to 4 rows, and as many columns as needed"
    print(C, '\n')

    # As you can see, reshaping preserves the order of the elements in matrix.
    # Transposing a matrix is a different operation - it 'swaps' the rows and columns:
    A_trans = np.transpose(A)   # \
    A_trans = A.transpose()     # - all three do the same
    A_trans = A.T               # /
    print('A transposed =\n{}\n'.format(A_trans))

    # Beware that transponing a numpy vector (with shape (n,) ) does not do anything!
    a = np.array([2, 4, 5, 6])
    print('a = {}'.format(a))
    print('a transposed = {}\n'.format(a.T))

    # If you want to convert to column vector or row vector, you must reshape it to (n,1) or (1,n),
    # or just use our prepared function vector:
    a_col = a.reshape((-1,1))
    print(a_col, '\n')
    a_col = vector(a)
    print(a_col, '\n\n\n')



##################################################
if False:
    print('=== 4) Generating useful matrices ===')

    # In scientific computation practice, it is often needed to create a matrix with certain
    # values, such as an array with all elements equal to zero or a matrix initialized with values
    # sampled from the normal distribution.
    # Numpy provides functions that allow us to quickly create objects like this:

    # Create a vector with elements [0, 1, 2, 3, 4]
    a = np.arange(5)
    print("a = {}\n".format(a))

    # Create a matrix filled with zeros of shape (10, 5)
    A = np.zeros((10, 5))
    print("A =")
    print(A, '\n')

    # Create a matrix filled with ones of shape (5, 10)
    B = np.ones((5, 10))
    print("B =")
    print(B, '\n')

    # Create a square 'eye' matrix (identity matrix) of size 5
    I = np.eye(5)
    print("I =")
    print(I, '\n')

    # Create a matrix of shape (5, 5) with uniform random values from interval <0,1>
    X = np.random.rand(5, 5)
    print("X with shape {}:".format(X.shape))
    print(X, '\n')

    # Create a matrix of shape (3, 5) with values sampled from normal distribution (mean=0, std=1)
    Y = np.random.randn(3, 5)
    print("Y with shape {}:".format(Y.shape))
    print(Y, '\n\n\n')


##################################################
if False:
    print('=== 5) Arithmetics ===')

    # Once you have some vectors and matrices initialized, there are a ton of arithmetic
    # operations you can do with them.
    a = np.arange(5)
    print('a =\t{}'.format(a))
    b = np.array([2, 2, 3, 2, 3])
    print("b =\t{}".format(b))

    # add a constant value to each element of a vector
    print("a + 5 =\t{}".format( a + 5 ))

    # multiply each element of a vector by a constant value
    print("a * 5 =\t{}".format( a * 5 ))

    # square the vector a
    print("a^2 =\t{}".format( a ** 2 ))

    # element-wise multiply vector a by vector b, result is vector!
    print("a * b =\t{}".format( a * b ))

    # dot product of vector a by vector b, result is scalar!
    a_dot_b = np.dot(a, b)  # \
    a_dot_b = a.dot(b)      # - all three do the same
    a_dot_b = a @ b         # /
    print("a @ b =\t{}".format( a_dot_b ))

    # outer product of vector a and vector b, result is matrix!
    print("a x b =\n{}\n".format( np.outer(a, b) ))


    # The same operations can be done with matrices too:
    A = np.array([[1, 2, 3],
                  [1, 2, 3]])
    B = np.array([[2, 2, 2],
                  [3, 3, 3]])
    print('A =\n{}'.format(A))
    print('B =\n{}'.format(B))
    print("A + 5 =\n{}".format( A + 5 ))
    print("A * 5 =\n{}".format( A * 5 ))
    print("A^2 =\n{}\n".format( A ** 2 ))

    # element-wise multiplication
    print('A * B =\n{}\n'.format(A * B))

    # dot product = "normal" matrix multiplication can only be done if shape of A and B are aligned
    A = np.random.rand(2, 3)
    print('A with shape {}\n{}\n'.format(A.shape, A))
    B = np.random.randn(3, 5)
    print('B with shape {}\n{}\n'.format(B.shape, B))
    AB = A @ B
    print('A @ B =\n{}\nits shape is {}\n'.format( AB, AB.shape))


    # Use with caution: "broadcasting" lets you do e.g. matrix + vector, if shapes are aligned:
    a = np.arange(5)
    A = np.ones((3, 5))
    print('A =\n{}\na = {}'.format(A, a))
    print('Broadcasting: A + a =\n{}\n\n\n'.format( A + a ))


##################################################
if False:
    print('=== 6) Matrix functions ===')

    # Numpy provides you with a variety of functions to work with matrices, for example sum,
    # mean (average), std, exp, sin, cos, ...
    # Almost everything you can think of, there is a function for it in numpy. Just ask us if you
    # ever need to use a specific operation with a matrix, we`ll tell you what numpy function to
    # call.
    # These functions are matrix-optimized, and they are typically much faster then your own
    # implementation (e.g. using for-cycles)!

    # We`ll take a closer look at sum function:
    a = np.arange(5)
    print('a =\t{}'.format(a))
    print('sum of a = {}\n'.format( np.sum(a) ))

    # With matrices, you can specify which axis (rows/columns) you want to calculate the sum:
    A = np.array([[1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]])
    print('A =\n{}'.format(A))
    print('np.sum(A, axis=0) = {}'.format( np.sum(A, axis=0) ))
    print('np.sum(A, axis=1) = {}'.format( np.sum(A, axis=1) ))
    print('np.sum(A) = {}'.format( np.sum(A) ))


    # Another useful function is np.linalg.norm(x), which tells you the norm (i.e. length) of a
    # vector:
    a = np.array([3,4])
    a_norm = np.linalg.norm(a)
    print('\n|a| = {}'.format(a_norm))
