#!/usr/bin/python

# BASIC NUMERIC MODULE
import numpy as np


# ALLOCATING NUMPY ARRAYS
N = 5
x1 = np.zeros(N)
x2 = np.ones(N)
x3 = np.random.randn(N)
x4 = np.random.rand(N)

# DOT PRODUCT
# exception for one dimenstional vectors
dot = np.dot(x3,x3) # inner product
outer = np.outer(x3,x3) #outer product
print "Dot", dot, "Outer", outer.shape

# general for two dimensional vectors
print "x3 1-DIM", x3.shape
x3 = x3.reshape(-1,1) # reshape into coloumn vector
print "x3 2-DIM", x3.shape
dot2 = np.dot(x3.T, x3)  #.T == transpose
outer2 = np.dot(x3, x3.T) #outer product
print "Dot2", dot2, "Outer2", outer2.shape


# CAUTION, TYPE OF ARRAYS IS IMPORTANT
y1 = np.ones(N, dtype='int32')
y3 = y1.copy()
y2 = np.ones(N, dtype='float64')
# with inplace multiplication, the type stays the same!
y1 *= 0.5
y2 *= 0.5
print "y1", y1, "y2", y2
# with normal multiplication, the type changes
y3 = y3 * 0.5
print "y3", y3
