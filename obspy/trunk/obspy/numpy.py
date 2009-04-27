# -*- coding: utf-8 -*-
"""
A simple wrapper around the Python array class to simulate NumPy arrays.

If NumPy is installed, NumPy.ndarray is used. Go for NumPy if you hit
performance issues.
"""

import sys, imp

# circumvent name clash
numpy = imp.load_module('numpy',*imp.find_module('numpy',sys.path[::-1])) 
#if False:
if numpy:
    class array(numpy.ndarray):
        def __new__(cls, obj, format='f'):
            # The __new__ method is read before the __inti__ method,
            # inheret from numpy array using the numpy.array fct
            # http://mail.scipy.org/pipermail/numpy-discussion/2006-February/006664.html
            return numpy.array(obj,dtype=format).view(cls)
else:
    # use barray as name, else we get a name clash
    from array import array as barray
    class array(barray):
        def __new__(cls,obj,format='l'):
            return barray.__new__(cls,format,obj)

