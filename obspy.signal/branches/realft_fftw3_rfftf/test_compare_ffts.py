#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: test_compare_ffts.py
#  Purpose: Show and test how to convert between different FFTs
#   Author: Moritz Beyreuther
#    Email: beyreuth@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Moritz Beyreuther
#---------------------------------------------------------------------
"""
Compare different FFTs.

Here is the documentation of the numpy fft NOT rfftf:
 * A[0] contains the zero-frequency term (the mean of the signal), which is
   always purely real for real inputs. Then
 * A[1:n/2] contains the positive-frequency terms 
 * A[n/2+1:] contains the negative-frequency terms, in order of decreasingly
   negative frequency. 
 * A[n/2] represents both positive and negative Nyquist frequency for an
   even number of input points, and is also purely real for real input
"""


import numpy as np
import ctypes as C
import time

# initialize library
clib = C.CDLL('./realft.so')

#
# Load in the data
#
data = np.loadtxt('tests/data/GRF_031102_0225_mod.GSE', dtype='float32').ravel()
#data -= data.mean()
data = np.concatenate((data,)*1000)
serieslth = clib.ff_next2pow(len(data))

#
# Numerical Recipies FFT
#
# add extra entry as numerical recipies realft starts with index 1
spec1 = np.zeros(serieslth + 1, dtype='float32')
spec1[1:len(data) + 1] = data # copy the data, including correct type
clib.nr_realft(spec1, serieslth / 2, 1)
spec1 = spec1[1:]

#
# Numpy FFT (fftpack)
#
clock = time.clock()
buf = np.fft.rfft(data, n=serieslth) # spec gets dtype complex128
print "Time NumPy FFT", time.clock() - clock
buf = np.require(buf, 'complex64')
spec2 = np.zeros(serieslth, dtype='float32')
spec2[2::2] = +buf.real[1:-1]
spec2[3::2] = -buf.imag[1:-1]
spec2[0] = buf.real[0]  # mean of signal
spec2[1] = buf.real[-1] # nyquist frequency

#
# FFTW3, call the floating part of libfftw3.so.3 
# i.e. /usr/lib/libfftw3f.so
#
if sys.platform == 'win32':
    lib_name = 'lib/libfftw3f-3.win32.dll'
else:
    lib_name = '/usr/lib/libfftw3f.so'

fftw3 = C.CDLL(lib_name)
buf = np.zeros(serieslth + 2, dtype='float32')
buf[0:len(data)] = data
clock = time.clock()
plan = fftw3.fftwf_plan_dft_r2c_1d(C.c_int(serieslth), buf.ctypes.data,
                                   buf.ctypes.data, 64)
fftw3.fftwf_execute(C.c_int(plan))
print "Time FFTW3", time.clock() - clock
spec3 = np.zeros(serieslth, dtype='float32')
spec3[2::2] = +buf[2:-2:2]
spec3[3::2] = -buf[3:-2:2]
spec3[0] = buf[0]  #mean of signal
spec3[1] = buf[-2] #nyquist frequency

#
# normalize for comparison
#
_max = float(spec1.max())
spec1 /= _max
spec2 /= _max
spec3 /= _max

#
# now compare the results
#
np.testing.assert_array_almost_equal(spec2, spec1, decimal=7)
np.testing.assert_array_almost_equal(spec3, spec1, decimal=7)
