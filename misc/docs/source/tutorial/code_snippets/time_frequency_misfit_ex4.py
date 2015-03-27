from __future__ import print_function

import numpy as np
from obspy.signal.tf_misfit import plot_tf_misfits


# general constants
tmax = 6.
dt = 0.01
npts = int(tmax / dt + 1)
t = np.linspace(0., tmax, npts)

fmin = .5
fmax = 10
nf = 100

# constants for the signal
A1 = 4.
t1 = 2.
f1 = 2.
phi1 = 0.

# amplitude error
amp_fac = 1.1

# generate the signal
H1 = (np.sign(t - t1) + 1) / 2
st1 = A1 * (t - t1) * np.exp(-2 * (t - t1))
st1 *= np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1

# reference signal
st2_1 = st1.copy()
st2_2 = st1.copy() * 5.
st2 = np.c_[st2_1, st2_2].T
print(st2.shape)

# signal with amplitude error
st1a = st2 * amp_fac

plot_tf_misfits(st1a, st2, dt=dt, fmin=fmin, fmax=fmax)
