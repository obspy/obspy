import numpy as np

from obspy.signal.tf_misfit import plot_tfr


# general constants
tmax = 6.
dt = 0.01
npts = int(tmax / dt + 1)
t = np.linspace(0., tmax, npts)

fmin = .5
fmax = 10

# constants for the signal
A1 = 4.
t1 = 2.
f1 = 2.
phi1 = 0.

# generate the signal
H1 = (np.sign(t - t1) + 1) / 2
st1 = A1 * (t - t1) * np.exp(-2 * (t - t1))
st1 *= np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1

plot_tfr(st1, dt=dt, fmin=fmin, fmax=fmax)
