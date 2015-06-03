import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from obspy.signal.tf_misfit import plot_tf_gofs


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

# amplitude and phase error
phase_shift = 0.8
amp_fac = 3.

# generate the signal
H1 = (np.sign(t - t1) + 1) / 2
st1 = A1 * (t - t1) * np.exp(-2 * (t - t1))
st1 *= np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1

# reference signal
st2 = st1.copy()

# generate analytical signal (hilbert transform) and add phase shift
st1p = hilbert(st1)
st1p = np.real(np.abs(st1p) *
               np.exp((np.angle(st1p) + phase_shift * np.pi) * 1j))

# signal with amplitude error
st1a = st1 * amp_fac

plot_tf_gofs(st1a, st2, dt=dt, fmin=fmin, fmax=fmax, show=False)
plot_tf_gofs(st1p, st2, dt=dt, fmin=fmin, fmax=fmax, show=False)

plt.show()
