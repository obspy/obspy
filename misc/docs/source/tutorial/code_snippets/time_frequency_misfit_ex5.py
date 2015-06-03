import numpy as np
import matplotlib.pyplot as plt

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

# amplitude and phase error
amp_fac = 1.1

# generate the signal
H1 = (np.sign(t - t1) + 1) / 2
st1 = A1 * (t - t1) * np.exp(-2 * (t - t1))
st1 *= np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H1

ste = 0.001 * A1 * np.exp(- (10 * (t - 2. * t1)) ** 2)

# reference signal
st2 = st1.copy()

# signal with amplitude error + small additional pulse aftert 4 seconds
st1a = st1 * amp_fac + ste

plot_tf_misfits(st1a, st2, dt=dt, fmin=fmin, fmax=fmax, show=False)
plot_tf_misfits(st1a, st2, dt=dt, fmin=fmin, fmax=fmax, norm='local',
                clim=0.15, show=False)

plt.show()
