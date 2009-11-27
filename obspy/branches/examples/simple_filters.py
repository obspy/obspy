# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from filter_bc import filter_bc
import matplotlib
matplotlib.rc('figure.subplot', hspace=.35, wspace=.35) #adjust subplot layout
matplotlib.rc('font', size=8) # adjust font size of plot
import matplotlib.pyplot as plt
import numpy as np


# Number of points (here must be even for fft) and sampling_rate
npts=1026
dt=0.05

# Determining nyquist frequency and asking for cut off frequency f0
fmax=1.0/(2*dt) # nyquist
f0 = float(raw_input('Give cut-off below Nyquist: fmax = %4.1f Hz ' % fmax))
print 'f0 =', f0

# Uncomment from random points
#y = random.rand(npts) - .5 # uniform random numbers, zero mean

# Spike at npts/2
y = np.zeros(npts,dtype='float')
y[npts/2] = 1

# Filter with filter_bc
freq, H, y_filt = filter_bc(y,dt,f0)

# Just for the plot, frequency domain representaion of y
y_f = np.fft.rfft(y)

# Remove offset
y_f = y_f[1:]
freq = freq[1:]
H = H[1:]

# For convenience
time = np.arange(0,npts)*dt



#
# Plot the whole filtering process
#
plt.close('all')

plt.subplot(231)
plt.plot(time, y)
plt.title('Original data')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(232)
plt.plot(freq, abs(y_f))
plt.title('Amplitude spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Arbitrary Amplitude')

plt.subplot(233), 
plt.plot(freq, np.angle(y_f))
plt.title('Phase spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase shift [rad]')

plt.subplot(234)
plt.plot(freq, H)
plt.title('The filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Filter amplitude')

plt.subplot(235)
plt.plot(freq, abs(y_f*H))
plt.title('The filtered spectrum')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')

plt.subplot(236)
plt.plot(time, y_filt)
plt.title('The filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.show()
