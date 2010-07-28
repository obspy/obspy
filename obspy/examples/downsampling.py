# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

import matplotlib
matplotlib.rc('font', size=8) # adjust font size of plot
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal import lowpass, lowpassZPHSH, cosTaper
from filter_bc import filter_bc
PI = np.pi

# Number of points
npts=64
dt=0.05

# Determine Nyquist and set generator frequency
fmax=1.0/(2*dt) # nyquist
fg1 = 8.0 # generator frequency
fg2 = 4.0 # generator frequency

# Generate f0 Hz sine wave
pg1 = 1.0/fg1 #in seconds
pg2 = 1.0/fg2 #in seconds
time = np.linspace(0,npts*dt,npts) # in seconds
y  = np.sin(2*PI/pg1 * time + PI/5)
y += np.sin(2*PI/pg2 * time + PI/5)

# Downsample by taking every second element
y_2 = y[::2]

# Hanning window
win_han = np.hanning(npts)
y_han = y * win_han

# Just for plotting the frequency domain
freq = np.linspace(0, fmax, npts//2+1) #// == integer devision
y_f = np.fft.rfft(y)
y_f2 = np.fft.rfft(y_2)

# Setting frequencies above nyquist to zero
nyq_new = fmax/2
y_fnew = np.where( freq > nyq_new, 0, y_f) # setting everything above nyquist to zero
y_new = np.fft.irfft(y_fnew)

# Remove offset
freq = freq[1:]
y_f = y_f[1:]
y_f2 = y_f2[1:]
y_fnew = y_fnew[1:]


#
# Plot the whole filtering process
#
plt.figure(figsize=(14,10))
plt.subplot(121)
plt.plot(time,y,'k',label="Original data",lw=2)
plt.plot(time[::2],y[::2],'r--',label="Every second element",lw=2)
plt.plot(time[::2],y_new[::2],'g',label="Removed frequencies above nyquist",lw=2)
plt.legend()
plt.title('Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(freq, abs(y_f), 'k', label="Frequency no Taper", lw=3)
plt.plot(freq, abs(y_fnew), 'g--', label="Zeroing Frequencies", lw=3)
plt.plot(freq[:len(y_f2)], abs(y_f2), 'r--', label="Just taking half", lw=3)
plt.legend()
plt.title('Frequency Domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Arbitrary Amplitude')

plt.show()
