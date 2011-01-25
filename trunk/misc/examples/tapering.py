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
fg = 2.0 # Hz

# Generate f0 Hz sine wave
pg = 1.0/fg #in seconds
time = np.linspace(0,npts*dt,npts) # in seconds
y = np.sin(2*PI/pg * time + PI/5)

# Hanning window
win_han = np.hanning(npts)
y_han = y * win_han

# Filter the trace
f0 = 3.0 #cut off frequency
freq, H, y_1 = filter_bc(y,dt,3.0)
freq, H, y_2 = filter_bc(y_han,dt,3.0)

# Just for plotting the frequency domain
y_f = np.fft.rfft(y)
y_f2 = np.fft.rfft(y_han)

# Remove offset
freq = freq[1:]
y_f = y_f[1:]
y_f2 = y_f2[1:]

# Print rms (root-mean-square)
print "Orig data/filtered RMS", np.sqrt(np.mean((y - y_1)**2))
print "Tapered data/filtered RMS", np.sqrt(np.mean((y_han - y_2)**2))

#
# Plot the whole filtering process
#
plt.figure(figsize=(14,10))
plt.subplot(121)
plt.plot(time,y,label="Original data")
plt.plot(time,y_han,label="Tapered data (Hanning Window)")
plt.plot(time, y_1, label="Data low passed at 5Hz")
plt.plot(time, y_2, label="Tapered data low passed at 5Hz")
plt.legend()
plt.title('Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.bar(freq, abs(y_f), label="Filtered no taper", width=dt*6)
plt.bar(freq, abs(y_f2), label="Filtered with taper", width=dt*6,color='g')
plt.legend()
plt.title('Frequency Domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Arbitrary Amplitude')

plt.show()
