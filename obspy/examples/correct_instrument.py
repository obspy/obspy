# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import read
from obspy.signal import seisSim, pazToFreqResp
from obspy.signal.seismometer import PAZ_WOOD_ANDERSON
import numpy as np
import matplotlib.pyplot as plt

# Define poles zeros and gain
le3d = {
    'poles': [-4.21000 + 4.66000j,
              - 4.21000 - 4.66000j,
              - 2.105000 + 0.00000j],
    'zeros': [0.0 + 0.0j] * 3, # add or remove zeros here
    'gain' : 0.4
}

# Read in the data
tr = read("loc_RJOB20050831023349.z")[0]

# Do the instrument correction
data_corr = seisSim(tr.data, tr.stats.sampling_rate, le3d, 
                    inst_sim=PAZ_WOOD_ANDERSON, water_level=60.0)

# Just for visualization, calculate transferfuction in frequency domain
trans, freq =  pazToFreqResp(le3d['poles'], le3d['zeros'], le3d['gain'],
                             1./tr.stats.sampling_rate, 2**12, freq=True)

#
# The plotting part
#
time = np.arange(0,tr.stats.npts)/tr.stats.sampling_rate
plt.figure()
plt.subplot(211)
plt.plot(time, tr.data, label="Original Data")
plt.legend()
plt.subplot(212)
plt.plot(time, data_corr, label="Wood Anderson Simulated Data")
plt.legend()
plt.xlabel("Time [s]")
plt.suptitle("Original and Corrected Data")

plt.figure()
plt.subplot(211)
plt.loglog(freq,abs(trans))
plt.title("Amplitude")
plt.subplot(212)
plt.title("Phase")
plt.semilogx(freq,np.angle(trans)/np.pi*360)
plt.xlabel("Frequency [Hz]")
plt.show()
