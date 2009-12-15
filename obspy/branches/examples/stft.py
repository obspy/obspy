# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

from obspy.core import read
from obspy.imaging.spectrogram import spectroGram
import matplotlib.pyplot as plt
import numpy as np

# read GSE2 file, only once trace in stream
#tr = read("data/GR.FUR..LHZ.D.2004.361")[0]
#tr = read("data/BW.RNON..EHZ.D.2008.108")[0]
tr = read("data/BW.RNON..EHZ.D.2008.107")[0]
npts, df = tr.stats.npts, tr.stats.sampling_rate

# plot seismogram
plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(0.0,npts)/df, tr.data)

# plot spectrogram
plt.subplot(212)
spectroGram(tr.data, df, log=True)
plt.show()

