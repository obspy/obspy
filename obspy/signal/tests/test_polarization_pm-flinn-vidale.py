#!/usr/bin/env python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime, Trace, Stream
from obspy.signal import polarization

x = np.arange(0,2048/20.,1/20.)
x *= 2. * np.pi
y = np.cos(x)
trZ = Trace(data=y)
trZ.stats.sampling_rate = 20.
trZ.stats.starttime = UTCDateTime('2014-03-01T00:0')
trZ.stats.station = 'POLT'
trZ.stats.channel = 'HHZ'
trZ.stats.network = 'XX'

trN = trZ.copy()
trN.data *= 2.
trN.stats.channel = 'HHN'
trE = trZ.copy()
trE.stats.channel = 'HHE'


sz = Stream()
sz.append(trZ)
sz.append(trN)
sz.append(trE)
sz.sort(reverse=True)

t = sz[0].stats.starttime
e = sz[0].stats.endtime
fl=1.0
fh=5.00
ll = ['pm','flinn','vidale']
print ll
for method in ll:
    print method
    var_noise = 0.

    kwargs = dict(
            win_len=10.0, win_frac=0.1,
            # frequency properties
            frqlow=fl, frqhigh=fh, 
            # restrict output
            verbose=False, timestamp='mlabday',
            stime=t, etime=e,method=method,var_noise=var_noise)

    out = polarization.polarizationAnalysis(sz, **kwargs)

    npt = len(out[:,0])
    npt /= 2

    if(method == 'pm'):
        print 'PM azimuth: %f incidence: %f az_err: %f inc_err: %f'%(out[npt,1],out[npt,2],out[npt,3],out[npt,4])
    if method == 'flinn':
        print 'Flinn azimuth: %f incidence: %f rect: %f plan: %f'%(out[npt,1],out[npt,2],out[npt,3],out[npt,4])
    if method == 'vidale':
        print 'Vidale azimuth: %f incidence: %f rect: %f plan: %f elip: %f'%(out[npt,1],out[npt,2],out[npt,3],out[npt,4],out[npt,5])
