#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare results from removing instrument response using
evalresp in SAC and ObsPy. Visual inspection shows that the traces are
pretty much identical but differences remain (rms ~ 0.042). Haven't
found the cause for those, yet.
"""

from obspy import Trace, UTCDateTime, read
from obspy.core.util.base import NamedTemporaryFile
from obspy.sac import attach_paz
from obspy.signal.invsim import seisSim, estimateMagnitude, evalresp
from obspy.signal.invsim import cosTaper
import gzip
import numpy as np
import os
import unittest

path = "/home/moritz/code/obspy/obspy/signal/tests/data"

evalrespf = os.path.join(path, 'CRLZ.HHZ.10.NZ.SAC_resp')
rawf = os.path.join(path, 'CRLZ.HHZ.10.NZ.SAC')
respf = os.path.join(path, 'RESP.NZ.CRLZ.10.HHZ')
fl1 = 0.00588
fl2 = 0.00625
fl3 = 30.
fl4 = 35.

#Set the following if-clause to True to run
#the sac-commands that created the testing file
if False:
    import subprocess as sp
    p = sp.Popen('sac', stdin=sp.PIPE)
    cd1 = p.stdin
    print >>cd1, "r %s" % rawf
    print >>cd1, "rmean"
    print >>cd1, "taper type cosine width 0.05"
    print >>cd1, "transfer from evalresp fname %s to vel freqlimits\
    %f %f %f %f" % (respf, fl1, fl2, fl3, fl4)
    print >>cd1, "w over %s" % evalrespf
    print >>cd1, "quit"
    cd1.close()
    p.wait()

tr = read(rawf)[0]
trtest = read(evalrespf)[0]
date = UTCDateTime(2003, 11, 1, 0, 0, 0)
seedresp = {'filename': respf, 'date': date, 'units': 'VEL'}
tr.data = seisSim(tr.data, tr.stats.sampling_rate, paz_remove=None,
                  pre_filt=(fl1, fl2, fl3, fl4),
                  seedresp=seedresp, taper_fraction=0.1,
                  pitsasim=False, sacsim=True)
tr.data *= 1e9
rms = np.sqrt(np.sum((tr.data - trtest.data) ** 2) /
              np.sum(trtest.data ** 2))
print rms
