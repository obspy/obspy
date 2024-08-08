"""
USAGE: export_seismograms_to_ascii.py in_file out_file calibration
"""
from __future__ import print_function

import sys

import numpy as np
import obspy


try:
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    calibration = float(sys.argv[3])
except Exception:
    print(__doc__)
    raise

st = obspy.read(in_file)
for i, tr in enumerate(st):
    f = open("%s_%d" % (out_file, i), "w")
    f.write("# STATION %s\n" % (tr.stats.station))
    f.write("# CHANNEL %s\n" % (tr.stats.channel))
    f.write("# START_TIME %s\n" % (str(tr.stats.starttime)))
    f.write("# SAMP_FREQ %f\n" % (tr.stats.sampling_rate))
    f.write("# NDAT %d\n" % (tr.stats.npts))
    np.savetxt(f, tr.data * calibration, fmt="%f")
    f.close()
