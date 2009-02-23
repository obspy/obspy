#Some tests for the libmseed wrapper class
from obspy.mseed.libmseed import *

import sys

try:
    file = sys.argv[1]
except:
    file = "BW.BGLD..EHE.D.2008.001"
outfile='out.mseed'
mseed=libmseed()
gapslist=mseed.findgapsandoverlaps(file)
import pdb;pdb.set_trace()