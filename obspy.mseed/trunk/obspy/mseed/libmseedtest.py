#Some tests for the libmseed wrapper class
from obspy.mseed.libmseed import *

import sys

try:
    file = sys.argv[1]
except:
    file = "test.mseed"
outfile='out.mseed'
mseed=libmseed()

header, data, numtraces=mseed.read_ms(file)
mseed.cut_ms(data, header, 0, 5000)
import pdb;pdb.set_trace()