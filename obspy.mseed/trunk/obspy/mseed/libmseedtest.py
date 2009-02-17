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
import pdb;pdb.set_trace()
mseed.write_ms(header, data, outfile, numtraces)
