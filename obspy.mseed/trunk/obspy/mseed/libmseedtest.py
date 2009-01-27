#Some tests for the libmseed wrapper class
from libmseedclass import *

import sys
      
try:
  file = sys.argv[1]
except:
  file = "test.mseed"

outfile = "out.mseed"
file2 = "test2.mseed"


mseed=libmseed("test.mseed")
#mseed.cut(file)
#print mseed.samprate
mseed.msr_print(file)
