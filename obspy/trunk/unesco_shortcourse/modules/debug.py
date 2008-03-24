# ipython -pylab

from numpy import *

import ar_picker
import baer_picker
import trigger

a=fromfile("loc_RJOB20050801145719850.z",sep="\\n",dtype=float32)
b=fromfile("loc_RJOB20050801145719850.n",sep="\\n",dtype=float32)
c=fromfile("loc_RJOB20050801145719850.e",sep="\\n",dtype=float32)
