from numpy import *
from ext_arpicker import *
a=fromfile("loc_RJOB20050801145719850.z",sep="\n",dtype=float32)
b=fromfile("loc_RJOB20050801145719850.n",sep="\n",dtype=float32)
c=fromfile("loc_RJOB20050801145719850.e",sep="\n",dtype=float32)
print arPick(a,b,c,200,1.0,20.0,1.0,0.05,4,1,2,8,0.1,0.1)
