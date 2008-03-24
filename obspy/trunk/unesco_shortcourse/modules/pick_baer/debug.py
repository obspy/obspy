from numpy import *
from ext_pk_mbaer import *
a=fromfile("loc_RJOB20020325181100.ascii",sep="\n",dtype=float32)
print baerPick(a,200,20,60,7,12,100,100)
