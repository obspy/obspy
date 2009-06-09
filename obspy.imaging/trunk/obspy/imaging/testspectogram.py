#!/scratch/seisoft/Python-2.5.2/bin/python
# -*- coding: utf-8 -*-

from obspy.gse2 import tests as gse2tests
import obspy, inspect, os
from pylab import *
import spectogram

path = os.path.dirname(inspect.getsourcefile(gse2tests))
file = os.path.join(path, 'data', 'loc_RJOB20050831023349.z')

g = obspy.Trace()
g.read(file,format='GSE2')

spectogram.spec(inp=g.data,sample_rate=200.0,samp_length=len(g.data),log=True)
spectogram.spectoGram(g.data,samp_rate=200.0,log=True,name="spec2")

#clf()
#plot(g.data)
#show()
