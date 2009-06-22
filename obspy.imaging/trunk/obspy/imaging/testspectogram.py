#!/scratch/seisoft/Python-2.5.2/bin/python
# -*- coding: utf-8 -*-

from obspy.gse2 import tests as gse2tests
import obspy, inspect, os
from pylab import *
import spectogram

path = os.path.dirname(inspect.getsourcefile(gse2tests))
file = os.path.join(path, 'data', 'loc_RJOB20050831023349.z')

g = obspy.read(file,format='GSE2')

spectogram.spec(inp=g[0].data,sample_rate=200.0,samp_length=len(g[0].data),log=True)
spectogram.spectoGram(g[0].data,samp_rate=200.0,log=True,outfile='hallo.png')

#clf()
#plot(g.data)
#show()
