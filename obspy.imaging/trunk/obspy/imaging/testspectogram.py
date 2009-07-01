#!/scratch/seisoft/Python-2.5.2/bin/python
# -*- coding: utf-8 -*-

from obspy.gse2 import tests as gse2tests
import obspy, inspect, os
from pylab import *
import spectogram
import time

path = os.path.dirname(inspect.getsourcefile(gse2tests))
file = os.path.join(path, 'data', 'loc_RJOB20050831023349.z')

g = obspy.read(file,format='GSE2')

s = time.clock()
#spectogram.spec(inp=g[0].data,sample_rate=200.0,samp_length=len(g[0].data),log=True)
spectogram.spectoGram(g[0].data,samp_rate=200.0,log=True,outfile='hallo.png')
print "Running time:", time.clock() - s, '[s]'

#clf()
#plot(g.data)
#show()
