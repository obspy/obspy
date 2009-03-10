# -*- coding: utf-8 -*-
from obspy.mseed import libmseed
import os

path = 'C:\\Users\\Robert\\Workspace\\obspy\\obspy.mseed\\trunk\\obspy\\mseed\\tests'
mseed_file = os.path.join(path, 'data', 'test.mseed')

mseed=libmseed()
#header, data, numtraces=mseed.read_ms_using_traces(mseed_file)
data = mseed.graph_createMinMaxList(mseed_file, 500)


# do this before importing pylab or pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data)
fig.savefig('test.png')
