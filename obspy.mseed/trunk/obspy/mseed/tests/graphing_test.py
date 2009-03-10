# -*- coding: utf-8 -*-
"""
Test for quick plotting of large Mini-SEED files.

DO NOT USE FOR ANY SERIOUS PURPOSE AS THE PRODUCED GRAPH DOES NOT PRINT ALL
VALUES!

Uses matplotlib to plot the data. Please configure matplotlib to set the output
format. Might not work when using a GUI backend for matplotlib.

This test first splits all data values and determines the minimum and maximum
values for each vertical line that will be plotted (depends on the width of the
final graph).
Then it uses matplotlib to plot a vertical line between each pair of minumum
and maximum. This is probably a good approximation considering that each pair
resembles about 10000-20000 datasamples.

BUGS/TODO:
-The readTraces method of the libmseed wrapper does not always return sane
values. There are jumps that appear to be random. Maybe the values need
to be initialized in some way? I already tried various approches but none
worked.
-Make the graph prettier and adjustable.
"""

from obspy.mseed import libmseed
from pylab import *
import time
from user import home

#Do not change file as the hack used in this example only works with this file
file='tests/data/BW.BGLD..EHE.D.2008.001'
outfile=home+'/graphtest'

mseed=libmseed()
a = time.time()

#Creates minmaxlist
mm=mseed.graph_createMinMaxList(file = file, width = 1000)

#Hack list and remove all small values
for _i in range(len(mm)):
    if mm[_i][0]<4000000 or mm[_i][1]<4000000:
        mm[_i]=[]
hackedlist = [x for x in mm if x != []]
length = len(hackedlist)

#Determine yrange
miny = 99999999999999999
maxy = 0
for _i in range(length):
    if hackedlist[_i][0] < miny:
        miny = hackedlist[_i][0]
    if hackedlist[_i][1] > maxy:
        maxy = hackedlist[_i][1]

#Make values smaller
for _i in range(length):
    hackedlist[_i][0] = int(hackedlist[_i][0]-miny)
    hackedlist[_i][1] = int(hackedlist[_i][1]-miny)

#Set new xrange
maxy=int(maxy-miny)
miny=0

#draw horizontal lines
for _i in range(length):
    yy = float(hackedlist[_i][0])/float(maxy)
    xx = float(hackedlist[_i][1])/float(maxy)
    axvline(x = _i, ymin = yy, ymax = xx)

#Set axes
ylim(miny, maxy)
xlim(0,length)

#Save file
savefig(outfile)

b = time.time()
print 'Total time taken:', b-a