"""
Displays the waveform data from a given Mini-SEED file using gnuplot or
matplotlib

Only works with Linux and installed gnuplot
If it does not work make sure gnuplot is in $PATH

DO NOT USE WITH BIG MINISEED RECORDS as it will be likely to crash or take a
long time.

It will override the file 'graph.png' if present.
"""
from obspy.mseed.libmseed import libmseed
import os
import sys

try:
    filename = sys.argv[1]
except:
    filename = 'data/test.mseed'

try:
    output=sys.argv[2]
except:
    output='graph.png'

try:
    xres = sys.argv[3]
except:
    xres = 1400

try:
    yres = sys.argv[4]
except:
    yres = 900

print 'Reading data...'
mseed=libmseed()
header, data, numtraces =mseed.read_ms(filename)


if True:
    # plotting with matplotlib
    from pylab import *
    plot(array(data))
    savefig(output)
else:
    #Writes the file containing the data
    print 'Processing data...'
    datafile=file('dat.tmp','w')
    for _i in range(data.__len__()):
        datafile.write(str(_i)+" "+str(data[_i])+"\n")
    datafile.close()

    #Writes the gnuplot configuration file
    print 'Configuring gnuplot...'
    gnufile=file('gnuplotconf.tmp','w')
    gnufile.write('set terminal png\n')
    gnufile.write('set output "'+output+'"\n')
    gnufile.write('set term png size '+str(xres)+','+str(yres)+'\n')
    gnufile.write('set nokey\n')
    gnufile.write('set title "File: '+filename+'"\n')
    gnufile.write('plot "dat.tmp" with lines \n')
    gnufile.close()
    
    print 'Creating graph...'
    os.system('gnuplot gnuplotconf.tmp')
    print 'Cleaning up temporary files...'
    os.remove('dat.tmp')
    os.remove('gnuplotconf.tmp')


print 'Done: Plotted '+filename+' to '+output+'.'
