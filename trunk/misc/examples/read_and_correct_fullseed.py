#2008-11-27 Moritz
"""
Script to read Full SEED and correct the instrument response

USAGE: read_and_correct_fullseed.py seedfile
"""

from obspy.core import read
from obspy.xseed import Parser
from obspy.signal import seisSim, cosTaper, highpass
from matplotlib.mlab import detrend
import matplotlib.pyplot as plt
import sys

try:
    file = sys.argv[1]
except:
    print __doc__
    raise

# parse DataLess part
sp = Parser(file)

# parse DataOnly/MiniSEED part
stream = read(file)

for tr in stream:
    # get poles, zeros, sensitivity and gain
    paz = sp.getPAZ(tr.stats.channel)
    # Uncomment the following for:
    # Integrate by adding a zero at the position zero
    # As for the simulation the poles and zeros are inverted and convolved
    # in the frequency domain this is basically mutliplying by 1/jw which
    # is an integration in the frequency domain
    # See "Of Poles and Zeros", Frank Scherbaum, Springer 2007
    #paz['zeros'].append(0j)
    # preprocessing
    tr.data = tr.data.astype('float64')     #convert data to float
    tr.data = detrend(tr.data, 'linear')    #detrend
    tr.data *= cosTaper(tr.stats.npts, 0.10) #costaper 5% at start and end
    # correct for instrument, play with water_level
    # this will results to unit of XSEEDs tag stage_signal_output_units
    # most common for seed is m/s, write xseed by sp.writeXSEED('xs.txt')
    tr.data = seisSim(tr.data, tr.stats.sampling_rate, paz, inst_sim=None, 
                      water_level=60.0)
    tr.data = tr.data/paz['sensitivity']
    # You need to do postprocessing the low freq are most likely artefacts (result from
    # dividing the freqresp / to high water_level), use a highpass to get
    # rid of the artefacts, e.g. highpass at e.g. 2.0Hz
    #tr.data = highpass(tr.data, 2.0, df=tr.stats.sampling_rate, corners=2)


#
# the plotting part
#
m = stream.count()
for i, tr in enumerate(stream):
    plt.subplot(m, 1, i+1)
    plt.plot(tr.data)
    plt.ylabel(tr.stats.channel)
plt.show()

