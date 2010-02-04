#!/usr/bin/env python
# 2010-01-27 Moritz Beyreuther
"""
USAGE: obspyscanp.py [-f MSEED] file1 file2 file3 ...

Plot range of available data in files. Gaps are plotted as vertical red
lines, startimes of available data are plotted as crosses. The sampling
rate must stay the same of each station, but can vary between the stations.

Supported formats: MSEED, GSE2, SAC, WAV, SH-ASC, SH-Q, SEISAN. If the
format option (-f) is specified, the reading is significantly faster,
otherwise the format is autodetected
"""

import sys
from obspy.core import read
from optparse import OptionParser
from matplotlib.dates import date2num, num2date
from matplotlib.pyplot import figure, show, savefig
import numpy as np

def main():
    parser = OptionParser(__doc__.strip())
    parser.add_option("-f", "--format", default=None,
                      type="string", dest="format",
                      help="Format log file to test report")
    (options, largs) = parser.parse_args()
    
    #
    # Generate dictionary containing nested lists of start and end times per
    # station
    data = {}
    samp_int = {}
    for i, file in enumerate(largs):
        try:
            stream = read(file, headonly=True)
        except:
            print "Can not read", file
            continue
        s = "%s %s" % (i, file)
        sys.stdout.write('\b' * len(s) + s)
        for tr in stream:
            _id = tr.getId()
            data.setdefault(_id, [])
            data[_id].append([date2num(tr.stats.starttime),
                              date2num(tr.stats.endtime)])
            samp_int.setdefault(_id, 1.0 / (24 * 3600 * tr.stats.sampling_rate))
    
    #
    # Loop throught this dictionary
    ids = data.keys()
    ids.sort()
    fig = figure()
    ax = fig.add_subplot(111)
    for _i, _id in enumerate(ids):
        data[_id].sort()
        startend = np.array(data[_id])
        offset = np.ones(len(startend)) * _i #generate list of y values
        ax.plot_date(startend[:, 0], offset, 'x', linewidth=2)
        ax.hlines(offset, startend[:,0], startend[:,1])
        # find the gaps
        diffs = startend[1:,0] - startend[:-1,1] #currend.start - last.end
        gaps = startend[diffs > 1.8 * samp_int[_id], 1]
        if len(gaps) > 0:
            offset = offset[:len(gaps)]
            ax.vlines(gaps, offset-0.4, offset + 0.4, 'r', linewidth=1)
    
    #
    # Pretty format the plot
    ax.set_ylim(0-0.5, _i + 0.5)
    ax.set_yticks(np.arange(_i + 1), ids)
    fig.autofmt_xdate() #rotate date
    ax.set_title(" --- ".join(x.strftime('%Y%m%d %H:%M:%S')
                              for x in num2date(ax.get_xlim())))
    show()
    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
