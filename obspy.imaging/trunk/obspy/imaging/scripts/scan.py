#!/usr/bin/env python
# 2010-01-27 Moritz Beyreuther
"""
USAGE: obspy-scan [-f MSEED] [OPTIONS] file1 file2 dir1 dir2 file3 ...

Scan all specified files/directories, determine which time spans are covered
for which stations and plot everything in summarized in one overview plot.
Start times of traces with available data are marked by crosses, gaps are
indicated by vertical red lines.
The sampling rate must stay the same for each station, but may vary between the
stations.

Directories can also be used as arguments. By default they are scanned
recursively (disable with "-n"). Symbolic links are followed by default
(disable with "-i"). Detailed information on all files is printed using "-v".

Supported formats: All formats supported by ObsPy modules (currently: MSEED,
GSE2, SAC, SACXY, WAV, SH-ASC, SH-Q, SEISAN).
If the format is known beforehand, the reading speed can be increased
significantly by explicitly specifying the file format ("-f FORMAT"), otherwise
the format is autodetected.

See also the example in the Tutorial section:
http://svn.geophysik.uni-muenchen.de/trac/obspy/wiki/ObspyTutorial
"""

import sys
import os
from obspy.core import read
from optparse import OptionParser
from matplotlib.dates import date2num, num2date
from matplotlib.pyplot import figure, show
import numpy as np


def parse_file_to_dict(data_dict, samp_int_dict, file, counter, format=None,
                       verbose=False, ignore_links=False):
    if ignore_links and os.path.islink(file):
        print "Ignoring symlink:", file
        return counter
    try:
        stream = read(file, format=format, headonly=True)
    except:
        print "Can not read", file
        return counter
    s = "%s %s" % (counter, file)
    if verbose:
        sys.stdout.write("%s\n" %s)
        for line in str(stream).split("\n"):
            sys.stdout.write("    " + line + "\n")
    else:
        sys.stdout.write("\r" + s)
        sys.stdout.flush()
    for tr in stream:
        _id = tr.getId()
        data_dict.setdefault(_id, [])
        data_dict[_id].append([date2num(tr.stats.starttime),
                               date2num(tr.stats.endtime)])
        samp_int_dict.setdefault(_id,
                                 1.0 / (24 * 3600 * tr.stats.sampling_rate))
    return (counter + 1)

def recursive_parse(data_dict, samp_int_dict, path, counter, format=None,
                    verbose=False, ignore_links=False):
    if ignore_links and os.path.islink(path):
        print "Ignoring symlink:", path
        return counter
    if os.path.isfile(path):
        counter = parse_file_to_dict(data_dict, samp_int_dict, path, counter, format, verbose)
    elif os.path.isdir(path):
        for file in (os.path.join(path, file) for file in os.listdir(path)):
            counter = recursive_parse(data_dict, samp_int_dict, file, counter, format, verbose, ignore_links)
    else:
        print "Problem with filename/dirname:", path
    return counter
    

def main():
    parser = OptionParser(__doc__.strip())
    parser.add_option("-f", "--format", default=None,
                      type="string", dest="format",
                      help="Optional, the file format.\n" + \
                      " ".join(__doc__.split('\n')[-4:]))
    parser.add_option("-v", "--verbose", default=False,
                      action="store_true", dest="verbose",
                      help="Optional. Verbose output.")
    parser.add_option("-n", "--non-recursive", default=True,
                      action="store_false", dest="recursive",
                      help="Optional. Do not descend into directories.")
    parser.add_option("-i", "--ignore-links", default=False,
                      action="store_true", dest="ignore_links",
                      help="Optional. Do not follow symbolic links.")
    (options, largs) = parser.parse_args()

    # Print help and exit if no arguments are given
    if len(largs) == 0:
        parser.print_help()
        sys.exit(1)

    # Use recursively parsing function?
    if options.recursive:
        parse_func = recursive_parse
    else:
        parse_func = parse_file_to_dict
    #
    # Generate dictionary containing nested lists of start and end times per
    # station
    data = {}
    samp_int = {}
    for path in largs:
        counter = parse_func(data, samp_int, path, 1, options.format,
                             options.verbose, options.ignore_links)
    
    if not data:
        print "No waveform data found."
        return
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
        ax.hlines(offset, startend[:, 0], startend[:, 1])
        # find the gaps
        diffs = startend[1:, 0] - startend[:-1, 1] #currend.start - last.end
        gaps = startend[diffs > 1.8 * samp_int[_id], 1]
        if len(gaps) > 0:
            offset = offset[:len(gaps)]
            ax.vlines(gaps, offset - 0.4, offset + 0.4, 'r', linewidth=1)

    #
    # Pretty format the plot
    ax.set_ylim(0 - 0.5, _i + 0.5)
    ax.set_yticks(np.arange(_i + 1))
    ax.set_yticklabels(ids)
    fig.autofmt_xdate() #rotate date
    ax.set_title(" --- ".join(x.strftime('%Y%m%d %H:%M:%S')
                              for x in num2date(ax.get_xlim())))
    show()
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
