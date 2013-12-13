#!/usr/bin/env python
# 2010-01-27 Moritz Beyreuther
"""
USAGE: obspy-scan [-f FORMAT] [OPTIONS] file1 file2 dir1 dir2 file3 ...

Scan all specified files/directories, determine which time spans are covered
for which stations and plot everything in summarized in one overview plot.
Start times of traces with available data are marked by crosses, gaps are
indicated by vertical red lines.
The sampling rate must stay the same for each station, but may vary between the
stations.

Directories can also be used as arguments. By default they are scanned
recursively (disable with "-n"). Symbolic links are followed by default
(disable with "-i"). Detailed information on all files is printed using "-v".

In case of memory problems during plotting with very large datasets, the
options --nox and --nogaps can help to reduce the size of the plot
considerably.

Gap data can be written to a numpy npz file. This file can be loaded later
for optionally adding more data and plotting.

Supported formats: All formats supported by ObsPy modules (currently: MSEED,
GSE2, SAC, SACXY, WAV, SH-ASC, SH-Q, SEISAN).
If the format is known beforehand, the reading speed can be increased
significantly by explicitly specifying the file format ("-f FORMAT"), otherwise
the format is autodetected.

See also the example in the Tutorial section:
http://tutorial.obspy.org
"""

import sys
import os
import warnings
from obspy import read, UTCDateTime
from optparse import OptionParser
import numpy as np


def compressStartend(x, stop_iteration):
    """
    Compress 2-dimensional array of piecewise continuous starttime/endtime
    pairs by merging overlapping and exactly fitting pieces into one.
    This reduces the number of lines needed in the plot considerably and is
    necessary for very large data sets.
    The maximum number of iterations can be specified.
    """
    diffs = x[1:, 0] - x[:-1, 1]
    inds = np.concatenate([(diffs <= 0), [False]])
    i = 0
    while any(inds):
        if i >= stop_iteration:
            msg = "Stopping to merge lines for plotting at iteration %d"
            msg = msg % i
            warnings.warn(msg)
            break
        i += 1
        first_ind = np.nonzero(inds)[0][0]
        # to use fast numpy methods currently we only can merge two consecutive
        # pieces, so we set every second entry to False
        inds[first_ind + 1::2] = False
        inds_next = np.roll(inds, 1)
        x[inds, 1] = x[inds_next, 1]
        inds_del = np.nonzero(inds_next)
        x = np.delete(x, inds_del, 0)
        diffs = x[1:, 0] - x[:-1, 1]
        inds = np.concatenate([(diffs <= 0), [False]])
    return x


def parse_file_to_dict(data_dict, samp_int_dict, file, counter, format=None,
                       verbose=False, ignore_links=False):
    from matplotlib.dates import date2num
    if ignore_links and os.path.islink(file):
        print("Ignoring symlink: %s" % (file))
        return counter
    try:
        stream = read(file, format=format, headonly=True)
    except:
        print("Can not read %s" % (file))
        return counter
    s = "%s %s" % (counter, file)
    if verbose:
        sys.stdout.write("%s\n" % s)
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
        try:
            samp_int_dict.setdefault(_id, [])
            samp_int_dict[_id].\
                append(1. / (24 * 3600 * tr.stats.sampling_rate))
        except ZeroDivisionError:
            print("Skipping file with zero samlingrate: %s" % (file))
            return counter
    return (counter + 1)


def recursive_parse(data_dict, samp_int_dict, path, counter, format=None,
                    verbose=False, ignore_links=False):
    if ignore_links and os.path.islink(path):
        print("Ignoring symlink: %s" % (path))
        return counter
    if os.path.isfile(path):
        counter = parse_file_to_dict(data_dict, samp_int_dict, path, counter,
                                     format, verbose)
    elif os.path.isdir(path):
        for file in (os.path.join(path, file) for file in os.listdir(path)):
            counter = recursive_parse(data_dict, samp_int_dict, file, counter,
                                      format, verbose, ignore_links)
    else:
        print("Problem with filename/dirname: %s" % (path))
    return counter


def write_npz(file_, data_dict, samp_int_dict):
    npz_dict = data_dict.copy()
    for key in samp_int_dict.keys():
        npz_dict[key + '_SAMP'] = samp_int_dict[key]
    np.savez(file_, **npz_dict)


def load_npz(file_, data_dict, samp_int_dict):
    npz_dict = np.load(file_)
    for key in npz_dict.keys():
        if key.endswith('_SAMP'):
            samp_int_dict[key[:-5]] = npz_dict[key].tolist()
        else:
            data_dict[key] = npz_dict[key].tolist()
    if hasattr(npz_dict, "close"):
        npz_dict.close()


def main(option_list=None):
    parser = OptionParser(__doc__.strip())
    parser.add_option("-f", "--format", default=None,
                      type="string", dest="format",
                      help="Optional, the file format.\n" +
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
    parser.add_option("--starttime", default=None,
                      type="string", dest="starttime",
                      help="Optional, a UTCDateTime compatible string. " +
                      "Only visualize data after this time and set " +
                      "time-axis axis accordingly.")
    parser.add_option("--endtime", default=None,
                      type="string", dest="endtime",
                      help="Optional, a UTCDateTime compatible string. " +
                      "Only visualize data after this time and set " +
                      "time-axis axis accordingly.")
    parser.add_option("--ids", default=None,
                      type="string", dest="ids",
                      help="Optional, a list of SEED channel identifiers " +
                      "separated by commas " +
                      "(e.g. 'GR.FUR..HHZ,BW.MANZ..EHN'). Only these " +
                      "channels will be plotted.")
    parser.add_option("-t", "--event-times", default=None,
                      type="string", dest="event_times",
                      help="Optional, a list of UTCDateTime compatible " +
                      "strings separated by commas " +
                      "(e.g. '2010-01-01T12:00:00,2010-01-01T13:00:00'). " +
                      "These get marked by vertical lines in the plot. " +
                      "Useful e.g. to mark event origin times.")
    parser.add_option("-w", "--write", default=None,
                      type="string", dest="write",
                      help="Optional, npz file for writing data "
                      "after scanning waveform files")
    parser.add_option("-l", "--load", default=None,
                      type="string", dest="load",
                      help="Optional, npz file for loading data "
                      "before scanning waveform files")
    parser.add_option("--nox", default=False,
                      action="store_true", dest="nox",
                      help="Optional, Do not plot crosses.")
    parser.add_option("--nogaps", default=False,
                      action="store_true", dest="nogaps",
                      help="Optional, Do not plot gaps.")
    parser.add_option("-o", "--output", default=None,
                      type="string", dest="output",
                      help="Save plot to image file (e.g. out.pdf, " +
                      "out.png) instead of opening a window.")
    parser.add_option("--print-gaps", default=False,
                      action="store_true", dest="print_gaps",
                      help="Optional, prints a list of gaps at the end.")
    (options, largs) = parser.parse_args(option_list)

    # Print help and exit if no arguments are given
    if len(largs) == 0 and options.load is None:
        parser.print_help()
        sys.exit(1)

    # Use recursively parsing function?
    if options.recursive:
        parse_func = recursive_parse
    else:
        parse_func = parse_file_to_dict

    if options.output is not None:
        import matplotlib
        matplotlib.use("agg")
    global date2num
    from matplotlib.dates import date2num, num2date
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot vertical lines if option 'event_times' was specified
    if options.event_times:
        times = options.event_times.split(',')
        times = map(UTCDateTime, times)
        times = map(date2num, times)
        for time in times:
            ax.axvline(time, color='k')

    if options.starttime:
        options.starttime = UTCDateTime(options.starttime)
        options.starttime = date2num(options.starttime)
    if options.endtime:
        options.endtime = UTCDateTime(options.endtime)
        options.endtime = date2num(options.endtime)

    # Generate dictionary containing nested lists of start and end times per
    # station
    data = {}
    samp_int = {}
    counter = 1
    if options.load:
        load_npz(options.load, data, samp_int)
    for path in largs:
        counter = parse_func(data, samp_int, path, counter, options.format,
                             options.verbose, options.ignore_links)
    if not data:
        print("No waveform data found.")
        return
    if options.write:
        write_npz(options.write, data, samp_int)

    # Loop through this dictionary
    ids = data.keys()
    # restrict plotting of results to given ids
    if options.ids:
        options.ids = options.ids.split(',')
        ids = filter(lambda x: x in options.ids, ids)
    ids = sorted(ids)[::-1]
    labels = [""] * len(ids)
    print
    for _i, _id in enumerate(ids):
        labels[_i] = ids[_i]
        data[_id].sort()
        startend = np.array(data[_id])
        if len(startend) == 0:
            continue
        # restrict plotting of results to given start/endtime
        if options.starttime:
            startend = startend[startend[:, 1] > options.starttime]
        if len(startend) == 0:
            continue
        if options.starttime:
            startend = startend[startend[:, 0] < options.endtime]
        if len(startend) == 0:
            continue
        timerange = startend[:, 1].max() - startend[:, 0].min()
        if timerange == 0.0:
            warnings.warn('Zero sample long data for _id=%s, skipping' % _id)
            continue

        startend_compressed = compressStartend(startend, 1000)

        offset = np.ones(len(startend)) * _i  # generate list of y values
        ax.xaxis_date()
        if not options.nox:
            ax.plot_date(startend[:, 0], offset, 'x', linewidth=2)
        ax.hlines(offset[:len(startend_compressed)], startend_compressed[:, 0],
                  startend_compressed[:, 1], 'b', linewidth=2, zorder=3)
        # find the gaps
        diffs = startend[1:, 0] - startend[:-1, 1]  # currend.start - last.end
        gapsum = diffs[diffs > 0].sum()
        perc = (timerange - gapsum) / timerange
        labels[_i] = labels[_i] + "\n%.1f%%" % (perc * 100)
        gap_indices = diffs > 1.8 * np.array(samp_int[_id][:-1])
        gap_indices = np.concatenate((gap_indices, [False]))
        if any(gap_indices):
            # dont handle last endtime as start of gap
            gaps_start = startend[gap_indices, 1]
            gaps_end = startend[np.roll(gap_indices, 1), 0]
            if not options.nogaps and any(gap_indices):
                rects = [Rectangle((start_, offset[0] - 0.4),
                                   end_ - start_, 0.8)
                         for start_, end_ in zip(gaps_start, gaps_end)]
                ax.add_collection(PatchCollection(rects, color="r"))
            if options.print_gaps:
                for start_, end_ in zip(gaps_start, gaps_end):
                    start_, end_ = num2date((start_, end_))
                    start_ = UTCDateTime(start_.isoformat())
                    end_ = UTCDateTime(end_.isoformat())
                    print "%s %s %s %.3f" % (_id, start_, end_, end_ - start_)

    # Pretty format the plot
    ax.set_ylim(0 - 0.5, _i + 0.5)
    ax.set_yticks(np.arange(_i + 1))
    ax.set_yticklabels(labels, family="monospace", ha="right")
    # set x-axis limits according to given start/endtime
    if options.starttime:
        ax.set_xlim(left=options.starttime, auto=None)
    if options.endtime:
        ax.set_xlim(right=options.endtime, auto=None)
    fig.autofmt_xdate()  # rotate date
    plt.subplots_adjust(left=0.2)
    if options.output is None:
        plt.show()
    else:
        fig.set_dpi(72)
        height = len(ids) * 0.5
        height = max(4, height)
        fig.set_figheight(height)
        # tight_layout() only available from matplotlib >= 1.1
        try:
            plt.tight_layout()
            days = ax.get_xlim()
            days = days[1] - days[0]
            width = max(6, days / 30.)
            width = min(width, height * 4)
            fig.set_figwidth(width)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
            plt.tight_layout()
        except:
            pass
        fig.savefig(options.output)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
