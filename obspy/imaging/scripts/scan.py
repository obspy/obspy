#!/usr/bin/env python
# 2010-01-27 Moritz Beyreuther
"""
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
options --no-x and --no-gaps can help to reduce the size of the plot
considerably.

Gap data can be written to a NumPy npz file. This file can be loaded later
for optionally adding more data and plotting.

Supported formats: All formats supported by ObsPy modules (currently: MSEED,
GSE2, SAC, SACXY, WAV, SH-ASC, SH-Q, SEISAN).
If the format is known beforehand, the reading speed can be increased
significantly by explicitly specifying the file format ("-f FORMAT"), otherwise
the format is autodetected.

See also the example in the Tutorial section:
https://tutorial.obspy.org
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import sys
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np

from obspy import UTCDateTime, __version__, read
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.misc import MatplotlibBackend
from obspy.imaging.util import ObsPyAutoDateFormatter, \
    decimal_seconds_format_date_first_tick


def compress_start_end(x, stop_iteration, merge_overlaps=False):
    """
    Compress 2-dimensional array of piecewise continuous start/end time pairs
    by merging overlapping and exactly fitting pieces into one.
    This reduces the number of lines needed in the plot considerably and is
    necessary for very large data sets.
    The maximum number of iterations can be specified.
    """
    diffs = x[1:, 0] - x[:-1, 1]
    if merge_overlaps:
        diffs[diffs < 0] = 0
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
        # to use fast NumPy methods currently we only can merge two consecutive
        # pieces, so we set every second entry to False
        inds[first_ind + 1::2] = False
        inds_next = np.roll(inds, 1)
        x[inds, 1] = x[inds_next, 1]
        inds_del = np.nonzero(inds_next)
        x = np.delete(x, inds_del, 0)
        diffs = x[1:, 0] - x[:-1, 1]
        if merge_overlaps:
            diffs[diffs < 0] = 0
        inds = np.concatenate([(diffs <= 0), [False]])
    return x


def parse_file_to_dict(data_dict, samp_int_dict, file, counter, format=None,
                       verbose=False, quiet=False, ignore_links=False):
    from matplotlib.dates import date2num
    if ignore_links and os.path.islink(file):
        if verbose or not quiet:
            print("Ignoring symlink: %s" % (file))
        return counter
    try:
        stream = read(file, format=format, headonly=True)
    except:
        if verbose or not quiet:
            print("Can not read %s" % (file))
        return counter
    s = "%s %s" % (counter, file)
    if verbose and not quiet:
        sys.stdout.write("%s\n" % s)
        for line in str(stream).split("\n"):
            sys.stdout.write("    " + line + "\n")
        sys.stdout.flush()
    for tr in stream:
        _id = tr.get_id()
        _samp_int_list = samp_int_dict.setdefault(_id, [])
        try:
            _samp_int_list.\
                append(1. / (24 * 3600 * tr.stats.sampling_rate))
        except ZeroDivisionError:
            if verbose or not quiet:
                print("Skipping file with zero samlingrate: %s" % (file))
            return counter
        _data_list = data_dict.setdefault(_id, [])
        _data_list.append([date2num(tr.stats.starttime.datetime),
                           date2num(tr.stats.endtime.datetime)])
    return (counter + 1)


def recursive_parse(data_dict, samp_int_dict, path, counter, format=None,
                    verbose=False, quiet=False, ignore_links=False):
    if ignore_links and os.path.islink(path):
        if verbose or not quiet:
            print("Ignoring symlink: %s" % (path))
        return counter
    if os.path.isfile(path):
        counter = parse_file_to_dict(data_dict, samp_int_dict, path, counter,
                                     format, verbose, quiet=quiet)
    elif os.path.isdir(path):
        for file in (os.path.join(path, file) for file in os.listdir(path)):
            counter = recursive_parse(data_dict, samp_int_dict, file, counter,
                                      format, verbose, quiet, ignore_links)
    else:
        if verbose or not quiet:
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


def main(argv=None):
    parser = ArgumentParser(prog='obspy-scan', description=__doc__.strip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-f', '--format', choices=ENTRY_POINTS['waveform'],
                        help='Optional, the file format.\n' +
                             ' '.join(__doc__.split('\n')[-4:]))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Optional. Verbose output.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Optional. Be quiet. Overwritten by --verbose '
                             'flag.')
    parser.add_argument('-n', '--non-recursive',
                        action='store_false', dest='recursive',
                        help='Optional. Do not descend into directories.')
    parser.add_argument('-i', '--ignore-links', action='store_true',
                        help='Optional. Do not follow symbolic links.')
    parser.add_argument('--start-time', default=None, type=UTCDateTime,
                        help='Optional, a UTCDateTime compatible string. ' +
                             'Only visualize data after this time and set ' +
                             'time-axis axis accordingly.')
    parser.add_argument('--end-time', default=None, type=UTCDateTime,
                        help='Optional, a UTCDateTime compatible string. ' +
                             'Only visualize data before this time and set ' +
                             'time-axis axis accordingly.')
    parser.add_argument('--id', action='append',
                        help='Optional, a SEED channel identifier '
                             "(e.g. 'GR.FUR..HHZ'). You may provide this " +
                             'option multiple times. Only these ' +
                             'channels will be plotted.')
    parser.add_argument('-t', '--event-time', default=None, type=UTCDateTime,
                        action='append',
                        help='Optional, a UTCDateTime compatible string ' +
                             "(e.g. '2010-01-01T12:00:00'). You may provide " +
                             'this option multiple times. These times get ' +
                             'marked by vertical lines in the plot. ' +
                             'Useful e.g. to mark event origin times.')
    parser.add_argument('-w', '--write', default=None,
                        help='Optional, npz file for writing data '
                             'after scanning waveform files')
    parser.add_argument('-l', '--load', default=None,
                        help='Optional, npz file for loading data '
                             'before scanning waveform files')
    parser.add_argument('--no-x', action='store_true',
                        help='Optional, Do not plot crosses.')
    parser.add_argument('--no-gaps', action='store_true',
                        help='Optional, Do not plot gaps.')
    parser.add_argument('-o', '--output', default=None,
                        help='Save plot to image file (e.g. out.pdf, ' +
                             'out.png) instead of opening a window.')
    parser.add_argument('--print-gaps', action='store_true',
                        help='Optional, prints a list of gaps at the end.')
    parser.add_argument('paths', nargs='*',
                        help='Files or directories to scan.')

    args = parser.parse_args(argv)

    if args.output is not None:
        MatplotlibBackend.switch_backend("AGG", sloppy=False)

    # Print help and exit if no arguments are given
    if len(args.paths) == 0 and args.load is None:
        parser.error('No paths specified.')

    # Use recursively parsing function?
    if args.recursive:
        parse_func = recursive_parse
    else:
        parse_func = parse_file_to_dict

    from matplotlib.dates import date2num, num2date
    from matplotlib.ticker import FuncFormatter
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot vertical lines if option 'event_time' was specified
    if args.event_time:
        times = [date2num(t.datetime) for t in args.event_time]
        for time in times:
            ax.axvline(time, color='k')

    if args.start_time:
        args.start_time = date2num(args.start_time.datetime)
    if args.end_time:
        args.end_time = date2num(args.end_time.datetime)

    # Generate dictionary containing nested lists of start and end times per
    # station
    data = {}
    samp_int = {}
    counter = 1
    if args.load:
        load_npz(args.load, data, samp_int)
    for path in args.paths:
        counter = parse_func(data, samp_int, path, counter, args.format,
                             verbose=args.verbose, quiet=args.quiet,
                             ignore_links=args.ignore_links)
    if not data:
        if args.verbose or not args.quiet:
            print("No waveform data found.")
        return
    if args.write:
        write_npz(args.write, data, samp_int)

    # either use ids specified by user or use ids based on what data we have
    # parsed
    ids = args.id or list(data.keys())
    ids = sorted(ids)[::-1]
    labels = [""] * len(ids)
    if args.verbose or not args.quiet:
        print('\n')
    for _i, _id in enumerate(ids):
        labels[_i] = ids[_i]
        # sort data list and sampling rate list
        if _id in data:
            startend = np.array(data[_id])
            _samp_int = np.array(samp_int[_id])
            indices = np.lexsort((startend[:, 1], startend[:, 0]))
            startend = startend[indices]
            _samp_int = _samp_int[indices]
        else:
            startend = np.array([])
            _samp_int = np.array([])
        if len(startend) == 0:
            if not (args.start_time and args.end_time):
                continue
            if not args.no_gaps:
                rects = [Rectangle((args.start_time, _i - 0.4),
                                   args.end_time - args.start_time, 0.8)]
                ax.add_collection(PatchCollection(rects, color="r"))
            if args.print_gaps and (args.verbose or not args.quiet):
                print("%s %s %s %.3f" % (
                    _id, args.start_time, args.end_time,
                    args.end_time - args.start_time))
            continue
        # restrict plotting of results to given start/end time
        if args.start_time:
            indices = startend[:, 1] > args.start_time
            startend = startend[indices]
            _samp_int = _samp_int[indices]
        if len(startend) == 0:
            continue
        if args.end_time:
            indices = startend[:, 0] < args.end_time
            startend = startend[indices]
            _samp_int = _samp_int[indices]
        if len(startend) == 0:
            continue
        data_start = startend[:, 0].min()
        data_end = startend[:, 1].max()
        timerange_start = args.start_time or data_start
        timerange_end = args.end_time or data_end
        timerange = timerange_end - timerange_start
        if timerange == 0.0:
            warnings.warn('Zero sample long data for _id=%s, skipping' % _id)
            continue

        startend_compressed = compress_start_end(startend, 1000)

        offset = np.ones(len(startend)) * _i  # generate list of y values
        if not args.no_x:
            ax.plot(startend[:, 0], offset, 'x', linewidth=2)
        ax.hlines(offset[:len(startend_compressed)], startend_compressed[:, 0],
                  startend_compressed[:, 1], 'b', linewidth=2, zorder=3)
        # find the gaps
        diffs = startend[1:, 0] - startend[:-1, 1]  # currend.start - last.end
        gapsum = diffs[diffs > 0].sum()
        # if start- and/or endtime is specified, add missing data at start/end
        # to gap sum
        has_gap = False
        gap_at_start = (
            args.start_time and
            data_start > args.start_time and
            data_start - args.start_time)
        gap_at_end = (
            args.end_time and
            args.end_time > data_end and
            args.end_time - data_end)
        if args.start_time and gap_at_start:
            gapsum += gap_at_start
            has_gap = True
        if args.end_time and gap_at_end:
            gapsum += gap_at_end
            has_gap = True
        perc = (timerange - gapsum) / timerange
        labels[_i] = labels[_i] + "\n%.1f%%" % (perc * 100)
        gap_indices = diffs > 1.8 * _samp_int[:-1]
        gap_indices = np.append(gap_indices, False)
        has_gap |= any(gap_indices)
        if has_gap:
            # don't handle last end time as start of gap
            gaps_start = startend[gap_indices, 1]
            gaps_end = startend[np.roll(gap_indices, 1), 0]
            if args.start_time and gap_at_start:
                gaps_start = np.append(gaps_start, args.start_time)
                gaps_end = np.append(gaps_end, data_start)
            if args.end_time and gap_at_end:
                gaps_start = np.append(gaps_start, data_end)
                gaps_end = np.append(gaps_end, args.end_time)
            if not args.no_gaps:
                rects = [Rectangle((start_, offset[0] - 0.4), end_ - start_,
                                   0.8)
                         for start_, end_ in zip(gaps_start, gaps_end)]
                ax.add_collection(PatchCollection(rects, color="r"))
            if args.print_gaps:
                for start_, end_ in zip(gaps_start, gaps_end):
                    start_, end_ = num2date((start_, end_))
                    start_ = UTCDateTime(start_.isoformat())
                    end_ = UTCDateTime(end_.isoformat())
                    if args.verbose or not args.quiet:
                        print("%s %s %s %.3f" % (_id, start_, end_,
                                                 end_ - start_))

    # Pretty format the plot
    ax.set_ylim(0 - 0.5, len(ids) - 0.5)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels(labels, family="monospace", ha="right")
    fig.autofmt_xdate()  # rotate date
    ax.xaxis_date()
    # set custom formatters to always show date in first tick
    formatter = ObsPyAutoDateFormatter(ax.xaxis.get_major_locator())
    formatter.scaled[1 / 24.] = \
        FuncFormatter(decimal_seconds_format_date_first_tick)
    formatter.scaled.pop(1/(24.*60.))
    ax.xaxis.set_major_formatter(formatter)
    plt.subplots_adjust(left=0.2)
    # set x-axis limits according to given start/end time
    if args.start_time and args.end_time:
        ax.set_xlim(left=args.start_time, right=args.end_time)
    elif args.start_time:
        ax.set_xlim(left=args.start_time, auto=None)
    elif args.end_time:
        ax.set_xlim(right=args.end_time, auto=None)
    else:
        left, right = ax.xaxis.get_data_interval()
        x_axis_range = right - left
        ax.set_xlim(left - 0.05 * x_axis_range, right + 0.05 * x_axis_range)
    if args.output is None:
        plt.show()
    else:
        fig.set_dpi(72)
        height = len(ids) * 0.5
        height = max(4, height)
        fig.set_figheight(height)
        plt.tight_layout()

        if not args.start_time or not args.end_time:
            days = ax.get_xlim()
            days = days[1] - days[0]
        else:
            days = args.end_time - args.start_time

        width = max(6, days / 30.)
        width = min(width, height * 4)
        fig.set_figwidth(width)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
        plt.tight_layout()

        fig.savefig(args.output)
    if args.verbose and not args.quiet:
        sys.stdout.write('\n')


if __name__ == '__main__':
    main()
