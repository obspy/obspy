#!/usr/bin/env python
# 2010-01-27 Moritz Beyreuther
"""
Scan a directory to determine the data availability.

Scan all specified files/directories, determine which time spans are covered
for which stations and plot everything in summarized in one overview plot.
Start times of traces with available data are marked by crosses, gaps are
indicated by vertical red lines, and overlaps are indicated by blue lines.
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
import fnmatch
import os
import sys
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.dates import date2num, num2date

from obspy import UTCDateTime, __version__, read, Trace, Stream
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.misc import MatplotlibBackend
from obspy.imaging.util import ObsPyAutoDateFormatter, \
    decimal_seconds_format_date_first_tick


def compress_start_end(x, stop_iteration, merge_overlaps=False,
                       margin_in_seconds=0.0):
    """
    Compress 2-dimensional array of piecewise continuous start/end time pairs
    (in matplotlib date numbers) by merging overlapping and exactly fitting
    pieces into one.
    This reduces the number of lines needed in the plot considerably and is
    necessary for very large data sets.
    The maximum number of iterations can be specified.

    Works in-place!

    :type margin_in_seconds: float
    :param margin_in_seconds: Allowance in seconds that has to be exceeded by
        adjacent expected next sample time (earlier trace's endtime+delta) and
        actual next sample time (later trace's starttime) so that the
        in-between is considered a gap or overlap (e.g. to allow for up to
        ``0.8`` times the sampling interval for a 100 Hz stream, use
        ``(1 / 100.0) * 0.8) == 0.008``).
    """
    # matplotlib date numbers are in days
    margin = margin_in_seconds / (24 * 3600)

    def _get_indices_to_merge(startend):
        """
        Return boolean array signaling at which positions a merge of adjacent
        tuples should be performed.
        """
        diffs = x[1:, 0] - x[:-1, 1]
        if merge_overlaps:
            # if overlaps should be merged we set any negative diff to zero and
            # it will be merged in the following commands
            diffs[diffs < 0] = 0
        # any diff of expected and actual next sample time that is smaller than
        # 0+-margin is considered no gap/overlap but rather merged together
        inds = np.concatenate(
            [(diffs >= -margin) & (diffs <= margin), [False]])
        return inds

    inds = _get_indices_to_merge(x)
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
        inds = _get_indices_to_merge(x)
    return x


def parse_file_to_dict(data_dict, samp_int_dict, file, counter, format=None,
                       verbose=False, quiet=False, ignore_links=False):
    if ignore_links and os.path.islink(file):
        if verbose or not quiet:
            print("Ignoring symlink: %s" % (file))
        return counter
    try:
        stream = read(file, format=format, headonly=True)
    except Exception:
        if verbose or not quiet:
            print("Can not read %s" % (file))
        return counter
    s = "%s %s" % (counter, file)
    if verbose and not quiet:
        sys.stdout.write("%s\n" % s)
        for line in str(stream).split("\n"):
            sys.stdout.write("    " + line + "\n")
        sys.stdout.flush()
    add_stream_to_dict(data_dict, samp_int_dict, stream,
                       verbose=verbose and not quiet)
    return (counter + 1)


def add_stream_to_dict(data_dict, samp_int_dict, stream, verbose=False):
    from matplotlib.dates import date2num
    for tr in stream:
        try:
            delta = 1. / (24 * 3600 * tr.stats.sampling_rate)
        except ZeroDivisionError:
            if verbose:
                print("Skipping trace with zero samlingrate: {!s}".format(tr))
            continue
        _id = tr.get_id()
        _samp_int_list = samp_int_dict.setdefault(_id, [])
        _data_list = data_dict.setdefault(_id, [])
        _samp_int_list.append(delta)
        _data_list.append(
            [date2num(tr.stats.starttime.datetime),
             date2num((tr.stats.endtime + tr.stats.delta).datetime)])


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
        try:
            dirlist = os.listdir(path)
        except PermissionError:
            if verbose or not quiet:
                print(f"Can not read directory {path} ('Permission denied')")
            return counter
        for file in (os.path.join(path, file) for file in dirlist):
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
    npz_dict["__version__"] = __version__
    np.savez(file_, **npz_dict)


def load_npz(file_, data_dict, samp_int_dict):
    npz_dict = np.load(file_)
    # check obspy version the npz was done with
    if "__version__" in npz_dict:
        version_string = npz_dict["__version__"].item()
        # saw some error loading a npz written on Py2 when loading on Py3 with
        # the version string being bytes so that the following `split(".")`
        # raises an Exception. Not sure what is going on and no time to track
        # it down properly right now, in any case the following hack should not
        # cause any problems and it's only used to check if a warning needs to
        # be shown at the moment
        try:
            version_string = version_string.decode('ASCII')
        except Exception:
            pass
    else:
        version_string = None
    # npz data computed with obspy < 1.1.0 are slightly different
    if version_string is None or \
            [int(x) for x in version_string.split(".")[:2]] < [1, 1]:
        msg = ("Loading npz data computed with ObsPy < 1.1.0. Definition of "
               "end times of individual time slices was changed by one time "
               "the sampling interval (see #1366), so it is best to recompute "
               "the npz from the raw data once.")
        warnings.warn(msg)
    # load data from npz
    for key in npz_dict.keys():
        if key == "__version__":
            continue
        elif key.endswith('_SAMP'):
            samp_int_dict[key[:-5]] = npz_dict[key].tolist()
        else:
            data_dict[key] = npz_dict[key].tolist()
    if hasattr(npz_dict, "close"):
        npz_dict.close()


class Scanner(object):
    """
    Class to scan contents of waveform files, file by file or recursively
    across directory trees.

    >>> scanner = Scanner()
    >>> scanner.parse("/some/directory/with/waveforms")  # doctest: +SKIP
    >>> scanner.plot()  # doctest: +SKIP

    .. plot::

        import os
        import obspy
        from obspy.imaging.scripts.scan import Scanner
        directory = os.path.join(os.path.dirname(obspy.__file__),
                                 "io", "gse2", "tests", "data")
        scanner = Scanner()
        scanner.parse(directory)
        scanner.plot()

    Information on gaps can be accessed in a dictionary structure as
    ``scanner._info`` after calling
    :meth:`~obspy.imaging.scripts.scan.Scanner.plot()` or
    :meth:`~obspy.imaging.scripts.scan.Scanner.analyze_parsed_data()` (all
    timing information is in matplotlib date numbers):

    >>> print(scanner._info.keys())  # doctest: +SKIP
    [u'.LE0083..  Z', u'.RNON..Z', u'.GRB1.. BZ', u'.RNHA..EHN', u'.GRA1.. BN',
     u'ABCD.ABCDE..HHZ', u'.RJOB..Z', u'.CLZ.. BZ',
     u'SEDZ.\x00\x00\x00\x00\x00..A\x00', u'.GRA1.. BZ']
    >>> print(scanner._info[".RJOB..Z"])  # doctest: +SKIP
    {u'data_startends_compressed': array(
        [[ 732189.10682697,  732189.10752141]]),
     u'data_starts': array(
        [ 732189.10682697,  732189.10682697,  732189.10682697]),
     u'gaps': [],
     u'overlaps': [(732189.10752141208, 732189.10682696756),
                   (732189.10752141208, 732189.10682696756)],
     u'percentage': 100.0}

    :type format: str
    :param format: Use fixed format for reading all files (e.g. ``MSEED``).
        This skips file format autodetection and speeds up reading waveform
        data if file format of all files is known to be the same (files with
        different waveform format will be skipped!).
    :type verbose: bool
    :param verbose: Whether to print information messages.
    :type recursive: bool
    :param recursive: Whether to parse directories recursively.
    :type ignore_links: bool
    :param ignore_links: Whether to ignore symbolic links.
    """
    def __init__(self, format=None, verbose=False, recursive=True,
                 ignore_links=False):
        """
        see :class:`~obspy.imaging.scripts.scan.Scanner`
        """
        self.format = format
        self.verbose = verbose
        self.recursive = recursive
        self.ignore_links = ignore_links
        # Generate dictionary containing nested lists of start and end times
        # per station
        self.data = {}
        self.samp_int = {}
        self.counter = 1

    def plot(self, outfile=None, show=True, fig=None, plot_x=True,
             plot_gaps=True, print_gaps=False, event_times=None,
             starttime=None, endtime=None, seed_ids=None):
        """
        Plot the information on parsed waveform files.

        :type outfile: str
        :param outfile: Filename for image output (e.g.
            ``"folder/image.png"``). No interactive plot is shown if an output
            filename is specified.
        :type show: bool
        :param show: Whether to open up any interactive plot after plotting.
        :type fig: :class:`matplotlib.figure.Figure`
        :param fig: Figure instance to reuse.
        :type plot_x: bool
        :param plot_x: Whether to plot "X" markers at start of all
            parsed :class:`~obspy.core.trace.Trace`s.
        :type plot_gaps: bool
        :param plot_gaps: Whether to plot filled rectangles at data gaps (red)
            and overlaps (blue).
        :type print_gaps: bool
        :param print_gaps: Whether to print information on all encountered gaps
            and overlaps.
        :type event_times: list of :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param event_times: Highlight given times (e.g. of events or phase
            onsets for visual inspection of data availability) by plotting
            vertical lines.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Whether to use a fixed start time for the plot and
            data percentage calculation.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Whether to use a fixed end time for the plot and
            data percentage calculation.
        :type seed_ids: list[str]
        :param seed_ids: Whether to consider only a specific set of SEED IDs
            (e.g. ``seed_ids=["GR.FUR..BHZ", "GR.WET..BHZ"]``) or just all SEED
            IDs encountered in data (if left ``None``). Given SEED IDs may
            contain ``fnmatch``-style wildcards (e.g. ``"BW.UH?..[EH]H*"``).
        """
        import matplotlib.pyplot as plt

        data_keys = list(self.data.keys())
        if seed_ids is not None:
            ids = []
            for id_ in seed_ids:
                # allow fnmatch type wildcards in given seed ids
                if any(special in id_ for special in '*?[]!'):
                    ids.extend(fnmatch.filter(data_keys, id_))
                else:
                    ids.append(id_)
            # make sure we don't have duplicates in case multiple wildcard
            # patterns were given and some ids were matched by more than one
            # pattern
            ids = list(set(ids))
            seed_ids = ids

        if fig:
            if fig.axes:
                ax = fig.axes[0]
            else:
                ax = fig.add_subplot(111)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        self.analyze_parsed_data(print_gaps=print_gaps, starttime=starttime,
                                 endtime=endtime, seed_ids=seed_ids)
        if starttime is not None:
            starttime = starttime.matplotlib_date
        if endtime is not None:
            endtime = endtime.matplotlib_date

        # Plot vertical lines if option 'event_time' was specified
        if event_times:
            times = [date2num(t.datetime) for t in event_times]
            for time in times:
                ax.axvline(time, color='k')

        labels = [""] * len(self._info)
        for _i, (id_, info) in enumerate(sorted(self._info.items(),
                                                reverse=True)):
            offset = np.ones(len(info["data_starts"])) * _i
            if plot_x:
                ax.plot(info["data_starts"], offset, 'x', linewidth=2)
            if len(info["data_startends_compressed"]):
                ax.hlines(offset[:len(info["data_startends_compressed"])],
                          info["data_startends_compressed"][:, 0],
                          info["data_startends_compressed"][:, 1],
                          'b', linewidth=2, zorder=3)

            label = id_
            if info["percentage"] is not None:
                label = label + "\n%.1f%%" % (info["percentage"])
            labels[_i] = label

            if plot_gaps:
                for key, color in zip(("gaps", "overlaps"), ("r", "b")):
                    data_ = info[key]
                    if len(data_):
                        rects = [
                            Rectangle((start_, _i - 0.4), end_ - start_, 0.8)
                            for start_, end_ in data_]
                        ax.add_collection(PatchCollection(rects, color=color))

        # Pretty format the plot
        ax.set_ylim(0 - 0.5, len(labels) - 0.5)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, family="monospace", ha="right")
        fig.autofmt_xdate()  # rotate date
        ax.xaxis_date()
        # set custom formatters to always show date in first tick
        formatter = ObsPyAutoDateFormatter(ax.xaxis.get_major_locator())
        formatter.scaled[1 / 24.] = \
            FuncFormatter(decimal_seconds_format_date_first_tick)
        formatter.scaled.pop(1 / (24. * 60.))
        ax.xaxis.set_major_formatter(formatter)
        plt.subplots_adjust(left=0.2)
        # set x-axis limits according to given start/end time
        if starttime and endtime:
            ax.set_xlim(left=starttime, right=endtime)
        elif starttime:
            ax.set_xlim(left=starttime, auto=None)
        elif endtime:
            ax.set_xlim(right=endtime, auto=None)
        else:
            left, right = ax.xaxis.get_data_interval()
            x_axis_range = right - left
            ax.set_xlim(left - 0.05 * x_axis_range,
                        right + 0.05 * x_axis_range)

        if outfile:
            fig.set_dpi(72)
            height = len(labels) * 0.5
            height = max(4, height)
            fig.set_figheight(height)
            plt.tight_layout()

            if not starttime or not endtime:
                days = ax.get_xlim()
                days = days[1] - days[0]
            else:
                days = endtime - starttime

            width = max(6, days / 30.)
            width = min(width, height * 4)
            fig.set_figwidth(width)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
            plt.tight_layout()

            fig.savefig(outfile)
            plt.close(fig)
        else:
            if show:
                plt.show()

        if self.verbose:
            sys.stdout.write('\n')
        return fig

    def analyze_parsed_data(self, print_gaps=False, starttime=None,
                            endtime=None, seed_ids=None):
        """
        Prepare information for plotting.

        Information is stored in a dictionary as ``scanner._info``, only
        containing these data matching the given parameters.

        :type print_gaps: bool
        :param print_gaps: Whether to print information on all encountered gaps
            and overlaps.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Whether to use a fixed start time for the plot and
            data percentage calculation.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Whether to use a fixed end time for the plot and
            data percentage calculation.
        :type seed_ids: list[str]
        :param seed_ids: Whether to consider only a specific set of SEED IDs
            (e.g. ``seed_ids=["GR.FUR..BHZ", "GR.WET..BHZ"]``) or just all SEED
            IDs encountered in data (if left ``None``).
        """
        data = self.data
        samp_int = self.samp_int
        if starttime is not None:
            starttime = starttime.matplotlib_date
        if endtime is not None:
            endtime = endtime.matplotlib_date
        # either use ids specified by user or use ids based on what data we
        # have parsed
        ids = seed_ids or list(data.keys())
        ids = sorted(ids)[::-1]
        if self.verbose:
            print('\n')
        self._info = {}
        for _i, _id in enumerate(ids):
            info = {"gaps": [], "overlaps": [], "data_starts": [],
                    "data_startends_compressed": [], "percentage": None}
            self._info[_id] = info
            gap_info = info["gaps"]
            overlap_info = info["overlaps"]
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
                if not (starttime and endtime):
                    continue
                gap_info.append((starttime, endtime))
                if print_gaps:
                    print("%s %s %s %.3f" % (
                        _id, starttime, endtime, endtime - starttime))
                continue
            # restrict plotting of results to given start/end time
            if starttime or endtime:
                indices = np.ones(len(startend), dtype=np.bool_)
                if starttime:
                    indices &= startend[:, 1] > starttime
                if endtime:
                    indices &= startend[:, 0] < endtime
                startend = startend[indices]
                _samp_int = _samp_int[indices]
            if len(startend) == 0:
                # if both start and endtime are given, add it to gap info
                if starttime and endtime:
                    gap_info.append((starttime, endtime))
                continue
            data_start = startend[:, 0].min()
            data_end = startend[:, 1].max()
            timerange_start = starttime or data_start
            timerange_end = endtime or data_end
            timerange = timerange_end - timerange_start
            if timerange == 0.0:
                msg = 'Zero sample long data for _id=%s, skipping' % _id
                warnings.warn(msg)
                continue

            startend_compressed = compress_start_end(startend.copy(), 1000,
                                                     merge_overlaps=True)

            info["data_starts"] = startend[:, 0]
            info["data_startends_compressed"] = startend_compressed

            # find the gaps
            # currend.start - last.end
            diffs = startend[1:, 0] - startend[:-1, 1]
            gapsum = diffs[diffs > 0].sum()
            # if start- and/or endtime is specified, add missing data at
            # start/end to gap sum
            has_gap = False
            gap_at_start = (
                starttime and
                data_start > starttime and
                data_start - starttime)
            gap_at_end = (
                endtime and
                endtime > data_end and
                endtime - data_end)
            if gap_at_start:
                gapsum += gap_at_start
                has_gap = True
            if gap_at_end:
                gapsum += gap_at_end
                has_gap = True
            info["percentage"] = (timerange - gapsum) / timerange * 100
            # define a gap as over 0.8 delta after expected sample time
            gap_indices = diffs > 0.8 * _samp_int[:-1]
            gap_indices = np.append(gap_indices, False)
            # define an overlap as over 0.8 delta before expected sample time
            overlap_indices = diffs < -0.8 * _samp_int[:-1]
            overlap_indices = np.append(overlap_indices, False)
            has_gap |= any(gap_indices)
            has_gap |= any(overlap_indices)
            if has_gap:
                # don't handle last end time as start of gap
                gaps_start = startend[gap_indices, 1]
                gaps_end = startend[np.roll(gap_indices, 1), 0]
                overlaps_end = startend[overlap_indices, 1]
                overlaps_start = startend[np.roll(overlap_indices, 1), 0]
                # but now, manually add start/end for gaps at start/end of user
                # specified start/end times
                if gap_at_start:
                    gaps_start = np.append(gaps_start, starttime)
                    gaps_end = np.append(gaps_end, data_start)
                if gap_at_end:
                    gaps_start = np.append(gaps_start, data_end)
                    gaps_end = np.append(gaps_end, endtime)

                _starts = np.concatenate((gaps_start, overlaps_end))
                _ends = np.concatenate((gaps_end, overlaps_start))
                sort_order = np.argsort(_starts)
                _starts = _starts[sort_order]
                _ends = _ends[sort_order]
                for start_, end_ in zip(_starts, _ends):
                    if print_gaps:
                        start__, end__ = num2date((start_, end_))
                        start__ = UTCDateTime(start__.isoformat())
                        end__ = UTCDateTime(end__.isoformat())
                        print("{} {} {} {:.3f}".format(
                            _id, start__, end__, end__ - start__))
                    if start_ < end_:
                        gap_info.append((start_, end_))
                    else:
                        overlap_info.append((start_, end_))

    def load_npz(self, filename):
        """
        Load information on scanned data from npz file.

        Currently, data can only be loaded from npz as the first operation,
        i.e. before parsing any files.

        :type filename: str
        :param filename: Filename to load from.
        """
        if self.data or self.samp_int:
            msg = ("Currently, data can only be loaded from npz as the first "
                   "operation, i.e. before parsing any files.")
            raise NotImplementedError(msg)
        load_npz(filename, data_dict=self.data, samp_int_dict=self.samp_int)

    def save_npz(self, filename):
        """
        Save information on scanned data to npz file.

        :type filename: str
        :param filename: Filename to save to.
        """
        write_npz(filename, data_dict=self.data, samp_int_dict=self.samp_int)

    def parse(self, path, recursive=None, ignore_links=None):
        """
        Parse file/directory and store information on encountered waveform
        files.

        :type path: str
        :param path: File or directory path (relative or absolute) to parse.
        :type recursive: bool
        :param recursive: Override for value of option ``recursive`` set at
            initialization.
        :type ignore_links: bool
        :param ignore_links: Override for value of option ``ignore_links`` set
            at initialization.
        """
        if recursive is None:
            recursive = self.recursive
        if ignore_links is None:
            ignore_links = self.ignore_links

        if recursive:
            parse_func = recursive_parse
        else:
            parse_func = parse_file_to_dict

        self.counter = parse_func(
            self.data, self.samp_int, path, self.counter, self.format,
            verbose=self.verbose, quiet=not self.verbose,
            ignore_links=ignore_links)

    def add_stream(self, stream):
        """
        Add information of provided stream to scanner object.

        :type stream: :class:`~obspy.core.stream.Stream` or
            :class:`~obspy.core.trace.Trace`
        """
        if isinstance(stream, Trace):
            stream = Stream(traces=[stream])
        add_stream_to_dict(self.data, self.samp_int, stream,
                           verbose=self.verbose)
        self.counter += 1


def scan(paths, format=None, verbose=False, recursive=True,
         ignore_links=False, starttime=None, endtime=None, seed_ids=None,
         event_times=None, npz_output=None, npz_input=None, plot_x=True,
         plot_gaps=True, print_gaps=False, plot=False):
    """
    :type plot: bool or str
    :param plot: False for no plot at all, True for interactive window, str for
        output to image file.
    """
    scanner = Scanner(format=format, verbose=verbose, recursive=recursive,
                      ignore_links=ignore_links)

    if plot is None:
        plot = False

    # Print help and exit if no arguments are given
    if len(paths) == 0 and npz_input is None:
        msg = "No paths specified and no npz data to load specified"
        raise ValueError(msg)

    if npz_input:
        scanner.load_npz(npz_input)
    for path in paths:
        scanner.parse(path)

    if not scanner.data:
        if verbose:
            print("No waveform data found.")
        return None
    if npz_output:
        scanner.save_npz(npz_output)

    kwargs = dict(starttime=starttime, endtime=endtime,
                  seed_ids=seed_ids)
    if plot:
        kwargs.update(dict(plot_x=plot_x, plot_gaps=plot_gaps,
                           print_gaps=print_gaps, event_times=event_times))
        if plot is True:
            scanner.plot(outfile=None, show=True, **kwargs)
        else:
            # plotting to file, so switch to non-interactive backend
            with MatplotlibBackend("AGG", sloppy=False):
                scanner.plot(outfile=plot, show=False, **kwargs)
    else:
        scanner.analyze_parsed_data(print_gaps=print_gaps, **kwargs)

    return scanner


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
                             "(e.g. 'GR.FUR..HHZ'). You may provide this "
                             'option multiple times. Only these '
                             'channels will be plotted. Given SEED IDs may '
                             'contain fnmatch-style wildcards (e.g. '
                             "'BW.UH?..[EH]H*').")
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

    if args.quiet:
        msg = ("'--quiet' is now the default behavior and the option has been "
               "deprecated.")
        warnings.warn(msg)

    scan(paths=args.paths, format=args.format, verbose=args.verbose,
         recursive=args.recursive,
         ignore_links=args.ignore_links, starttime=args.start_time,
         endtime=args.end_time, seed_ids=args.id,
         event_times=args.event_time, npz_output=args.write,
         npz_input=args.load, plot_x=not args.no_x,
         plot_gaps=not args.no_gaps, print_gaps=args.print_gaps,
         plot=args.output or True)


if __name__ == '__main__':
    main()
