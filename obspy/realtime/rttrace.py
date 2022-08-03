# -*- coding: utf-8 -*-
"""
Module for handling ObsPy RtTrace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import warnings

import numpy as np

from obspy import Trace
from obspy.core import Stats
from obspy.realtime import signal
from obspy.realtime.rtmemory import RtMemory


# dictionary to map given type-strings to processing functions keys must be all
# lower case - values are tuples: (function name, number of RtMemory objects)
REALTIME_PROCESS_FUNCTIONS = {
    'scale': (signal.scale, 0),
    'offset': (signal.offset, 0),
    'integrate': (signal.integrate, 1),
    'differentiate': (signal.differentiate, 1),
    'boxcar': (signal.boxcar, 1),
    'tauc': (signal.tauc, 2),
    'mwpintegral': (signal.mwpintegral, 1),
    'kurtosis': (signal.kurtosis, 3),
}


class RtTrace(Trace):
    """
    An object containing data of a continuous series constructed dynamically
    from sequential data packets.

    New data packets may be periodically appended. Registered time-domain
    processes can be applied to the new data and the resulting trace will be
    left trimmed to maintain a specified maximum trace length.

    :type max_length: int, optional
    :param max_length: maximum trace length in seconds

    .. rubric:: Example

    RtTrace has been built to handle real time processing of periodically
    append data packets, such as adding and processing data requested from an
    SeedLink server. See :mod:`obspy.clients.seedlink` for further information.

    For the sake of simplicity we will just split data of an existing example
    file into multiple chucks (Trace objects) of about equal size (step 1 + 2)
    and append those chunks in a simple loop (step 4) into an RtTrace object.
    Additionally there are two real time processing functions registered to the
    RtTrace object (step 3) which will automatically process any appended data
    chunks.

    1. Read first trace of example SAC data file and extract contained time
       offset and epicentral distance of an earthquake::

        >>> from obspy.realtime import RtTrace
        >>> from obspy import read
        >>> from obspy.realtime.signal import calculate_mwp_mag
        >>> data_trace = read('/path/to/II.TLY.BHZ.SAC')[0]
        >>> len(data_trace)
        12684
        >>> ref_time_offset = data_trace.stats.sac.a
        >>> print(ref_time_offset)
        301.506
        >>> epicentral_distance = data_trace.stats.sac.gcarc
        >>> print(epicentral_distance)
        30.0855

    2. Split given trace into a list of three sub-traces::

        >>> traces = data_trace / 3
        >>> [len(tr) for tr in traces]
        [4228, 4228, 4228]

    3. Assemble real time trace and register two processes::

        >>> rt_trace = RtTrace()
        >>> rt_trace.register_rt_process('integrate')
        1
        >>> rt_trace.register_rt_process('mwpintegral', mem_time=240,
        ...     ref_time=(data_trace.stats.starttime + ref_time_offset),
        ...     max_time=120, gain=1.610210e+09)
        2

    4. Append and auto-process packet data into RtTrace::

        >>> for tr in traces:
        ...     processed_trace = rt_trace.append(tr, gap_overlap_check=True)
        ...
        >>> len(rt_trace)
        12684

    5. Some post processing to get Mwp::

        >>> peak = np.amax(np.abs(rt_trace.data))
        >>> print(peak)
        0.136404
        >>> mwp = calculate_mwp_mag(peak, epicentral_distance)
        >>> print(mwp)  # doctest: +ELLIPSIS
        8.78902911791...
    """
    have_appended_data = False

    @classmethod
    def rt_process_functions_to_string(cls):
        """
        Return doc string for all predefined real-time processing functions.

        :rtype: str
        :return: String containing doc for all real-time processing functions.
        """
        string = 'Real-time processing functions (use as: ' + \
            'RtTrace.register_rt_process(process_name, [parameter values])):\n'
        for key in REALTIME_PROCESS_FUNCTIONS:
            string += '\n'
            string += '  ' + (str(key) + ' ' + 80 * '-')[:80]
            string += str(REALTIME_PROCESS_FUNCTIONS[key][0].__doc__)
        return string

    def __init__(self, max_length=None, *args, **kwargs):  # @UnusedVariable
        """
        Initializes an RtTrace.

        See :class:`obspy.core.trace.Trace` for all parameters.
        """
        # set window length attribute
        if max_length is not None and max_length <= 0:
            raise ValueError("Input max_length out of bounds: %s" % max_length)
        self.max_length = max_length

        # initialize processing list
        self.processing = []

        # initialize parent Trace with no data or header - all data must be
        # added using append
        super(RtTrace, self).__init__(data=np.array([]), header=None)

    def __eq__(self, other):
        """
        Implements rich comparison of RtTrace objects for "==" operator.

        Traces are the same, if both their data and stats are the same.
        """
        # check if other object is a RtTrace
        if not isinstance(other, RtTrace):
            return False
        return super(RtTrace, self).__eq__(other)

    def __add__(self, **kwargs):  # @UnusedVariable
        """
        Too ambiguous, throw an Error.

        .. seealso:: :meth:`obspy.realtime.rttrace.RtTrace.append`.
        """
        msg = "Too ambiguous for realtime trace data. Try: RtTrace.append()"
        raise NotImplementedError(msg)

    def append(self, trace, gap_overlap_check=False, verbose=False):
        """
        Appends a Trace object to this RtTrace.

        Registered real-time processing will be applied to copy of appended
        Trace object before it is appended.  This RtTrace will be truncated
        from the beginning to RtTrace.max_length, if specified.
        Sampling rate, data type and trace.id of both traces must match.

        :type trace: :class:`~obspy.core.trace.Trace`
        :param trace:  :class:`~obspy.core.trace.Trace` object to append to
            this RtTrace
        :type gap_overlap_check: bool, optional
        :param gap_overlap_check: Action to take when there is a gap or overlap
            between the end of this RtTrace and start of appended Trace:

            * If True, raise TypeError.
            * If False, all trace processing memory will be re-initialized to
              prevent false signal in processed trace.

            (default is ``True``).
        :type verbose: bool, optional
        :param verbose: Print additional information to stdout
        :return: NumPy :class:`~numpy.ndarray` object containing processed
            trace data from appended Trace object.
        """
        if not isinstance(trace, Trace):
            # only add Trace objects
            raise TypeError("Only obspy.core.trace.Trace objects are allowed")

        # sanity checks
        if self.have_appended_data:
            #  check id
            if self.get_id() != trace.get_id():
                raise TypeError("Trace ID differs:", self.get_id(),
                                trace.get_id())
            #  check sample rate
            if self.stats.sampling_rate != trace.stats.sampling_rate:
                raise TypeError("Sampling rate differs:",
                                self.stats.sampling_rate,
                                trace.stats.sampling_rate)
            #  check calibration factor
            if self.stats.calib != trace.stats.calib:
                raise TypeError("Calibration factor differs:",
                                self.stats.calib, trace.stats.calib)
            # check data type
            if self.data.dtype != trace.data.dtype:
                raise TypeError("Data type differs:",
                                self.data.dtype, trace.data.dtype)
        # TODO: IMPORTANT? Should improve check for gaps and overlaps
        # and handle more elegantly
        # check times
        gap_or_overlap = False
        if self.have_appended_data:
            # delta = int(math.floor(\
            #    round((rt.stats.starttime - lt.stats.endtime) * sr, 5) )) - 1
            diff = trace.stats.starttime - self.stats.endtime
            delta = diff * self.stats.sampling_rate - 1.0
            if verbose:
                msg = "%s: Overlap/gap of (%g) samples in data: (%s) (%s) " + \
                    "diff=%gs  dt=%gs"
                print(msg % (self.__class__.__name__,
                             delta, self.stats.endtime, trace.stats.starttime,
                             diff, self.stats.delta))
            if delta < -0.1:
                msg = "Overlap of (%g) samples in data: (%s) (%s) diff=%gs" + \
                    "  dt=%gs"
                msg = msg % (-delta, self.stats.endtime, trace.stats.starttime,
                             diff, self.stats.delta)
                if gap_overlap_check:
                    raise TypeError(msg)
                gap_or_overlap = True
            if delta > 0.1:
                msg = "Gap of (%g) samples in data: (%s) (%s) diff=%gs" + \
                    "  dt=%gs"
                msg = msg % (delta, self.stats.endtime, trace.stats.starttime,
                             diff, self.stats.delta)
                if gap_overlap_check:
                    raise TypeError(msg)
                gap_or_overlap = True
            if gap_or_overlap:
                msg += " - Trace processing memory will be re-initialized."
                warnings.warn(msg, UserWarning)
            else:
                # correct start time to pin absolute trace timing to start of
                # appended trace, this prevents slow drift of nominal trace
                # timing from absolute time when nominal sample rate differs
                # from true sample rate
                self.stats.starttime = \
                    self.stats.starttime + diff - self.stats.delta
                if verbose:
                    print("%s: self.stats.starttime adjusted by: %gs"
                          % (self.__class__.__name__, diff -
                             self.stats.delta))
        # first apply all registered processing to Trace
        for proc in self.processing:
            process_name, options, rtmemory_list = proc
            # if gap or overlap, clear memory
            if gap_or_overlap and rtmemory_list is not None:
                for n in range(len(rtmemory_list)):
                    rtmemory_list[n] = RtMemory()
            # apply processing
            trace = trace.copy()
            dtype = trace.data.dtype
            if hasattr(process_name, '__call__'):
                # check if direct function call
                trace.data = process_name(trace.data, **options)
            else:
                # got predefined function
                func = REALTIME_PROCESS_FUNCTIONS[process_name.lower()][0]
                options['rtmemory_list'] = rtmemory_list
                trace.data = func(trace, **options)
            # assure dtype is not changed
            trace.data = np.require(trace.data, dtype=dtype)
        # if first data, set stats
        if not self.have_appended_data:
            self.data = np.array(trace.data)
            self.stats = Stats(header=trace.stats)
            self.have_appended_data = True
            return trace
        # handle all following data sets
        # fix Trace.__add__ parameters
        # TODO: IMPORTANT? Should check for gaps and overlaps and handle
        # more elegantly
        sum_trace = Trace.__add__(
            self, trace, method=0, interpolation_samples=0,
            fill_value='latest', sanity_checks=True)
        # Trace.__add__ returns new Trace, so update to this RtTrace
        self.data = sum_trace.data
        # left trim if data length exceeds max_length
        if self.max_length is not None:
            max_samples = int(self.max_length * self.stats.sampling_rate + 0.5)
            if np.size(self.data) > max_samples:
                starttime = self.stats.starttime + \
                    (np.size(self.data) - max_samples) / \
                    self.stats.sampling_rate
                self._ltrim(starttime, pad=False, nearest_sample=True,
                            fill_value=None)
        return trace

    def register_rt_process(self, process, **options):
        """
        Adds real-time processing algorithm to processing list of this RtTrace.

        Processing function must be one of:
            %s. % REALTIME_PROCESS_FUNCTIONS.keys()
            or a non-recursive, time-domain NumPy or ObsPy function which takes
            a single array as an argument and returns an array

        :type process: str or callable
        :param process: Specifies which processing function is added,
            e.g. ``"boxcar"`` or ``np.abs``` (functions without brackets).
            See :mod:`obspy.realtime.signal` for all predefined processing
            functions.
        :type options: dict, optional
        :param options: Required keyword arguments to be passed the respective
            processing function, e.g. ``width=100`` for ``'boxcar'`` process.
            See :mod:`obspy.realtime.signal` for all options.
        :rtype: int
        :return: Length of processing list after registering new processing
            function.
        """
        # create process_name either from string or function name
        process_name = ("%s" % process).lower()

        # set processing entry for this process
        entry = False
        rtmemory_list = None
        if hasattr(process, '__call__'):
            # direct function call
            entry = (process, options, None)
        elif process_name in REALTIME_PROCESS_FUNCTIONS:
            # predefined function
            num = REALTIME_PROCESS_FUNCTIONS[process_name][1]
            if num:
                # make sure we have num new RtMemory instances
                rtmemory_list = [RtMemory() for _i in range(num)]
            entry = (process_name, options, rtmemory_list)
        else:
            # check if process name is contained within a predefined function,
            # e.g. 'int' for 'integrate'
            for key in REALTIME_PROCESS_FUNCTIONS:
                if not key.startswith(process_name):
                    continue
                process_name = key
                num = REALTIME_PROCESS_FUNCTIONS[process_name][1]
                if num:
                    # make sure we have num new RtMemory instances
                    rtmemory_list = [RtMemory() for _i in range(num)]
                entry = (process_name, options, rtmemory_list)
                break

        if not entry:
            raise NotImplementedError("Can't register process %s" % (process))

        # add process entry
        self.processing.append(entry)

        # add processing information to the stats dictionary
        proc_info = "realtime_process:%s:%s" % (process_name, options)
        self._internal_add_processing_info(proc_info)

        return len(self.processing)

    def copy(self, *args, **kwargs):
        """
        Returns a deepcopy of this RtTrace.
        """
        # XXX: ugly hack to allow deepcopy of an RtTrace object containing
        # registered NumPy function (numpy.ufunc) calls
        temp = copy.copy(self.processing)
        self.processing = []
        new = copy.deepcopy(self, *args, **kwargs)
        new.processing = temp
        return new


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
