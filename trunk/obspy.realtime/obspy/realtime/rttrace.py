# -*- coding: utf-8 -*-
"""
Module for handling ObsPy RtTrace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core import Trace, Stats
from obspy.realtime.rtmemory import RtMemory
from obspy.realtime.signal import util
import numpy as np


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

    >>> import numpy as np
    >>> from obspy.core import read
    >>> import obspy.realtime.rttrace as rt
    >>> from obspy.realtime.signal.util import *
    >>> import os
    >>>
    >>> # read data file
    >>> data_stream = read(os.path.join(os.path.dirname(rt.__file__), \
            os.path.join('tests', 'data'), 'II.TLY.BHZ.SAC'))
    >>> data_trace = data_stream[0]
    >>> ref_time_offest = data_trace.stats['sac']['a']
    >>> print 'ref_time_offest (sac.a):' + str(ref_time_offest)
    ref_time_offest (sac.a):301.506
    >>> epicentral_distance = data_trace.stats['sac']['gcarc']
    >>> print 'epicentral_distance (sac.gcarc):' + str(epicentral_distance)
    epicentral_distance (sac.gcarc):30.0855
    >>>
    >>> # create set of contiguous packet data in an array of Trace objects
    >>> total_length = np.size(data_trace.data)
    >>> num_pakets = 3
    >>> packet_length = int(total_length / num_pakets)  # may give int truncate
    >>> delta_time = 1.0 / data_trace.stats.sampling_rate
    >>> tstart = data_trace.stats.starttime
    >>> tend = tstart + delta_time * packet_length
    >>> traces = []
    >>> for i in range(num_pakets):
    ...     tr = data_trace.copy()
    ...     tr = tr.slice(tstart, tend)
    ...     traces.append(tr)
    ...     tstart = tend + delta_time
    ...     tend = tstart + delta_time * packet_length
    ...
    >>> # assemble realtime trace
    >>> rt_trace = rt.RtTrace()
    >>> rt_trace.registerRtProcess('integrate')
    1
    >>> rt_trace.registerRtProcess('mwpIntegral', mem_time=240,
    ...     ref_time=(data_trace.stats.starttime + ref_time_offest),
    ...     max_time=120, gain=1.610210e+09)
    2
    >>>
    >>> # append packet data to RtTrace
    >>> for i in range(num_pakets):
    ...     appended_trace = rt_trace.append(traces[i], gap_overlap_check=True)
    ...
    >>> # post processing to get Mwp
    >>> peak = np.amax(np.abs(rt_trace.data))
    >>> print 'mwpIntegral peak = ', peak
    mwpIntegral peak =  0.136404
    >>> print 'epicentral_distance = ', epicentral_distance
    epicentral_distance =  30.0855
    >>> mwp = calculateMwpMag(peak, epicentral_distance)
    >>> print 'Mwp = ', mwp
    Mwp =  8.78902911791
    """

    # dictionary to map given type-strings to processing functions
    # keys must be all lower case
    # values are lists: [function name, number of RtMemory objects]
    rtprocess_functions = {
        'scale': [util.scale, 0],
        'integrate': [util.integrate, 1],
        'differentiate': [util.differentiate, 1],
        'boxcar': [util.boxcar, 1],
        'tauc': [util.tauc, 2],
        'mwpintegral': [util.mwpIntegral, 1],
    }

    max_length = None
    have_appended_data = False

    @classmethod
    def rtProcessFunctionsToString(cls):
        """
        :return: str String containing doc for all realt-time processing
        functions.
        """

        string = 'Real-time processing functions (use as: ' + \
            'RtTrace.registerRtProcess(process_name, [parameter values])):\n'
        for key in RtTrace.rtprocess_functions:
            string += '\n'
            string += '  ' + (str(key) + ' ' + 80 * '-')[:80]
            string += str(RtTrace.rtprocess_functions[key][0].__doc__)
        return(string)

    def __init__(self, data=None, header=None,  # @UnusedVariable
                 max_length=None):
        """
        Initializes an RtTrace.

        See :class:`obspy.core.trace.Trace` for all parameters.
        """
        # set window length attribute
        if max_length != None and max_length <= 0:
            raise ValueError("Input max_length out of bounds: %s" % max_length)
        self.max_length = max_length

        # initialize processing list
        self.processing = []

        # ignore any passed data or header
        data = np.array([])
        header = None

        # initialize parent Trace with no data or header - all data must be
        #   added using __add__
        Trace.__init__(self, data, header)

    def __eq__(self, other):
        """
        Implements rich comparison of RtTrace objects for "==" operator.

        Traces are the same, if both their data and stats are the same.
        """
        #check if other object is a RtTrace
        if not isinstance(other, RtTrace):
            return False
        # call superclass operator
        return Trace.__eq__(self, other)

    def __ne__(self, other):
        """
        Implements rich comparison of Trace objects for "!=" operator.

        Calls __eq__() and returns the opposite.
        """
        return not self.__eq__(other)

    def __str__(self, id_length=None):
        return Trace.__str__(self, id_length)

    def __add__(self, **kwargs):  # @UnusedVariable
        """
        Too ambiguous, throw an Error.

        .. seealso:: :meth:`obsppy.realtime.RtTrace.append`.
        """
        msg = "Too ambiguous for realtime trace data. Try: RtTrace.append()"
        raise NotImplementedError(msg)

    def append(self, trace, gap_overlap_check=False, verbose=False):
        """
        Appends a Trace object to this RtTrace.

        Registered real-time processing will be applied to appended Trace
        object before it is appended.  This RtTrace will be truncated from
        the beginning to RtTrace.max_length, if specified.
        Sampling rate, data type and trace.id of both traces must match.

        :type trace: :class:`~obspy.core.trace.Trace`
        :param trace:  :class:`~obspy.core.trace.Trace` object to append to
            this RtTrace
        :type gap_overlap_check: bool, optional
        :param gap_overlap_check: Action to take when there is a gap or overlap
            between the end of this RtTrace and start of appended Trace:
                If True, raise TypeError.
                If False, all trace processing memory will be re-initialized to
                    prevent false signal in processed trace.
            (default is ``True``).
        :type verbose: bool, optional
        :param verbose: Print additional information to stdout
        :return: NumPy :class:`np.ndarray` object containing processed trace
            data from appended Trace object.
        """
        # make sure datatype is compatible with Trace.__add__() which returns
        #   array of float32
        # convert f4 datatype to float32
        if trace.data.dtype == '>f4' or trace.data.dtype == '<f4':
            trace.data = np.array(trace.data, dtype=np.float32)

        # sanity checks
        if self.have_appended_data:
            if not isinstance(trace, Trace):
                raise TypeError
            #  check id
            if self.getId() != trace.getId():
                raise TypeError("Trace ID differs:", self.getId(),
                                trace.getId())
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
        #   and handle more elegantly
        # check times
        gap_or_overlap = False
        if self.have_appended_data:
            #if self.stats.starttime <= trace.stats.starttime:
            #    lt = self
            #    rt = trace
            #else:
            #    rt = self
            #    lt = trace
            sr = self.stats.sampling_rate
            #delta = int(math.floor(\
            #    round((rt.stats.starttime - lt.stats.endtime) * sr, 5) )) - 1
            diff = trace.stats.starttime - self.stats.endtime
            delta = diff * sr - 1.0
            if verbose:
                msg = "%s: Overlap/gap of (%g) samples in data: (%s) (%s) " + \
                    "diff=%gs  dt=%gs"
                print  msg % (self.__class__.__name__,
                              delta, self.stats.endtime, trace.stats.starttime,
                              diff, 1.0 / sr)
            if delta < -0.1:
                msg = self.__class__.__name__ + ": " \
                "Overlap of (%g) samples in data: (%s) (%s) diff=%gs  dt=%gs" \
                    % (-delta, self.stats.endtime, trace.stats.starttime, \
                       diff, 1.0 / sr)
                if gap_overlap_check:
                    raise TypeError(msg)
                gap_or_overlap = True
            if delta > 0.1:
                msg = self.__class__.__name__ + ": " \
                "Gap of (%g) samples in data: (%s) (%s) diff=%gs  dt=%gs" \
                    % (delta, self.stats.endtime, trace.stats.starttime, \
                       diff, 1.0 / sr)
                if gap_overlap_check:
                    raise TypeError(msg)
                gap_or_overlap = True
            if gap_or_overlap:
                print "Warning: " + msg
                print "   Trace processing memory will be re-initialized."
            else:
                # correct start time to pin absolute trace timing to start of
                # appended trace, this prevents slow drift of nominal trace
                # timing from absolute time when nominal sample rate differs
                # from true sample rate
                self.stats.starttime = self.stats.starttime + diff - 1.0 / sr
                if verbose:
                    print "%s: self.stats.starttime adjusted by: %gs" \
                    % (self.__class__.__name__, diff - 1.0 / sr)

        # first apply all registered processing to Trace
        for proc in self.processing:
            #print 'DEBUG: Applying processing: ', proc
            process_name, options, rtmemory_list = proc
            # if gap or overlap, clear memory
            if gap_or_overlap and rtmemory_list != None:
                for n in range(len(rtmemory_list)):
                    rtmemory_list[n] = RtMemory()
            #print 'DEBUG: Applying processing: ', process_name, ' ', options
            # apply processing
            trace.data = self._rtProcess(trace, process_name, rtmemory_list,
                                         **options)

        # if first data, set stats
        if not self.have_appended_data:
            self.data = np.array(trace.data)
            self.stats = Stats(header=trace.stats)
        else:
            # fix Trace.__add__ parameters
            # TODO: IMPORTANT? Should check for gaps and overlaps and handle
            # more elegantly
            method = 0
            interpolation_samples = 0
            fill_value = 'latest'
            sanity_checks = True
            #print "DEBUG: ->trace.stats.endtime:", trace.stats.endtime
            sum_trace = Trace.__add__(self, trace, method,
                                      interpolation_samples,
                                      fill_value, sanity_checks)
            # Trace.__add__ returns new Trace, so update to this RtTrace
            self.data = sum_trace.data
            # set derived values, including endtime
            self.stats.__setitem__('npts', sum_trace.stats.npts)
            #print "DEBUG: add->self.stats.endtime:", self.stats.endtime

            # left trim if data length exceeds max_length
            #print "DEBUG: max_length:", self.max_length
            if self.max_length != None:
                max_samples = int(self.max_length * \
                                  self.stats.sampling_rate + 0.5)
                #print "DEBUG: max_samples:", max_samples,
                #    " np.size(self.data):", np.size(self.data)
                if np.size(self.data) > max_samples:
                    starttime = self.stats.starttime \
                        + (np.size(self.data) - max_samples) \
                        / self.stats.sampling_rate
                    # print "DEBUG: self.stats.starttime:",
                    #     self.stats.starttime, " new starttime:", starttime
                    self._ltrim(starttime, pad=False, nearest_sample=True,
                                fill_value=None)
                    #print "DEBUG: self.stats.starttime:",
                    #     self.stats.starttime, " np.size(self.data):",
                    #     np.size(self.data)
        self.have_appended_data = True
        return(trace)

    def _rtProcess(self, trace, process, rtmemory_list, **options):
        """
        Runs a real-time processing algorithm on the a given input array trace.

        :type trace: :class:`~obspy.core.trace.Trace`
        :param trace:  :class:`~obspy.core.trace.Trace` object to process
        :type process: str or function
        :param process: Specifies which processing function is applied,
            e.g. ``'boxcar'`` or ``np.abs``` (functions without brackets).
        :type rtmemory_list: list
        :param rtmemory_list: Persistent memory used by process_name on this
            RtTrace.
        :type options: dict, optional
        :param options: Required keyword arguments to be passed the respective
            processing function (e.g. width=100).
        :return: NumPy :class:`np.ndarray` object containing processed trace
            data.
        """
        # check if direct function call
        if hasattr(process, '__call__'):
            return process(trace.data, **options)

        # got function defined within rtprocess_functions dictionary
        process_func = RtTrace.rtprocess_functions[process.lower()][0]
        return process_func(trace, rtmemory_list, **options)

    def registerRtProcess(self, process, **options):
        """
        Adds real-time processing algorithm to processing list of this RtTrace.

        Processing function must be one of:
            %s. % RtTrace.rtprocess_functions.keys()
            or a non-recursive, time-domain np or obspy function which takes
            a single array as an argument and returns an array

        :type process: str or function
        :param process: Specifies which processing function is added,
            e.g. ``'boxcar'`` or ``np.abs``` (functions without brackets).
        :type options: dict, optional
        :param options: Required keyword arguments to be passed the respective
            processing function, e.g. ``width=100``.
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
        elif process_name in RtTrace.rtprocess_functions:
            # predefined function within RtTrace.rtprocess_functions
            num = RtTrace.rtprocess_functions[process_name][1]
            if num:
                rtmemory_list = [RtMemory()] * num
            entry = (process_name, options, rtmemory_list)
        else:
            # check if process name is contained within a rtprocess_function,
            # e.g. 'int' for 'integrate'
            for key in RtTrace.rtprocess_functions:
                if not key.startswith(process_name):
                    continue
                process_name = key
                num = RtTrace.rtprocess_functions[process_name][1]
                if num:
                    rtmemory_list = [RtMemory()] * num
                entry = (process_name, options, rtmemory_list)
                break

        if not entry:
            raise NotImplementedError("Can't register process %s" % (process))

        # add process entry
        self.processing.append(entry)

        # add processing information to the stats dictionary
        proc_info = "realtime_process:%s:%s" % (process_name, options)
        self._addProcessingInfo(proc_info)

        return len(self.processing)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
