# -*- coding: utf-8 -*-
"""
Module for handling ObsPy RtTrace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import math

from copy import deepcopy
import numpy as np
import obspy.core.trace as octrace
import obspy.realtime.rtmemory as rtmem
import obspy.realtime.signal.util



class RtTrace(octrace.Trace):
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
    #global rtprocess_functions
    rtprocess_functions = {
        'scale': [obspy.realtime.signal.util.scale, 0],
        'integrate': [obspy.realtime.signal.util.integrate, 1],
        'differentiate': [obspy.realtime.signal.util.differentiate, 1],
        'boxcar': [obspy.realtime.signal.util.boxcar, 1],
        'tauc': [obspy.realtime.signal.util.tauc, 2],
        'mwpintegral': [obspy.realtime.signal.util.mwpIntegral, 1],
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


    def __init__(self, data=None, header=None, max_length=None):

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
        octrace.Trace.__init__(self, data, header)


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
        return octrace.Trace.__str__(self, id_length)


    def __add__(self, trace, method=0, interpolation_samples=0,
                fill_value='latest', sanity_checks=True):
        """
        Too ambiguous, throw an Error.
        See also: :meth:`RtTrace.append`.

        """
        raise NotImplementedError("Too ambiguous for realtime trace data, " + \
                                  "therefore not implemented.  Try: RtTrace.append()")

    #  TODO: temporary use of modified version of Trace.__add__ to correct bug
    #   in delta calcualtion (20111209, ObsPy version: 0.4.8.dev-r2921)
    def __addTrace__(self, trace, method=0, interpolation_samples=0,
                     fill_value=None, sanity_checks=True):
        """
        Adds another Trace object to current trace.

        :type method: ``0`` or ``1``, optional
        :param method: Method to handle overlaps of traces. Defaults to ``0``.
            See the table given in the notes section below for further details.
        :type fill_value: int, float or ``'latest'``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present. If the keyword ``'latest'`` is provided it will
            use the latest value before the gap. If keyword ``'interpolate'``
            is provided, missing values are linearly interpolated (not
            changing the data type e.g. of integer valued traces).
        :type interpolation_samples: int, optional
        :param interpolation_samples: Used only for ``method=1``. It specifies
            the number of samples which are used to interpolate between
            overlapping traces. Defaults to ``0``. If set to ``-1`` all
            overlapping samples are interpolated.
        :type sanity_checks: boolean, optional
        :param sanity_checks: Enables some sanity checks before merging traces.
            Defaults to ``True``.

        Trace data will be converted into a NumPy masked array data type if
        any gaps are present. This behavior may be prevented by setting the
        ``fill_value`` parameter. The ``method`` argument controls the
        handling of overlapping data values.

        Sampling rate, data type and trace.id of both traces must match.

        .. rubric:: Notes

        ======  ===============================================================
        Method  Description
        ======  ===============================================================
        0       Discard overlapping data. Overlaps are essentially treated the
                same way as gaps::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAA----FFFF

                Contained traces with differing data will be marked as gap::

                    Trace 1: AAAAAAAAAAAA
                    Trace 2:     FF
                    1 + 2  : AAAA--AAAAAA
        1       Discard data of the previous trace assuming the following trace
                contains data with a more correct time value. The parameter
                ``interpolation_samples`` specifies the number of samples used
                to linearly interpolate between the two traces in order to
                prevent steps. Note that if there are gaps inside, the
                returned array is still a masked array, only if fill_value
                is set, the returned array is a normal array and gaps are
                filled with fill value.

                No interpolation (``interpolation_samples=0``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAAFFFFFFFF

                Interpolate first two samples (``interpolation_samples=2``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAACDFFFFFF (interpolation_samples=2)

                Interpolate all samples (``interpolation_samples=-1``)::

                    Trace 1: AAAAAAAA
                    Trace 2:     FFFFFFFF
                    1 + 2  : AAAABCDEFFFF

                Any contained traces with different data will be discarded::

                    Trace 1: AAAAAAAAAAAA (contained trace)
                    Trace 2:     FF
                    1 + 2  : AAAAAAAAAAAA

                Traces with gaps::

                    Trace 1: AAAA
                    Trace 2:         FFFF
                    1 + 2  : AAAA----FFFF

                Traces with gaps and given ``fill_value=0``::

                    Trace 1: AAAA
                    Trace 2:         FFFF
                    1 + 2  : AAAA0000FFFF

                Traces with gaps and given ``fill_value='latest'``::

                    Trace 1: ABCD
                    Trace 2:         FFFF
                    1 + 2  : ABCDDDDDFFFF

                Traces with gaps and given ``fill_value='interpolate'``::

                    Trace 1: AAAA
                    Trace 2:         FFFF
                    1 + 2  : AAAABCDEFFFF
        ======  ===============================================================
        """
        if sanity_checks:
            if not isinstance(trace, octrace.Trace):
                raise TypeError
            #  check id
            if self.getId() != trace.getId():
                raise TypeError("Trace ID differs")
            #  check sample rate
            if self.stats.sampling_rate != trace.stats.sampling_rate:
                raise TypeError("Sampling rate differs")
            #  check calibration factor
            if self.stats.calib != trace.stats.calib:
                raise TypeError("Calibration factor differs")
            # check data type
            if self.data.dtype != trace.data.dtype:
                raise TypeError("Data type differs")
        # check times
        if self.stats.starttime <= trace.stats.starttime:
            lt = self
            rt = trace
        else:
            rt = self
            lt = trace
        # check whether to use the latest value to fill a gap
        if fill_value == "latest":
            fill_value = lt.data[-1]
        elif fill_value == "interpolate":
            fill_value = (lt.data[-1], rt.data[0])
        sr = self.stats.sampling_rate
        # original delta algorithm
        #delta = int(math.floor(round((rt.stats.starttime - \
        #                              lt.stats.endtime) * sr, 7))) - 1
        diff = trace.stats.starttime - self.stats.endtime
        if diff > 0.0:
            delta = int(0.5 + diff * sr) - 1
        else:
            delta = -int(0.5 + -diff * sr) - 1

        delta_endtime = lt.stats.endtime - rt.stats.endtime
        # create the returned trace
        out = self.__class__(header=deepcopy(lt.stats))
        # check if overlap or gap
        if delta < 0 and delta_endtime < 0:
            # overlap
            delta = abs(delta)
            if np.all(np.equal(lt.data[-delta:], rt.data[:delta])):
                # check if data are the same
                data = [lt.data[:-delta], rt.data]
            elif method == 0:
                overlap = createEmptyDataChunk(delta, lt.data.dtype,
                                               fill_value)
                data = [lt.data[:-delta], overlap, rt.data[delta:]]
            elif method == 1 and interpolation_samples >= -1:
                try:
                    ls = lt.data[-delta - 1]
                except:
                    ls = lt.data[0]
                if interpolation_samples == -1:
                    interpolation_samples = delta
                elif interpolation_samples > delta:
                    interpolation_samples = delta
                try:
                    rs = rt.data[interpolation_samples]
                except IndexError:
                    # contained trace
                    data = [lt.data]
                else:
                    # include left and right sample (delta + 2)
                    interpolation = np.linspace(ls, rs,
                                                interpolation_samples + 2)
                    # cut ls and rs and ensure correct data type
                    interpolation = np.require(interpolation[1:-1],
                                               lt.data.dtype)
                    data = [lt.data[:-delta], interpolation,
                        rt.data[interpolation_samples:]]
            else:
                raise NotImplementedError
        elif delta < 0 and delta_endtime >= 0:
            # contained trace
            delta = abs(delta)
            lenrt = len(rt)
            t1 = len(lt) - delta
            t2 = t1 + lenrt
            if np.all(lt.data[t1:t2] == rt.data):
                # check if data are the same
                data = [lt.data]
            elif method == 0:
                gap = createEmptyDataChunk(lenrt, lt.data.dtype, fill_value)
                data = [lt.data[:t1], gap, lt.data[t2:]]
            elif method == 1:
                data = [lt.data]
            else:
                raise NotImplementedError
        elif delta == 0:
            data = [lt.data, rt.data]
        else:
            # gap
            # use fixed value or interpolate in between
            gap = createEmptyDataChunk(delta, lt.data.dtype, fill_value)
            data = [lt.data, gap, rt.data]
        # merge traces depending on numpy array type
        if True in [isinstance(_i, np.ma.masked_array) for _i in data]:
            data = np.ma.concatenate(data)
        else:
            data = np.concatenate(data)
        out.data = data
        return out


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

        # make sure dataype is compatible with Trace.__add__() which returns
        #   array of float32
        # convert f4 datatype to float32
        if trace.data.dtype == '>f4' or trace.data.dtype == '<f4':
            trace.data = np.array(trace.data, dtype=np.float32)

        # sanity checks
        if self.have_appended_data:
            if not isinstance(trace, octrace.Trace):
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
            #            round((rt.stats.starttime - lt.stats.endtime) * sr, 5) \
            #            )) - 1
            diff = trace.stats.starttime - self.stats.endtime
            delta = diff * sr - 1.0
            if verbose:
                print "%s: Overlap/gap of (%g) samples in data: (%s) (%s) diff=%gs  dt=%gs" \
                    % (self.__class__.__name__,
                       delta, self.stats.endtime, trace.stats.starttime, \
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
                #   appended trace,
                #   this prevents slow drift of nominal trace timing from absolute
                #   time when nominal sample rate differs from true sample rate
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
                    rtmemory_list[n] = rtmem.RtMemory()
            #print 'DEBUG: Applying processing: ', process_name, ' ', options
            # apply processing
            trace.data = self._rtProcess(trace, process_name, rtmemory_list,
                                         ** options)

        # if first data, set stats
        if not self.have_appended_data:
            self.data = np.array(trace.data)
            self.stats = octrace.Stats(header=trace.stats)
        else:
            # fix Trace.__add__ parameters
            # TODO: IMPORTANT? Should check for gaps and overlaps and handle more elegantly
            method = 0
            interpolation_samples = 0
            fill_value = 'latest'
            sanity_checks = True
            #print "DEBUG: ->trace.stats.endtime:", trace.stats.endtime
            #sum_trace = octrace.Trace.__add__(self, trace, method,
            #                                  interpolation_samples,
            #                                  fill_value, sanity_checks)
            #  TODO: temporary use of modified version of Trace.__add__ to correct bug
            #   in delta calcualtion (20111209, ObsPy version: 0.4.8.dev-r2921)
            sum_trace = self.__addTrace__(trace, method,
                                          interpolation_samples,
                                          fill_value, sanity_checks)
            #print 'DEBUG: sum_trace.data.dtype: ', sum_trace.data.dtype
            # Trace.__add__ returns new Trace, so update to this RtTrace
            self.data = sum_trace.data
            # set derived values, including endtime
            self.stats.__setitem__('npts', sum_trace.stats.npts)               
            #print "DEBUG: add->self.stats.endtime:", self.stats.endtime
 

            # left trim if data length exceeds max_length
            #print "DEBUG: max_length:", self.max_length
            if self.max_length != None:
                max_samples = int(self.max_length * self.stats.sampling_rate + 0.5)
                #print "DEBUG: max_samples:", max_samples, \
                #    " np.size(self.data):", np.size(self.data)
                if np.size(self.data) > max_samples:
                    starttime = self.stats.starttime \
                        + (np.size(self.data) - max_samples) \
                        / self.stats.sampling_rate
                    #print "DEBUG: self.stats.starttime:", self.stats.starttime, \
                    #    " new starttime:", starttime
                    self._ltrim(starttime, pad=False, nearest_sample=True,
                                fill_value=None)
                    #print "DEBUG: self.stats.starttime:", self.stats.starttime, \
                    #    " np.size(self.data):", np.size(self.data)
        self.have_appended_data = True

        return(trace)


    def _rtProcess(self, trace, process_name, rtmemory_list, ** options):

        """
        Runs a real-time processing algorithm on the a given input array trace.

        This is performed in place on the actual data array. The original data
        is not accessible anymore afterwards.
        To keep your original data, use :meth:`~obspy.core.trace.Trace.copy`
        to make a copy of your trace.
        This also makes an entry with information on the applied processing
        in ``trace.stats.processing``.
        For details see :mod:`obspy.realtime`.

        :type trace: :class:`~obspy.core.trace.Trace`
        :param trace:  :class:`~obspy.core.trace.Trace` object to process
        :type process_name: str
        :param process_name: Specifies which processing function is applied
            (e.g. 'boxcar').
        :type rtmemory_list: list
        :param rtmemory_list: Persistent memory used by process_name on this
            RtTrace.
        :type options: dict, optional
        :param options: Required keyword arguments to be passed the respective
            processing function (e.g. width=100).
        :return: NumPy :class:`np.ndarray` object containing processed trace
            data.

        """

        # make process_name string comparison case insensitive
        process_name = process_name.lower()

        # do the actual processing. the options dictionary is passed as
        # kwargs to the function that is mapped according to the
        #   rtprocess_functions dictionary.
        if process_name not in RtTrace.rtprocess_functions:
            # assume standard obspy or np data function
            #print 'DEBUG: eval: ', process_name + '(trace.data, ** options)'
            #print 'DEBUG:   --> ', eval(process_name + '(trace.data, ** options)')
            data = eval(process_name + '(trace.data, ** options)')
        else:
            data = RtTrace.rtprocess_functions[
                process_name][0](trace, rtmemory_list, ** options)

        return data



    def registerRtProcess(self, process_name, ** options):
        """
        Adds a real-time processing algorithm to the processing list of this
            RtTrace.

        Processing function must be one of:
            %s. % RtTrace.rtprocess_functions.keys()
            or a non-recursive, time-domain np or obspy function which takes
            a single array as an argument and returns an array

        :type process_name: str
        :param process_name: Specifies which processing function is applied
            (e.g. 'boxcar').
        :type options: dict, optional
        :param options: Required keyword arguments to be passed the respective
            processing function
                (e.g. width=100).
        :return: int Length of processing list after registering new processing
            function.

        """

        # make process_name string comparison case insensitive
        process_name = process_name.lower()

        # add processing information to the stats dictionary
        if 'processing' not in self.stats:
            self.stats['processing'] = []
        proc_info = "realtime_process:%s:%s" % (process_name, options)
        self.stats['processing'].append(proc_info)

        #print 'DEBUG: Appending processing: ', process_name, ' ', options

        # set processing entry for this process
        entry = None
        # check if process in in defined RtTrace.rtprocess_functions
        if process_name not in RtTrace.rtprocess_functions:
            # check if process name is prefix to a defined rtprocess_function
            for key in RtTrace.rtprocess_functions:
                if key.startswith(process_name):
                    process_name = key
                    break
            entry = (process_name, options, None)
        if process_name in RtTrace.rtprocess_functions:
            num_mem = RtTrace.rtprocess_functions[process_name][1]
            if num_mem < 1:
                rtmemory_list = None
            else:
                rtmemory_list = []
            for i in range(num_mem):
                rtmemory_list = rtmemory_list + [rtmem.RtMemory()]
            entry = (process_name, options, rtmemory_list)
        # process not found in defined RtTrace.rtprocess_functions, 
        #   assume obspy or np function
        if entry is None:
            entry = (process_name, options, None)
        self.processing.append(entry)
        name, opt, mem = self.processing[len(self.processing)-1]
        #print 'DEBUG: Appended processing: ', name, ' ', opt

        return len(self.processing)



if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
