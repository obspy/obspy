# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: htov.py
#  Purpose: Routines for calculating HVSR.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from copy import deepcopy
from math import ceil
from mtspec import mtspec, sine_psd
import numpy as np
from obspy.core.util import scoreatpercentile as quantile
from obspy.signal.filter import highpass, lowpass, bandpass
from obspy.signal.trigger import zdetect
from scipy.signal import resample

from utils import *
from konno_ohmachi_smoothing import calculate_smoothing_matrix


def resampleFilterAndCutTraces(stream, resampling_rate, lowpass_value,
                               highpass_value, zerophase, corners, starttime,
                               endtime, message_function=None):
    """
    Resamples, filters and cuts all Traces in a Stream object.
    
    It will always apply each operation to every trace in the order described
    above.

    :param stream: obspy.core.stream object
        Will be altered and has to contain at least one Trace.
    :param resampling_rate: float
        Desired new sample rate.
    :param lowpass_value: float
        High filter frequency.
    :param highpass_value: float
        Low filter frequency.
    :param zerophase: bool
        Whether or not to use a zerophase filter.
    :param corners: int
        Number of corners for the used Butterworth-Filter.
    :param starttime: obspy.core.UTCDateTime
        New starttime of each Trace.
    :param endtime: obspy.core.UTCDateTime
        New endtime of each Trace.
    :param message_function: Python function
        If given, a string will be passed to this function to document the
        current progress.
    """
    # Convert to floats for more exact handling. Also level the data.
    for trace in stream:
        trace.data = np.require(trace.data, 'float32')
        trace.data -= np.linspace(trace.data[0], trace.data[-1], len(trace.data))
    # The first step is to resample the data. This is done before trimming
    # so that any boundary effects that might occur can be cut away later
    # on.
    if resampling_rate != stream[0].stats.sampling_rate:
        time_range = stream[0].stats.endtime - \
                     stream[0].stats.starttime
        new_npts = time_range / \
                   (1 / resampling_rate) + 1
        new_freq = 1.0 / (time_range / float(new_npts - 1))
        for _i, trace in enumerate(stream):
            if message_function:
                msg = 'Resampling traces to %.2f Hz [%i/%i]...' % \
                        (resampling_rate, _i + 1, len(stream))
                message_function(msg)
            # Use scipy to resample the traces.
            trace.data = resample(trace.data, new_npts, window='hamming')
            trace.stats.sampling_rate = new_freq
    # Filter the trace. Differentiate between low-, high-, and bandpass
    if lowpass_value and highpass_value:
        if message_function:
            msg = 'Bandpass filtering traces from %.2f Hz to %.2f Hz...' % \
                    (highpass_value, highpass_value)
            message_function(msg)
        for trace in stream:
            trace.data = bandpass(trace.data, highpass_value,
                                  lowpass_value, trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    elif lowpass_value:
        if message_function:
            msg = 'Lowpass filtering traces with %.2f Hz...' % lowpass_value
            message_function(msg)
        for trace in stream:
            trace.data = lowpass(trace.data, lowpass_value,
                                  trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    elif highpass_value:
        if message_function:
            msg = 'Highpass filtering traces with %.2f Hz...' % highpass_value
            message_function(msg)
        for trace in stream:
            trace.data = highpass(trace.data, highpass_value,
                                  trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    # Trim the trace if it is necessary.
    if message_function:
        message_function('Trimming traces...')
    stream.trim(starttime, endtime)

def calculateCharacteristicNoiseFunction(stream, threshold, window_length,
                                         message_function=None):
    """
    Calculates a characteristic function for the noise and threshold.

    Uses the z-detector to do this. Returns a list with the characteristic
    functions and a second list with the thresholds.

    :param stream: obspy.core.Stream
    :param threshold: Percentile value, e.g 0.95 mean the 95% percentile.
    :param window_length: Window length passed to the z-detector.
    :param message_function: Python function
        If given, a string will be passed to this function to document the
        current progress.
    """
    charNoiseFunctions = []
    if message_function:
        message_function('Calculating characteristic noise function...')
    # Get the characteristic function for each Trace.
    for trace in stream:
        charNoiseFunctions.append(zdetect(trace.data, window_length))
    lengths = [len(tr.data) for tr in stream]
    if message_function:
        message_function('Applying threshold...')
    # Calculate the thresholds from the percentage values.
    thresholds = []
    for data in charNoiseFunctions:
        length = len(data)
        s_data = np.sort(data)
        thresholds.append(s_data[threshold * length])
    return charNoiseFunctions, thresholds


def getQuietIntervals(charNoiseFunctions, thresholds, window_length, npts):
    """
    Parses the characteristic Noise Function and find the areas that are under
    the threshold and at least have a length of window length and they also all
    have to be in all Traces.

    :param charNoiseFunction: list
        Contains numpy.ndarrays with characteristic functions.
    :param thresholds: list
        Contains the threshold value for each characteristic function.
    :param window_length: int
        Resulting window length.
    :param npts: int
        Number of samples for one original Trace. Needed because the
        characteristic function might contain less or more samples then the
        Trace.

    Returns three things:
        1. A numpy.ndarray containing the interval start- and endsamples in
        samples.
        2. A list containing the quiet areas for each Trace.
        3. A list containing the common quiet areas for each Trace.
    """
    # Find the areas within the treshold.
    quiet_areas = []
    for _i, data in enumerate(charNoiseFunctions):
        quiet_areas.append(getAreasWithinThreshold(data,
                thresholds[_i], window_length, 0))

    # Find the common quiet areas.
    common_quiet_areas = findCommonQuietAreas(quiet_areas,
                              npts,
                              window_length)

    # Find the intervals in the areas.
    intervals = np.array(getIntervalsInAreas(common_quiet_areas,
                                         min_interval=window_length))
    return intervals, quiet_areas, common_quiet_areas

def findCommonQuietAreas(areas, length, min_length):
    """
    areas is a list with arrays. Each array contains quiet areas as a tuple
    of two samples which represent the start and the end of the quiet
    areas.

    This function returns one array with the areas that are common to each
    array in the areas list.

    Length is an integer to avoid having to calulate the maximum number in
    each sample.
    """
    common_quiet = np.zeros(length)
    # Loop over each area and over each zone. Write every quiet zone to the
    # same array. At the end all quiet zone will have a value of zero and
    # every not quiet zone a value of one.
    for area in areas:
        position = 0
        for start, end in area:
            common_quiet[position:start] = 1
            position = end + 1
        # Do the end seperately
        common_quiet[position:length + 1] = 1
    # Now again create tuples.
    # XXX: Search for faster way of doing this.
    common_quiet_times = getAreasWithinThreshold(common_quiet, 0.5,
                                                 min_length)
    return common_quiet_times

def calculateHVSR(stream, intervals, window_length, method, options,
                  master_method, cutoff_value, smoothing=None,
                  smoothing_count=1, smoothing_constant=40,
                  message_function=None):
    """
    Calculates the HVSR curve.
    """
    # Some arithmetics.
    length = len(intervals)
    good_length = window_length // 2 + 1
    # Create the matrix that will be used to store the single spectra.
    hvsr_matrix = np.empty((length, good_length))
    # The stream that will be used.
    # XXX: Add option to use the raw data stream.
    if method == 'multitaper':
        if options['nfft']:
            good_length = options['nfft']// 2 + 1
            # Create the matrix that will be used to store the single
            # spectra.
            hvsr_matrix = np.empty((length, good_length))
        # Loop over each interval
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            # Figure out which traces are vertical and which are horizontal.
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = mtspec(v, stream[0].stats.delta,
                                    options['time_bandwidth'],
                                    nfft=options['nfft'],
                                    number_of_tapers=options['number_of_tapers'],
                                    quadratic=options['quadratic'],
                                    adaptive=options['adaptive'])
            h1_spec, h1_freq = mtspec(h1, stream[0].stats.delta,
                                      options['time_bandwidth'], nfft=options['nfft'],
                                      number_of_tapers=options['number_of_tapers'],
                                      quadratic=options['quadratic'],
                                      adaptive=options['adaptive'])
            h2_spec, h2_freq = mtspec(h2, stream[0].stats.delta,
                                      options['time_bandwidth'],
                                      nfft=options['nfft'],
                                      number_of_tapers=options['number_of_tapers'],
                                      quadratic=options['quadratic'],
                                      adaptive=options['adaptive'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(h1_spec * h2_spec) / v_spec
            if _i == 0:
                good_freq = v_freq
            hvsr_matrix[_i, :] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    elif method == 'sine multitaper':
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            # Figure out which traces are vertical and which are horizontal.
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = sine_psd(v, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            h1_spec, h1_freq = sine_psd(h1, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            h2_spec, h2_freq = sine_psd(h2, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(h1_spec * h2_spec) / v_spec
            if _i == 0:
                good_freq = v_freq
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[_i,:] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    # Use a single taper spectrum with different available tapers.
    elif method == 'single taper':
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = single_taper_spectrum(v,
                                    stream[0].stats.delta, options['taper'])
            h1_spec, h1_freq = single_taper_spectrum(h1,
                                    stream[0].stats.delta, options['taper'])
            h2_spec, h2_freq = single_taper_spectrum(h2,
                                    stream[0].stats.delta, options['taper'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(h1_spec * h2_spec) / v_spec
            if _i == 0:
                good_freq = v_freq
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[_i, :] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    # Should never happen.
    else:
        msg = 'Something went wrong.'
        raise Exception(msg)
    # Copy once to be able to calculate standard deviations.
    original_matrix = deepcopy(hvsr_matrix)
    # Sort it for quantile operations.
    hvsr_matrix.sort(axis=0)
    # Only senseful for mean calculations. Omitted for the median.
    if cutoff_value != 0.0 and master_method != 'median':
        hvsr_matrix = hvsr_matrix[int(length * cutoff_value):
                              ceil(length * (1 - cutoff_value)), :]
    length = len(hvsr_matrix)
    # Mean.
    if master_method == 'mean':
        master_curve = hvsr_matrix.mean(axis=0)
    # Geometric average.
    elif master_method == 'geometric average':
        master_curve = hvsr_matrix.prod(axis=0) ** (1.0 / length)
    # Median.
    elif master_method == 'median':
        # Use another method because interpolation might be necessary.
        master_curve = np.empty(len(hvsr_matrix[0, :]))
        error = np.empty((len(master_curve), 2))
        for _i in xrange(len(master_curve)):
            cur_row = hvsr_matrix[:, _i]
            master_curve[_i] = quantile(cur_row, 50)
            error[_i, 0] = quantile(cur_row, 25)
            error[_i, 1] = quantile(cur_row, 75)
    # Calculate the standard deviation for the two mean methods.
    if master_method != 'median':
        error = np.empty((len(master_curve), 2))
        std = (hvsr_matrix[:][:] - master_curve) ** 2
        std = std.sum(axis=0)
        std /= float(length)
        std **= 0.5
        error[:, 0] = master_curve - std
        error[:, 1] = master_curve + std
    return original_matrix, good_freq, length, master_curve, error

def detectTraceOrientation(stream):
    """
    Detects the orientation of each Trace in a Stream object simply based
    on it ending with z.

    If that does not work any 'z' in the channel attribute will do or else
    the last trace will be the vertical trace.
    """
    for trace in stream:
        if trace.stats.channel.lower().endswith('z'):
            trace.stats.orientation = 'vertical'
        else:
            trace.stats.orientation = 'horizontal'
    # Check if it worked. Only one vertical component should be available.
    check = [True for trace in stream if trace.stats.orientation=='vertical']
    if len(check) == 1:
        return
    # Try matchmaking based on any z in it.
    for trace in stream:
        if 'z' in trace.stats.channel.lower():
            trace.stats.orientation = 'vertical'
        else:
            trace.stats.orientation = 'horizontal'
    # Check if it worked. Only one vertical component should be available.
    check = [True for trace in stream if trace.stats.orientation=='vertical']
    if len(check) == 1:
        return
    # The last one will be the vertical component.
    for trace in stream:
        trace.stats.orientation == 'horizontal'
    stream[-1].stats.oriantatio = 'vertical'

