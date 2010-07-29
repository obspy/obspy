# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: batch.py
#  Purpose: Provides single routine to calculate HVSR.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from obspy.core import read

from htov import *

def create_HVSR(filename, vertical_channel=None, spectra_method='multitaper',
                spectra_options={'time_bandwidth':3.5, 'number_of_tapers':
                None, 'quadratic':False, 'adaptive':True,
                'nfft':None},
                master_curve_method='mean', lowpass_value=None,
                highpass_value=None, new_sample_rate=None, zerophase=False,
                corners=4, starttime=None, endtime=None, threshold=0.95,
                window_length=25, cutoff_value=0.1, zdetector_window_length=40,
                message_function=None):
    """
    Single function that takes lot of parameters and creates an HVSR spectrum.

    :param filename: string
        Filename that can be read with obspy. Needs to contain three channel
        stream with the necessary data. The Stream object needs to contain
        exactly three Traces and one channel needs to be a vertical channel.
    :param vertical_channel: Determines which channel is to be the vertical
        channel. If None it will be determined automatically.
    :param spectra_method: string
        Method with which the single spectra are created. Available are
        'multitaper', 'sine multitaper', and 'single taper'
    :param spectra_options: dictionary
        Contains the parameters for the chosen method. Available are:
        - for multitaper: 'time_bandwidth', 'number_of_tapers', 'quadratic',
              'adaptive', 'nfft
        - for sine multitaper: 'number_of_tapers', 'degree_of_smoothing',
              'number_of_iterations'
        - for single taper: 'taper'
        Every necessary options needs to be available or otherwise it will
        result in an error.
    :param master_curve_method: string
        How to determine the master curve. Available are 'mean', 'geometric
        average', and 'median'
    :param lowpass_value: float
        Lowpass filter value. Will not apply lowpass filter if None is given.
    :param highpass_value: float
        Highpass filter value. Will not apply highpass filter if None is given.
    :param new_sample_rate: float
        New sample rate. The original data will be resampled in case 
    :param zerophase: bool
        Whether or not to use a zero-phase Butterworth filter.
    :param corners: int
        Number of corners for the Butterworth filter.
    :param starttime: UTCDateTime, float
        Determines where to cut the beginning of the Stream object. If it is a
        float it will be considered as the offset in seconds.
    :param endtime: UTCDateTime, float
        Determines where to cut the end of the Stream object. If it is a float
        it will be considered as the offset in seconds.
    :param threshold: float
        Threshold for the characteristic function to find the quiet areas.
        Everything under this value will be considered quiet. If it is between
        0.00 and 1.00 it will be treated as a percentile value which is the
        recommended choise. The percentile will be applied to each single trace
        seperately.
    :param window_length: float
        Window length for the single HVSR windows in seconds.
    :param message_function: Python function
        If given, a string will be passed to this function to document the
        current progress.
    :param zdetector_window_length: int
        Window length for the zdetector.
    :param cutoff_value: float
        If given, than this value determines which part of the bottom and top
        frequencies are discarded for the calculation of the master HVSR curve.
        e.g. a value of 0.1 will throw away the bottom and top 10% for each
        frequency.
    """
    if type(filename) == str:
        # Read the file
        stream = read(filename)
    else:
        stream = filename
    # Check if exactly three traces are in the Stream object.
    if len(stream) != 3:
        msg = 'The file needs to contain exactly three channels.'
        raise Exception(msg)
    # Autodetect the orientation of the Traces.
    if vertical_channel is None:
        detectTraceOrientation(stream)
    # Or determine it using the vertical_channel parameter.
    else:
        vert_count = 0
        for trace in stream:
            if trace.stats.channel == vertical_channel.strip():
                trace.stats.orientation = 'vertical'
                vert_count += 1
            else:
                trace.stats.orientation = 'horizontal'
        # Check if exactly on vertical trace has been found.
        if vert_count != 1:
            msg = 'Ambiguous vertical_channel attribute. Please use a ' + \
                  'unique channel name or rename the channels.'
            raise Exception(msg)
    # If the times are not UTCDateTime objects they are considered to be
    # offsets.
    if starttime is None:
        starttime = stream[0].stats.starttime
    elif type(starttime) is not UTCDateTime:
        starttime = stream[0].stats.starttime + starttime
    if endtime is None:
        endtime = stream[0].stats.endtime
    elif type(endtime) is not UTCDateTime:
        endtime = stream[0].stats.endtime - endtime
    if new_sample_rate is None:
        new_sample_rate = stream[0].stats.sampling_rate
    sampling_rate = stream[0].stats.sampling_rate
    if lowpass_value >= 0.5 * sampling_rate - 1:
        lowpass_value = None
    if highpass_value == 0.0:
        highpass_value = 0.0
    # Resample, filter and cut the Traces.
    resampleFilterAndCutTraces(stream, new_sample_rate, lowpass_value,
                               highpass_value, zerophase, corners, starttime,
                               endtime, message_function=message_function)
    # Convert the window length to samples.
    window_length = int(window_length * stream[0].stats.sampling_rate)
    # Calculate the characteristic noise Function.
    charNoiseFunctions, thresholds = \
            calculateCharacteristicNoiseFunction(stream, threshold,
            zdetector_window_length, message_function=message_function)
    npts = stream[0].stats.npts
    # Find the quiet intervals in each Trace.
    intervals, _, _ = getQuietIntervals(charNoiseFunctions, thresholds,
                                        window_length, npts)
    if not len(intervals):
        return False
    hvsr_matrix, hvsr_freq, length, master_curve, error = \
            calculateHVSR(stream, intervals, window_length, spectra_method,
                          spectra_options, master_curve_method, cutoff_value,
                          message_function=message_function)
    return master_curve, hvsr_freq, error
