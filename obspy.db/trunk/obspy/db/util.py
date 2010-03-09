# -*- coding: utf-8 -*-
from obspy.core import Trace
from pkg_resources import iter_entry_points
import numpy as np


def _getInstalledWaveformFeaturesPlugins():
    """
    Gets a dictionary of all available waveform features plug-ins.
    """
    features = {}
    for ep in iter_entry_points(group='obspy.db.feature'):
        features[ep.name] = ep
    return features


def createPreview(trace, delta=60.0):
    """
    Creates a preview trace.
    """
    data = trace.data
    # number of samples for a DELTA second slice
    number_of_samples = delta * trace.stats.sampling_rate
    # number of 60 second slices with data
    number_of_slices = int(len(data) / number_of_samples)
    # minimum and maximum of remaining samples
    npts = number_of_samples * number_of_slices
    if data[npts:].size:
        last_min = data[npts:].min()
        last_max = data[npts:].max()
        last_diff = [last_max - last_min]
    else:
        last_diff = []
    # Fill NaN value with -1.
    if np.isnan(last_diff):
        last_diff = -1
    # reshape matrix
    data = trace.data[0:npts].reshape([number_of_slices, number_of_samples])
    # get minimum and maximum for each row
    diff = data.ptp(axis=1)
    # fill NaN with -1 -> means missing data
    data = np.ma.filled(diff, -1)
    # append value of last diff
    data = np.concatenate([data, last_diff])
    tr = Trace(data=data, header=trace.stats)
    tr.stats.delta = delta
    tr.stats.npts = len(data)
    return tr
