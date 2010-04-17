# -*- coding: utf-8 -*-

from obspy.core import Trace, UTCDateTime
import numpy as np


def createPreview(trace, delta=60.0):
    """
    Creates a preview trace.
    
    A preview trace consists of maxima minus minima of all samples within
    ``delta`` seconds. The parameter ``delta`` must be a multiple of the
    sampling rate of the ``trace`` object.
    """
    data = trace.data
    start_time = trace.stats.starttime.timestamp
    # number of samples for a single slice of delta seconds
    samples_per_slice = delta * trace.stats.sampling_rate
    # minimum and maximum of samples before a static time marker
    start = (delta - start_time % delta) * trace.stats.sampling_rate
    start_time = start_time - start_time % delta
    if start > (delta / 2) and data[0:start].size:
        first_diff = [data[0:start].max() - data[0:start].min()]
    else:
        # skip starting samples
        first_diff = []
        start_time += delta
    # number of complete slices of data
    number_of_slices = int((len(data) - start) / samples_per_slice)
    # minimum and maximum of remaining samples
    end = samples_per_slice * number_of_slices + start
    if end > (delta / 2) and data[end:].size:
        last_diff = [data[end:].max() - data[end:].min()]
    else:
        # skip tailing samples
        last_diff = []
    # Fill NaN value with -1.
    if np.isnan(last_diff):
        last_diff = -1
    # reshape matrix
    data = trace.data[start:end].reshape([number_of_slices, samples_per_slice])
    # get minimum and maximum for each row
    diff = data.ptp(axis=1)
    # fill NaN with -1 -> means missing data
    data = np.ma.filled(diff, -1)
    # append value of last diff
    data = np.concatenate([first_diff, data, last_diff])
    tr = Trace(data=data, header=trace.stats)
    tr.stats.delta = delta
    tr.stats.npts = len(data)
    tr.stats.starttime = UTCDateTime(start_time)
    return tr


def parseMappingData(lines):
    """
    """
    results = {}
    for line in lines:
        if line.startswith('#'):
            continue
        if line.strip() == '':
            continue
        temp = {}
        data = line.split()
        msg = "Invalid format in mapping data: "
        # check old and new ids
        if len(data) < 2 or len(data) > 4:
            raise Exception(msg + 'expected "old_id new_id starttime endtime"')
        elif data[0].count('.') != 3:
            raise Exception(msg + "old id %s must contain 3 dots" % data[0])
        elif data[1].count('.') != 3:
            raise Exception(msg + "new id %s must contain 3 dots" % data[1])
        old_id = data[0]
        n0, s0, l0, c0 = old_id.split('.')
        n1, s1, l1, c1 = data[1].split('.')
        if len(n0) > 2 or len(n1) > 2:
            raise Exception(msg + "network ids must not exceed 2 characters")
        elif len(s0) > 5 or len(s1) > 5:
            raise Exception(msg + "station ids must not exceed 5 characters")
        elif len(l0) > 2 or len(l1) > 2:
            raise Exception(msg + "location ids must not exceed 2 characters")
        elif len(c0) > 3 or len(c1) > 3:
            raise Exception(msg + "channel ids must not exceed 3 characters")
        temp['network'] = n1
        temp['station'] = s1
        temp['location'] = l1
        temp['channel'] = c1
        # check datetimes if any
        if len(data) > 2:
            try:
                temp['starttime'] = UTCDateTime(data[2])
            except:
                raise msg + "starttime '%s' is not a time format" % data[2]
        else:
            temp['starttime'] = None
        if len(data) > 3:
            try:
                temp['endtime'] = UTCDateTime(data[3])
            except:
                raise msg + "endtime '%s' is not a time format" % data[3]
            if temp['endtime'] < temp['starttime']:
                raise msg + "endtime '%s' should be after starttime" % data[3]
        else:
            temp['endtime'] = None
        results.setdefault(old_id, [])
        results.get(old_id).append(temp)
    return results
