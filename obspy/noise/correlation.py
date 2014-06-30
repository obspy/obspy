import h5py
import numpy as np

from obspy.noise.correlation_functions import phase_xcorr


class Correlation(object):
    def __init__(self, trace_a, trace_b, correlation, max_lag,
                 correlation_type, correlation_options=None):
        self.trace_a = trace_a
        self.trace_b = trace_b
        self.correlation = correlation
        self.max_lag = max_lag
        self.correlation_type = correlation_type
        self.correlation_options = correlation_options \
            if correlation_options else {}

    def __add__(self, other):
        corr = CorrelationStack()
        corr += self
        corr += other
        return corr


class CorrelationStack(object):
    def __init__(self, correlations):
        self.__correlations = correlations
        self.__locked = False

    def __add__(self, other):
        if self.__locked:
            msg = "Stack already locked."
            raise ValueError(msg)
        self.__correlations.append(other)

    def stack(self):
        if self.__locked:
            msg = "Stack already locked."
            raise ValueError(msg)
        self.__stack = \
            np.hstack([_i.correlation for _i in self.__correlations])
        self.__locked = True

    def write(self, filename):
        f = h5py.File('myfile.hdf5', 'r')


def correlate_trace(trace_a, trace_b, max_lag, correlation_type, **kwargs):
    # XXX: Add more checks...
    if trace_a.stats.npts != trace_b.stats.npts:
        msg = "Not the same amount of samples."
        raise ValueError(msg)

    max_lag /= int(trace_a.stats.delta)
    if correlation_type == "phase_correlation":
        corr = phase_xcorr(trace_a.data, trace_b.data, max_lag,
                           kwargs.get("nu", 1.0))

    return Correlation(trace_a.stats.copy(), trace_b.stats.copy(),
                       max_lag=max_lag,
                       correlation=corr,
                       correlation_type=correlation_type,
                       correlation_options=kwargs)
