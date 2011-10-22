# -*- coding: utf-8 -*-
"""
Feature generators for ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import scoreatpercentile


class BandpassPreviewFeature(object):
    def __init__(self):
        # for API doc
        pass

    def process(self, trace):
        # applying bandpass on preview trace
        trace.filter("bandpass", freqmin=0.1, freqmax=20.0)
        return {}


class MinMaxAmplitudeFeature(object):
    def __init__(self):
        # for API doc
        pass

    def process(self, trace):
        """
        Gets some statistics about the amplitude values.

        This may take a while to calculate - use a moderate looping interval.

        .. rubric:: Example

        >>> from obspy.core import Trace
        >>> import numpy as np
        >>> tr = Trace(data=np.arange(-5,5))
        >>> result = AmplitudeFeature().process(tr)
        >>> result['max']
        4
        >>> result['upper_quantile']
        1.75
        """
        result = {}
        result['min'] = float(trace.data.min())
        result['max'] = float(trace.data.max())
        result['avg'] = float(trace.data.mean())
        result['median'] = float(scoreatpercentile(trace.data, 50, False))
        result['lower_quantile'] = float(scoreatpercentile(trace.data, 25))
        result['upper_quantile'] = float(scoreatpercentile(trace.data, 75))
        return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
