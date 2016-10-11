# -*- coding: utf-8 -*-
"""
Optional feature generators for ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.core.util import score_at_percentile


class BandpassPreviewFeature(object):
    """
    Bandpass filter (freqmin=0.1, freqmax=20.0) all trace previews.
    """

    def process(self, trace):
        """
        Bandpass filter (freqmin=0.1, freqmax=20.0) all trace previews.
        """
        # applying bandpass on trace directly - this will not modify the
        # original waveform file but it will influence the preview trace
        trace.filter("bandpass", freqmin=0.1, freqmax=20.0)
        return {}


class MinMaxAmplitudeFeature(object):
    """
    Generates statistics about the amplitude values.
    """

    def process(self, trace):
        """
        Generates statistics about the amplitude values.

        This may take a while to calculate - use a moderate looping interval.

        .. rubric:: Example

        >>> from obspy import Trace
        >>> import numpy as np
        >>> tr = Trace(data=np.arange(-5,5))
        >>> result = MinMaxAmplitudeFeature().process(tr)
        >>> result['max']
        4.0
        >>> result['upper_quantile']
        1.75
        """
        result = {}
        result['min'] = float(trace.data.min())
        result['max'] = float(trace.data.max())
        result['avg'] = float(trace.data.mean())
        result['median'] = float(score_at_percentile(trace.data, 50, False))
        result['lower_quantile'] = float(score_at_percentile(trace.data, 25))
        result['upper_quantile'] = float(score_at_percentile(trace.data, 75))
        return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
