# -*- coding: utf-8 -*-
"""
Feature generators for ObsPy Trace objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util import quantile


def getMinMaxAmplitude(trace, qtype=7, *args, **kwargs):
    """
    Gets some statistics about the amplitude values.

    Example
    -------
    >>> from obspy.core import Trace
    >>> import numpy as np
    >>> tr = Trace(data=np.arange(-5,5))
    >>> result = getMinMaxAmplitude(tr)
    >>> result['max']
    4
    >>> result['upper_quantile']
    1.75
    """
    result = {}
    result['min'] = trace.data.min()
    result['max'] = trace.data.max()
    result['avg'] = trace.data.mean()
    result['median'] = quantile(trace.data, 0.5, issorted=False, qtype=qtype)
    result['lower_quantile'] = quantile(trace.data, 0.25, qtype=qtype)
    result['upper_quantile'] = quantile(trace.data, 0.75, qtype=qtype)
    return result


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
