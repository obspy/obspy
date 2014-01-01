# -*- coding: utf-8 -*-
"""
Py3k compatibility module
"""
from future.utils import PY2
import numpy as np

if PY2:
    import urllib2
    urlopen = urllib2.urlopen
else:
    import urllib.request
    urlopen = urllib.request.urlopen


if PY2:
    from StringIO import StringIO
else:
    import io
    StringIO = io.StringIO

if PY2:
    from StringIO import StringIO as BytesIO
else:
    import io
    BytesIO = io.BytesIO

if PY2:
    from string import maketrans
else:
    maketrans = bytes.maketrans

if PY2:
    from urlparse import urlparse
    from urllib import urlencode
else:
    from urllib.parse import urlparse
    from urllib.parse import urlencode

def round_away(number):
    """
    Simple function that rounds a number to the nearest integer. If the number
    is halfway between two integers, it will round away from zero. Of course
    only works up machine precision. This should hopefully behave like the
    round() function in Python 2.

    This is potentially desired behaviour in the trim functions but some more
    thought should be poured into it.

    The np.round() function rounds towards the even nearest even number in case
    of half-way splits.

    >>> round_away(2.5)
    3
    >>> round_away(-2.5)
    -3

    >>> round_away(10.5)
    11
    >>> round_away(-10.5)
    -11

    >>> round_away(11.0)
    11
    >>> round_away(-11.0)
    -11
    """

    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))
