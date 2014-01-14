# -*- coding: utf-8 -*-
"""
Py3k compatibility module
"""
from future.utils import PY2
import numpy as np

if PY2:
    import urllib2
    urlopen = urllib2.urlopen
    from urlparse import urlparse  # NOQA
    from urllib import urlencode  # NOQA
    from urllib2 import HTTPPasswordMgrWithDefaultRealm  # NOQA
    from urllib2 import HTTPBasicAuthHandler  # NOQA
    from urllib2 import HTTPDigestAuthHandler  # NOQA
    from urllib2 import build_opener  # NOQA
    from urllib2 import install_opener  # NOQA
    from urllib2 import HTTPError  # NOQA
    from urllib2 import Request  # NOQA
    from httplib import HTTPConnection  # NOQA
else:
    import urllib.request
    urlopen = urllib.request.urlopen
    from urllib.parse import urlparse  # NOQA
    from urllib.parse import urlencode  # NOQA
    from urllib.request import HTTPPasswordMgrWithDefaultRealm  # NOQA
    from urllib.request import HTTPBasicAuthHandler  # NOQA
    from urllib.request import HTTPDigestAuthHandler  # NOQA
    from urllib.request import build_opener  # NOQA
    from urllib.request import install_opener  # NOQA
    from urllib.request import HTTPError  # NOQA
    from urllib.request import Request  # NOQA
    from http.client import HTTPConnection  # NOQA

if PY2:
    from StringIO import StringIO
    from StringIO import StringIO as BytesIO
else:
    import io
    StringIO = io.StringIO
    BytesIO = io.BytesIO

if PY2:
    from string import maketrans
else:
    maketrans = bytes.maketrans


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
