# -*- coding: utf-8 -*-
"""
Py3k compatibility module
"""
from future.utils import PY2

if PY2:
    import urllib2
    urlopen = urllib2.urlopen
    import StringIO
    StringIO = StringIO.StringIO
else:
    import urllib.request
    urlopen = urllib.request.urlopen
    import io
    StringIO = io.StringIO
