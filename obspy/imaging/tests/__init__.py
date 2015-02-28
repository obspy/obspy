# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

import matplotlib

# this code is needed to run the tests without any X11 or any other
# display, e.g. via a SSH connection. Import it only once, else a nasty
# warning occurs.
# see also: http://matplotlib.org/faq/howto_faq.html
try:
    matplotlib.use('AGG', warn=False)
except TypeError:  # needed for matplotlib 0.91.2
    matplotlib.use('AGG')

from obspy.core.util import add_doctests, add_unittests


MODULE_NAME = "obspy.imaging"


def suite():
    suite = unittest.TestSuite()
    add_doctests(suite, MODULE_NAME)
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
