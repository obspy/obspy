# -*- coding: utf-8 -*-

import matplotlib
# this code is needed to run the tests without any X11 or any other
# display, e.g. via a SSH connection. Import it only once, else a nasty
# warning occurs.
# see also: http://matplotlib.sourceforge.net/faq/howto_faq.html
try:
    matplotlib.use('AGG', warn=False)
except TypeError:  # needed for matplotlib 0.91.2
    matplotlib.use('AGG')

import unittest
from obspy.core.util import add_doctests, add_unittests


MODULE_NAME = "obspy.imaging"


def suite():
    suite = unittest.TestSuite()
    add_doctests(suite, MODULE_NAME)
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
