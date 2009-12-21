# -*- coding: utf-8 -*-

# this code is needed to run the tests without any X11 or any other
# display, e.g. via a ssh connection. Import it only once, else a nasty
# warning occurs.
import matplotlib
try:
    matplotlib.use('Agg', warn=False)
except TypeError: #needed for matplotlib 0.91.2
    matplotlib.use('Agg')
from matplotlib import pyplot as plt

from obspy.imaging.tests import test_backend, test_beachball, test_spectrogram
from obspy.imaging import beachball
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_backend.suite())
    suite.addTest(test_beachball.suite())
    suite.addTest(test_spectrogram.suite())
    try:
        suite.addTest(doctest.DocTestSuite(beachball))
    except:
        pass
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
