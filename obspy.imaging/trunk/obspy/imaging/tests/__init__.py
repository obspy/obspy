# -*- coding: utf-8 -*-

import matplotlib

# this code is needed to run the tests without any X11 or any other
# display, e.g. via a SSH connection. Import it only once, else a nasty
# warning occurs.
# :see: http://matplotlib.sourceforge.net/faq/howto_faq.html#matplotlib-in-a-web-application-server
try:
    matplotlib.use('AGG', warn=False)
except TypeError: #needed for matplotlib 0.91.2
    matplotlib.use('AGG')
from matplotlib import pyplot as plt

from obspy.imaging.tests import test_backend, test_beachball, test_spectrogram
from obspy.imaging.tests import test_waveform
from obspy.imaging import beachball
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_backend.suite())
    suite.addTest(test_beachball.suite())
    suite.addTest(test_spectrogram.suite())
    suite.addTest(test_waveform.suite())
    try:
        suite.addTest(doctest.DocTestSuite(beachball))
    except:
        pass
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
