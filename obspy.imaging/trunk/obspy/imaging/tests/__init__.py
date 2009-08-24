# -*- coding: utf-8 -*-

from obspy.imaging import beachball
from obspy.imaging.tests import test_beachball, test_spectrogram
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_beachball.suite())
    suite.addTest(test_spectrogram.suite())
    try:
        suite.addTest(doctest.DocTestSuite(beachball))
    except:
        pass
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
