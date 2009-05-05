# -*- coding: utf-8 -*-

from obspy.imaging import beachball
from obspy.imaging.tests import test_beachball
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_beachball.suite())
    suite.addTest(doctest.DocTestSuite(beachball))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
