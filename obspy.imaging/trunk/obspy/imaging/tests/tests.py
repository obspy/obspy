# -*- coding: utf-8 -*-

from obspy.imaging import beachball
from obspy.imaging.tests import test_beachball
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_beachball.suite())
    return suite


if __name__ == '__main__':
    doctest.testmod(beachball)
    unittest.main(defaultTest='suite')
