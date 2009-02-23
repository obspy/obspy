# -*- coding: utf-8 -*-

from obspy.mseed.tests import test_libmseed
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_libmseed.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')