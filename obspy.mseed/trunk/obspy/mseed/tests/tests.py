# -*- coding: utf-8 -*-

import unittest

from obspy.mseed.tests import test_libmseed
#from obspy.mseed.tests import test_doctests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_libmseed.suite())
#    suite.addTest(test_doctests.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')