# -*- coding: utf-8 -*-

from obspy.filter import invsim
from obspy.filter.tests import test_invsim
from obspy.filter.tests import test_filter
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_invsim.suite())
    suite.addTest(doctest.DocTestSuite(invsim))
    suite.addTest(test_filter.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
