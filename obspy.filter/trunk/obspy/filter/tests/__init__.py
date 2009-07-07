# -*- coding: utf-8 -*-

from obspy.filter import invsim
from obspy.filter.tests import test_invsim,test_filter,test_rotate
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_invsim.suite())
    suite.addTest(doctest.DocTestSuite(invsim))
    suite.addTest(test_filter.suite())
    suite.addTest(test_rotate.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
