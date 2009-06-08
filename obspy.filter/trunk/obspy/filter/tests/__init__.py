# -*- coding: utf-8 -*-

from obspy.filter import invsim
from obspy.filter.tests import test_invsim
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_invsim.suite())
    suite.addTest(doctest.DocTestSuite(invsim))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
