# -*- coding: utf-8 -*-

from obspy.core import util
from obspy.core.tests import test_core
import unittest
import doctest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(util))
    suite.addTest(test_core.suite())
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

