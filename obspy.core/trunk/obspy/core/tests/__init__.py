# -*- coding: utf-8 -*-

from obspy.core import util
import unittest
import doctest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(util))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

