# -*- coding: utf-8 -*-

from obspy.picker import ctypes_recstalta
import unittest, doctest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(ctypes_recstalta))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
