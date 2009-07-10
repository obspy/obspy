# -*- coding: utf-8 -*-

from obspy.trigger import trigger
import unittest, doctest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(trigger))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
