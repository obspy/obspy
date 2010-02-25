# -*- coding: utf-8 -*-

from obspy.gse2.tests import test_libgse2, test_core
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_core.suite())
    suite.addTest(test_libgse2.suite())
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
