# -*- coding: utf-8 -*-

from obspy.sac import sacio
from obspy.sac.tests import test_sacio, test_core
import unittest, doctest
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(sacio))
    suite.addTest(test_sacio.suite())
    suite.addTest(test_core.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
