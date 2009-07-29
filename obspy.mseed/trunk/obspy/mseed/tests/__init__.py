# -*- coding: utf-8 -*-

import obspy, unittest
from obspy.mseed.tests import test_libmseed, test_core, test_plotting

def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_core.suite())
    suite.addTest(test_libmseed.suite())
    #suite.addTest(test_plotting.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
