# -*- coding: utf-8 -*-

import obspy
import time
import unittest
import sys


def suite():
    """
    The obspy test suite.
    """
    suite = unittest.TestSuite()
    for module in ['core', 'gse2', 'mseed', 'sac', 'wav', 'filter', 
                   'imaging', 'xseed', 'trigger', 'arclink']:
        try:
            name = 'obspy.%s.tests' % module
            __import__(name)
            suite.addTests(sys.modules[name].suite())
        except:
            print "Cannot import test suite of module obspy.%s" % module
            time.sleep(0.5)
            continue
    return suite


def runTests():
    """
    This function runs all available tests in obspy
    """
    unittest.TextTestRunner(verbosity=2).run(suite())
#    unittest.main(defaultTest='suite')


if __name__ == '__main__':
    runTests()
