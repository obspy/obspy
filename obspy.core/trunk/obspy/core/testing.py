# -*- coding: utf-8 -*-

import sys
import time
import unittest


def suite():
    """
    The obspy test suite.
    """
    suite = unittest.TestSuite()
    for module in ['core', 'gse2', 'mseed', 'sac', 'wav', 'filter', 'imaging',
                   'xseed', 'trigger', 'arclink']:
        name = 'obspy.%s.tests' % module
        try:
            __import__(name)
        except ImportError:
            print "Cannot import test suite of module obspy.%s" % module
            time.sleep(0.5)
        else:
            suite.addTests(sys.modules[name].suite())
    return suite


def runTests():
    """
    This function runs all available tests in obspy
    """
    unittest.TextTestRunner(verbosity=2).run(suite())
#    unittest.main(defaultTest='suite')


if __name__ == '__main__':
    runTests()
