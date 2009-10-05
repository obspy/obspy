# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

To run the tests, there are the following possibilities

{{{
    python -c 'import obspy; print obspy.runTests()' # Run all tests
    python obspy/core/testing.py    # Run all tests
    python obspy/core/testing.py -v # Verbose output
    python core/testing.py -v obspy.core.tests.test_stream.StreamTestCase.test_adding
}}}

Find out the name of a specific test by using the -v options.
"""

import obspy
import sys
import time
import unittest


_dirs = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging', 'xseed']
modules = ['obspy.%s.tests' % d for d in _dirs]


def suite():
    """
    The obspy test suite.
    """
    suite = unittest.TestSuite()
    for module in modules:
        try:
            __import__(module)
        except ImportError, e:
            print "Cannot import test suite of module obspy.%s" % module
            print e
            time.sleep(0.5)
        else:
            suite.addTests(sys.modules[module].suite())
    return suite


def runTests():
    """
    This function runs all available tests in obspy, from python
    """
    unittest.main(defaultTest='suite', module=obspy.core.testing)


if __name__ == '__main__':
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            pass
    unittest.main(defaultTest='suite')
