# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

To run the tests, there are the following possibilities

{{{
    python -c 'import obspy; print obspy.runTests()'
    python obspy/core/testing.py
    python obspy/core/testing.py -v # For a more verbose output
    python core/testing.py -v obspy.core.tests.test_stream.StreamTestCase
}}}

You can see the name of the test case by -v option.
"""

import obspy, sys, time, unittest
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
        except ImportError:
            print "Cannot import test suite of module obspy.%s" % module
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
