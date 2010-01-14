# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

All tests in ObsPy are located in the tests directory of the certain
module. The __init__.py of the tests directory itself as well as every test
has a function called suite.

To run all tests/a single test from the shell/cmd do one of the following::

    python -c "import obspy.core; obspy.core.runTests()"       # Run all tests
    python -c "import obspy.core; obspy.core.runTests(True)"   # Verbose output
    python obspy/core/testing.py      # Run all tests
    python obspy/core/testing.py -v   # Verbose output
    python obspy/core/tests/test_stats.py -v
    python obspy/core/tests/test_stats.py -v StatsTestCase.test_pickleStats

To run all tests/a single test inside Python do one of the following::

    import obspy.core
    obspy.core.runTests()               # Run all tests
    obspy.core.runTests(verbose=True)   # Verbose output

    from unittest import TextTestRunner
    from obspy.core.tests import suite
    TextTestRunner().run(suite())              # Run all tests
    TextTestRunner(verbosity=2).run(suite())   # Verbose output

    from unittest import TextTestRunner
    from obspy.core.tests.test_stats import suite
    TextTestRunner().run(suite())              # Run all tests
    TextTestRunner(verbosity=2).run(suite())   # Verbose output

Running the test verbose exposes the available tests.
"""

import sys
import time
import unittest
import obspy

_dirs = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging', 'xseed',
         'seisan']
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
            print "Cannot import test suite of module %s" % module
            print e
            time.sleep(0.5)
        else:
            suite.addTests(sys.modules[module].suite())
    return suite


def runTests(verbose=False):
    """
    This function runs all available tests in obspy, from python
    """
    if verbose:
        unittest.TextTestRunner(verbosity=2).run(suite())
    else:
        unittest.main(defaultTest='suite', module="obspy.core.testing")


if __name__ == '__main__':
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            pass
    unittest.main(defaultTest='suite')
