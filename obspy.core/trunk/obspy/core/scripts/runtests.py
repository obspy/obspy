#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

All tests in ObsPy are located in the tests directory of the certain
module. The __init__.py of the tests directory itself as well as every test
has a function called suite.

To run all tests/a single test from the shell/cmd do one of the following::

    python runtests.py                         # Run all tests
    python runtests.py -v                      # Verbose output
    python runtests.py obspy.mseed.tests.suite # run tests of obspy.mseed
    python -c "import obspy.core; obspy.core.runTests()"            # Run all tests
    python -c "import obspy.core; obspy.core.runTests(verbosity=2)" # Verbose output

To run all tests/a single test inside Python do one of the following::

    import obspy.core
    obspy.core.runTests()                      # Run all tests
    obspy.core.runTests(verbosity=1)           # Verbose output
    obspy.core.runTests(verbosity=2, \\        # Special Test case
        modules=['obspy.core.tests.test_stats.StatsTestCase.test_init'])

Running the test verbose exposes the names of available tests.
"""

from optparse import OptionParser
import sys
import time
import unittest
import obspy


DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan']

def suite(modules=[]):
    """
    The obspy test suite.
    """
    for id in DEFAULT_MODULES:
        module = 'obspy.%s.tests' % id
        try:
            __import__(module)
        except ImportError, e:
            print "Cannot import test suite of module %s" % module
            print e
            time.sleep(0.5)
    if modules == []:
        modules = ['obspy.%s.tests.suite' % d for d in DEFAULT_MODULES]
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    return suite


def runTests(verbosity=1, modules=[]):
    """
    This function runs all available tests in obspy, from Python

    :param verbosity: Run tests in verbose mode. 0 quiet, 1 normal, 2 verbose
    :param modules: List of modules for testing. Default runs all.
                    Example ['obspy.core.tests.suite']
    """
    unittest.TextTestRunner(verbosity=verbosity).run(suite(modules))


if __name__ == "__main__":
    usage = "USAGE: %prog [options] modules\n\n" + \
        """Examples:
    %prog                               - run default set of tests
    %prog MyTestSuite                   - run suite 'MyTestSuite'
    %prog MyTestCase.testSomething      - run MyTestCase.testSomething
    %prog MyTestCase                    - run all 'test*' test methods
                                                in MyTestCase
    """
    parser = OptionParser(usage.strip())
    parser.add_option("-v", "--verbose", default=False,
                      action="store_true", dest="verbose",
                      help="verbose mode")
    parser.add_option("-q", "--quiet", default=False,
                      action="store_true", dest="quiet",
                      help="quiet mode")
    (options, _) = parser.parse_args()
    # Simulate same behaviour as standard unittest args parsing
    if options.verbose:
        verbosity = 2
    elif options.quiet:
        verbosity = 0
    else:
        verbosity = 1
    runTests(verbosity, parser.largs)
