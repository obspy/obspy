#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

All tests in ObsPy are located in the tests directory of the certain
module. The __init__.py of the tests directory itself as well as every test
file located in the tests directory has a function called suite. The suite
function is the one that needs to be called. Running the test verbose
exposes the names of the available tests.

Examples:
    python runtests.py                         # Run all tests
    python runtests.py -v                      # Verbose output
    python runtests.py obspy.mseed.tests.suite # run tests of obspy.mseed
    python runtests.py mseed                   # obspy shortcut
    python runtests.py obspy.core.tests.test_stats.StatsTestCase.test_init
    python -c "import obspy.core; obspy.core.runTests()"            # Run all tests
    python -c "import obspy.core; obspy.core.runTests(verbosity=2)" # Verbose output
    python -c "import obspy.core; obspy.core.runTests(verbosity=2,\\# Special Test case
        tests=['obspy.core.tests.test_stats.StatsTestCase.test_init'])
"""

from optparse import OptionParser
import unittest

DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh']


def suite(tests=[]):
    """
    The obspy test suite.
    """
    if tests == []:
        names = ['obspy.%s.tests.suite' % d for d in DEFAULT_MODULES]
    else:
        names = []
        # Search for short cuts in tests, if there are no short cuts
        # names is equal to tests
        for test in tests:
            if test in DEFAULT_MODULES:
                test = 'obspy.%s.tests.suite' % test
            names.append(test)
    # Construct the test suite from the given names. Note modules need not
    # be imported before in this case
    suite = unittest.TestLoader().loadTestsFromNames(names)
    return suite


def runTests(verbosity=1, tests=[]):
    """
    This function runs all available tests in obspy, from Python

    :param verbosity: Run tests in verbose mode. 0 quiet, 1 normal, 2 verbose
    :param tests: List of tests to run. Defaults to runs all.
                    Example ['obspy.core.tests.suite']
    """
    unittest.TextTestRunner(verbosity=verbosity).run(suite(tests))


def main():
    usage = "USAGE: %prog [options] modules\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip())
    parser.add_option("-v", "--verbose", default=False,
                      action="store_true", dest="verbose",
                      help="verbose mode")
    parser.add_option("-q", "--quiet", default=False,
                      action="store_true", dest="quiet",
                      help="quiet mode")
    (options, _) = parser.parse_args()
    # set correct verbosity level
    if options.verbose:
        verbosity = 2
    elif options.quiet:
        verbosity = 0
    else:
        verbosity = 1
    runTests(verbosity, parser.largs)


if __name__ == "__main__":
    # It is not possible to add the code of main directly to here.
    # This script is automatically installed with name obspy-runtests by
    # setup.py to the Scripts or bin directory of your Python distribution
    # setup.py needs a function to which it's scripts can be linked.
    main()
