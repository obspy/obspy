#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that runs all ObsPy tests.

All tests in ObsPy are located in the tests directory of the each specific
module. The __init__.py of the tests directory itself as well as every test
file located in the tests directory has a function called suite, which is
executed using this script. Running the script with the verbose keyword exposes
the names of all available test cases.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

.. rubric:: Examples

(1) Run all local tests (ignoring tests requiring a network connection) on
    command line::

        $ obspy-runtests

    or via Python interpreter

    >>> import obspy.core
    >>> obspy.core.run_tests()  # DOCTEST: +SKIP

(2) Run all tests on command line::

        $ obspy-runtests --all

    or via Python interpreter

    >>> import obspy.core
    >>> obspy.core.run_tests(all=True)  # DOCTEST: +SKIP

(3) Verbose output::

        $ obspy-runtests -v

    or

    >>> import obspy.core
    >>> obspy.core.run_tests(verbosity=2)  # DOCTEST: +SKIP

(4) Run tests of module :mod:`obspy.io.mseed`::

        $ obspy-runtests obspy.io.mseed.tests.suite

    or as shortcut::

        $ obspy-runtests io.mseed

(5) Run tests of multiple modules, e.g. :mod:`obspy.io.wav` and
    :mod:`obspy.io.sac`::

        $ obspy-runtests io.wav io.sac

(6) Run a specific test case::

        $ obspy-runtests obspy.core.tests.test_stats.StatsTestCase.test_init

    or

    >>> import obspy.core
    >>> tests = ['obspy.core.tests.test_stats.StatsTestCase.test_init']
    >>> obspy.core.run_tests(verbosity=2, tests=tests)  # DOCTEST: +SKIP

(7) Report test results to http://tests.obspy.org/::

        $ obspy-runtests -r

(8) To get a full list of all options, use::

        $ obspy-runtests --help

Of course you may combine most of the options here, e.g. in order to test
all modules except the module obspy.io.sh and obspy.clients.seishub, have a
verbose output and report everything, you would run::

        $ obspy-runtests -r -v -x clients.seishub -x io.sh --all
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import copy
import doctest
import glob
import importlib
import operator
import os
import platform
import sys
import time
import types
import unittest
import warnings
from argparse import ArgumentParser

import numpy as np

from obspy.core.util import ALL_MODULES, DEFAULT_MODULES, NETWORK_MODULES
from obspy.core.util.testing import MODULE_TEST_SKIP_CHECKS
from obspy.core.util.version import get_git_version


DEPENDENCIES = ['numpy', 'scipy', 'matplotlib', 'lxml.etree', 'sqlalchemy',
                'suds', 'mpl_toolkits.basemap', 'mock', 'future',
                "flake8", "pyflakes", "pyimgur"]

PSTATS_HELP = """
Call "python -m pstats obspy.pstats" for an interactive profiling session.

The following commands will produce the same output as shown above:
  sort cumulative
  stats obspy. 20

Type "help" to see all available options.
"""

HOSTNAME = platform.node().split('.', 1)[0]


# XXX: start of ugly monkey patch for Python 2.7
# classes _TextTestRunner and _WritelnDecorator have been marked as depreciated
class _WritelnDecorator(object):
    """
    Used to decorate file-like objects with a handy 'writeln' method
    """
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream, attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n')  # text-mode streams translate to \r\n if needed

unittest._WritelnDecorator = _WritelnDecorator
# XXX: end of ugly monkey patch


def _get_suites(verbosity=1, names=[]):
    """
    The ObsPy test suite.
    """
    # Construct the test suite from the given names. Modules
    # need not be imported before in this case
    suites = {}
    ut = unittest.TestLoader()
    status = True
    for name in names:
        suite = []
        if name in ALL_MODULES:
            # Search for short cuts in tests
            test = 'obspy.%s.tests.suite' % name
        else:
            # If no short cuts names variable = test variable
            test = name
        try:
            suite.append(ut.loadTestsFromName(test, None))
        except Exception as e:
            status = False
            if verbosity:
                print(e)
                print("Cannot import test suite for module obspy.%s" % name)
        else:
            suites[name] = ut.suiteClass(suite)
    return suites, status


def _create_report(ttrs, timetaken, log, server, hostname, sorted_tests):
    # import additional libraries here to speed up normal tests
    from future import standard_library
    with standard_library.hooks():
        import urllib.parse
        import http.client
    import codecs
    from xml.etree import ElementTree as etree
    from xml.sax.saxutils import escape
    timestamp = int(time.time())
    result = {'timestamp': timestamp}
    result['slowest_tests'] = [("%0.3fs" % dt, "%s" % desc)
                               for (desc, dt) in sorted_tests[:20]]
    result['timetaken'] = timetaken
    if log:
        try:
            data = codecs.open(log, 'r', encoding='UTF-8').read()
            result['install_log'] = escape(data)
        except:
            print("Cannot open log file %s" % log)
    # get ObsPy module versions
    result['obspy'] = {}
    tests = 0
    errors = 0
    failures = 0
    skipped = 0
    try:
        installed = get_git_version()
    except:
        installed = ''
    result['obspy']['installed'] = installed
    for module in sorted(ALL_MODULES):
        result['obspy'][module] = {}
        if module not in ttrs:
            continue
        result['obspy'][module]['installed'] = installed
        # test results
        ttr = ttrs[module]
        result['obspy'][module]['timetaken'] = ttr.__dict__['timetaken']
        result['obspy'][module]['tested'] = True
        result['obspy'][module]['tests'] = ttr.testsRun
        # skipped is not supported for Python < 2.7
        try:
            skipped += len(ttr.skipped)
            result['obspy'][module]['skipped'] = len(ttr.skipped)
        except AttributeError:
            skipped = ''
            result['obspy'][module]['skipped'] = ''
        tests += ttr.testsRun
        # depending on module type either use failure (network related modules)
        # or errors (all others)
        result['obspy'][module]['errors'] = {}
        result['obspy'][module]['failures'] = {}
        if module in NETWORK_MODULES:
            for _, text in ttr.errors:
                result['obspy'][module]['failures']['f%s' % (failures)] = text
                failures += 1
            for _, text in ttr.failures:
                result['obspy'][module]['failures']['f%s' % (failures)] = text
                failures += 1
        else:
            for _, text in ttr.errors:
                result['obspy'][module]['errors']['f%s' % (errors)] = text
                errors += 1
            for _, text in ttr.failures:
                result['obspy'][module]['errors']['f%s' % (errors)] = text
                errors += 1
    # get dependencies
    result['dependencies'] = {}
    for module in DEPENDENCIES:
        temp = module.split('.')
        try:
            mod = __import__(module,
                             fromlist=[native_str(temp[1:])])
            if module == '_omnipy':
                result['dependencies'][module] = mod.coreVersion()
            else:
                result['dependencies'][module] = mod.__version__
        except ImportError:
            result['dependencies'][module] = ''
    # get system / environment settings
    result['platform'] = {}
    for func in ['system', 'release', 'version', 'machine',
                 'processor', 'python_version', 'python_implementation',
                 'python_compiler', 'architecture']:
        try:
            temp = getattr(platform, func)()
            if isinstance(temp, tuple):
                temp = temp[0]
            result['platform'][func] = temp
        except:
            result['platform'][func] = ''
    # set node name to hostname if set
    result['platform']['node'] = hostname
    # post only the first part of the node name (only applies to MacOS X)
    try:
        result['platform']['node'] = result['platform']['node'].split('.')[0]
    except:
        pass
    # test results
    result['tests'] = tests
    result['errors'] = errors
    result['failures'] = failures
    result['skipped'] = skipped

    # generate XML document
    def _dict2xml(doc, result):
        for key, value in result.items():
            key = key.split('(')[0].strip()
            if isinstance(value, dict):
                child = etree.SubElement(doc, key)
                _dict2xml(child, value)
            elif value is not None:
                if isinstance(value, (str, native_str)):
                    etree.SubElement(doc, key).text = value
                elif isinstance(value, (str, native_str)):
                    etree.SubElement(doc, key).text = str(value, 'utf-8')
                else:
                    etree.SubElement(doc, key).text = str(value)
            else:
                etree.SubElement(doc, key)
    root = etree.Element("report")
    _dict2xml(root, result)
    xml_doc = etree.tostring(root)
    print()
    # send result to report server
    params = urllib.parse.urlencode({
        'timestamp': timestamp,
        'system': result['platform']['system'],
        'python_version': result['platform']['python_version'],
        'architecture': result['platform']['architecture'],
        'tests': tests,
        'failures': failures,
        'errors': errors,
        'modules': len(ttrs),
        'xml': xml_doc
    })
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain"}
    conn = http.client.HTTPConnection(server)
    conn.request("POST", "/", params, headers)
    # get the response
    response = conn.getresponse()
    # handle redirect
    if response.status == 301:
        o = urllib.parse.urlparse(response.msg['location'])
        conn = http.client.HTTPConnection(o.netloc)
        conn.request("POST", o.path, params, headers)
        # get the response
        response = conn.getresponse()
    # handle errors
    if response.status == 200:
        print("Test report has been sent to %s. Thank you!" % (server))
    else:
        print("Error: Could not sent a test report to %s." % (server))
        print(response.reason)


class _TextTestResult(unittest._TextTestResult):
    """
    A custom test result class that can print formatted text results to a
    stream. Used by TextTestRunner.
    """
    timer = []

    def startTest(self, test):
        self.start = time.time()
        super(_TextTestResult, self).startTest(test)

    def stopTest(self, test):
        super(_TextTestResult, self).stopTest(test)
        self.timer.append((test, time.time() - self.start))


def _skip_test(test_case, msg):
    """
    Helper method intended to be bound to a `unittest.TestCase`
    instance overwriting the `setUp()` method to immediately and
    unconditionally skip the test when executed.

    :type test_case: unittest.TestCase
    :type msg: str
    :param msg: Reason for unconditionally skipping the test.
    """
    test_case.skipTest(msg)


def _recursive_skip(test_suite, msg):
    """
    Helper method to recursively skip all tests aggregated in `test_suite`
    with the the specified message.

    :type test_suite: unittest.TestSuite
    :type msg: str
    :param msg: Reason for unconditionally skipping the tests.
    """
    def _custom_skip_test(testcase):
        _skip_test(testcase, msg)

    if isinstance(test_suite, unittest.TestSuite):
        for obj in test_suite:
            _recursive_skip(obj, msg)
    elif isinstance(test_suite, unittest.TestCase):
        # overwrite setUp method
        test_suite.setUp = types.MethodType(_custom_skip_test, test_suite)
    else:
        raise NotImplementedError()


class _TextTestRunner:
    def __init__(self, stream=sys.stderr, descriptions=1, verbosity=1,
                 timeit=False):
        self.stream = unittest._WritelnDecorator(stream)  # @UndefinedVariable
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.timeit = timeit

    def _make_result(self):
        return _TextTestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, suites):
        """
        Run the given test case or test suite.
        """
        results = {}
        time_taken = 0
        keys = sorted(suites.keys())
        for id in keys:
            test = suites[id]
            # run checker routine if any,
            # to see if module's tests can be executed
            msg = None
            if id in MODULE_TEST_SKIP_CHECKS:
                # acquire function specified by string
                mod, func = MODULE_TEST_SKIP_CHECKS[id].rsplit(".", 1)
                mod = importlib.import_module(mod)
                func = getattr(mod, func)
                msg = func()
            # we encountered an error message, so skip all tests with given
            # message
            if msg:
                _recursive_skip(test, msg)
            result = self._make_result()
            start = time.time()
            test(result)
            stop = time.time()
            results[id] = result
            total = stop - start
            results[id].__dict__['timetaken'] = total
            if self.timeit:
                self.stream.writeln('')
                self.stream.write("obspy.%s: " % (id))
                num = test.countTestCases()
                try:
                    avg = float(total) / num
                except:
                    avg = 0
                msg = '%d tests in %.3fs (average of %.4fs per test)'
                self.stream.writeln(msg % (num, total, avg))
                self.stream.writeln('')
            time_taken += total
        runs = 0
        faileds = 0
        erroreds = 0
        wasSuccessful = True
        if self.verbosity:
            self.stream.writeln()
        for result in results.values():
            failed, errored = map(len, (result.failures, result.errors))
            faileds += failed
            erroreds += errored
            if not result.wasSuccessful():
                wasSuccessful = False
                result.printErrors()
            runs += result.testsRun
        if self.verbosity:
            self.stream.writeln(unittest._TextTestResult.separator2)
            self.stream.writeln("Ran %d test%s in %.3fs" %
                                (runs, runs != 1 and "s" or "", time_taken))
            self.stream.writeln()
        if not wasSuccessful:
            self.stream.write("FAILED (")
            if faileds:
                self.stream.write("failures=%d" % faileds)
            if erroreds:
                if faileds:
                    self.stream.write(", ")
                self.stream.write("errors=%d" % erroreds)
            self.stream.writeln(")")
        elif self.verbosity:
            self.stream.writeln("OK")
        return results, time_taken, (faileds + erroreds)


def run_tests(verbosity=1, tests=[], report=False, log=None,
              server="tests.obspy.org", all=False, timeit=False,
              interactive=False, slowest=0, exclude=[], tutorial=False,
              hostname=HOSTNAME):
    """
    This function executes ObsPy test suites.

    :type verbosity: int, optional
    :param verbosity: Run tests in verbose mode (``0``=quiet, ``1``=normal,
        ``2``=verbose, default is ``1``).
    :type tests: list of str, optional
    :param tests: Test suites to run. If no suite is given all installed tests
        suites will be started (default is a empty list).
        Example ``['obspy.core.tests.suite']``.
    :type report: bool, optional
    :param report: Submits a test report if enabled (default is ``False``).
    :type log: str, optional
    :param log: Filename of install log file to append to report.
    :type server: str, optional
    :param server: Report server URL (default is ``"tests.obspy.org"``).
    """
    if all:
        tests = copy.copy(ALL_MODULES)
    elif not tests:
        tests = copy.copy(DEFAULT_MODULES)
    # remove any excluded module
    if exclude:
        for name in exclude:
            try:
                tests.remove(name)
            except ValueError:
                pass
    # fetch tests suites
    suites, status = _get_suites(verbosity, tests)
    # add testsuite for all of the tutorial's rst files
    if tutorial:
        try:
            # assume we are in the trunk
            tut_path = os.path.dirname(__file__)
            tut_path = os.path.join(tut_path, '..', '..', '..', '..', 'misc',
                                    'docs', 'source', 'tutorial', '*.rst')
            tut_suite = unittest.TestSuite()
            for file in glob.glob(tut_path):
                filesuite = doctest.DocFileSuite(file, module_relative=False)
                tut_suite.addTest(filesuite)
            suites['tutorial'] = tut_suite
        except:
            msg = "Could not add tutorial files to tests."
            warnings.warn(msg)
    # run test suites
    ttr, total_time, errors = _TextTestRunner(verbosity=verbosity,
                                              timeit=timeit).run(suites)
    # sort tests by time taken
    mydict = {}
    # loop over modules
    for mod in ttr.values():
        mydict.update(dict(mod.timer))
    sorted_tests = sorted(iter(mydict.items()), key=operator.itemgetter(1))
    sorted_tests = sorted_tests[::-1]

    if slowest:
        slowest_tests = ["%0.3fs: %s" % (dt, desc)
                         for (desc, dt) in sorted_tests[:slowest]]
        print()
        print("Slowest Tests")
        print("-------------")
        print(os.linesep.join(slowest_tests))
        print()
        print()
    if interactive and not report:
        msg = "Do you want to report this to %s? [n]: " % (server)
        var = input(msg).lower()
        if var in ('y', 'yes', 'yoah', 'hell yeah!'):
            report = True
    if report:
        _create_report(ttr, total_time, log, server, hostname, sorted_tests)
    # make obspy-runtests exit with 1 if a test suite could not be added,
    # indicating failure
    if status is False:
        errors += 1
    if errors:
        return errors


def run(argv=None, interactive=True):
    import matplotlib
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            matplotlib.use('Agg')
    except UserWarning:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
    if matplotlib.get_backend().upper() != "AGG":
        msg = "unable to change backend to 'AGG' (to avoid windows popping up)"
        warnings.warn(msg)

    parser = ArgumentParser(prog='obspy-runtests',
                            description='A command-line program that runs all '
                                        'ObsPy tests.')
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + get_git_version())
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='quiet mode')

    # filter options
    filter = parser.add_argument_group('Module Filter',
                                       'Providing no modules will test all '
                                       'ObsPy modules which do not require an '
                                       'active network connection.')
    filter.add_argument('-a', '--all', action='store_true',
                        help='test all modules (including network modules)')
    filter.add_argument('-x', '--exclude', action='append',
                        help='exclude given module from test')
    filter.add_argument('tests', nargs='*',
                        help='test modules to run')

    # timing / profile options
    timing = parser.add_argument_group('Timing/Profile Options')
    timing.add_argument('-t', '--timeit', action='store_true',
                        help='shows accumulated run times of each module')
    timing.add_argument('-s', '--slowest', default=0, type=int, dest='n',
                        help='lists n slowest test cases')
    timing.add_argument('-p', '--profile', action='store_true',
                        help='uses cProfile, saves the results to file ' +
                             'obspy.pstats and prints some profiling numbers')

    # reporting options
    report = parser.add_argument_group('Reporting Options')
    report.add_argument('-r', '--report', action='store_true',
                        help='automatically submit a test report')
    report.add_argument('-d', '--dontask', action='store_true',
                        help="don't explicitly ask for submitting a test "
                             "report")
    report.add_argument('-u', '--server', default='tests.obspy.org',
                        help='report server (default is tests.obspy.org)')
    report.add_argument('-n', '--node', dest='hostname', default=HOSTNAME,
                        help='nodename visible at the report server')
    report.add_argument('-l', '--log', default=None,
                        help='append log file to test report')

    # other options
    others = parser.add_argument_group('Additional Options')
    others.add_argument('--tutorial', action='store_true',
                        help='add doctests in tutorial')
    others.add_argument('--no-flake8', action='store_true',
                        help='skip code formatting test')
    others.add_argument('--keep-images', action='store_true',
                        help='store images created during image comparison '
                             'tests in subfolders of baseline images')
    others.add_argument('--keep-only-failed-images', action='store_true',
                        help='when storing images created during testing, '
                             'only store failed images and the corresponding '
                             'diff images (but not images that passed the '
                             'corresponding test).')

    args = parser.parse_args(argv)
    # set correct verbosity level
    if args.verbose:
        verbosity = 2
        # raise all NumPy warnings
        np.seterr(all='raise')
        # raise user and deprecation warnings
        warnings.simplefilter("error", UserWarning)
    elif args.quiet:
        verbosity = 0
        # ignore user and deprecation warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        # don't ask to send a report
        args.dontask = True
    else:
        verbosity = 1
        # show all NumPy warnings
        np.seterr(all='print')
        # ignore user warnings
        warnings.simplefilter("ignore", UserWarning)
    # check for send report option or environmental settings
    if args.report or 'OBSPY_REPORT' in os.environ.keys():
        report = True
    else:
        report = False
    if 'OBSPY_REPORT_SERVER' in os.environ.keys():
        args.server = os.environ['OBSPY_REPORT_SERVER']
    # check interactivity settings
    if interactive and args.dontask:
        interactive = False
    if args.keep_images:
        os.environ['OBSPY_KEEP_IMAGES'] = ""
    if args.keep_only_failed_images:
        os.environ['OBSPY_KEEP_ONLY_FAILED_IMAGES'] = ""
    if args.no_flake8:
        os.environ['OBSPY_NO_FLAKE8'] = ""
    return run_tests(verbosity, args.tests, report, args.log, args.server,
                     args.all, args.timeit, interactive, args.n,
                     exclude=args.exclude, tutorial=args.tutorial,
                     hostname=args.hostname)


def main(argv=None, interactive=True):
    """
    Entry point for setup.py.

    Wrapper for a profiler if requested otherwise just call run() directly.
    If profiling is enabled we disable interactivity as it would wait for user
    input and influence the statistics. However the -r option still works.
    """
    # catch and ignore a NumPy deprecation warning
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            "ignore", 'The compiler package is deprecated and removed in '
            'Python 3.x.', DeprecationWarning)
        np.safe_eval('1')

    if '-p' in sys.argv or '--profile' in sys.argv:
        try:
            import cProfile as Profile
        except ImportError:
            import Profile
        Profile.run('from obspy.core.scripts.runtests import run; run()',
                    'obspy.pstats')
        import pstats
        stats = pstats.Stats('obspy.pstats')
        print()
        print("Profiling:")
        stats.sort_stats('cumulative').print_stats('obspy.', 20)
        print(PSTATS_HELP)
    else:
        errors = run(argv, interactive)
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    # It is not possible to add the code of main directly to here.
    # This script is automatically installed with name obspy-runtests by
    # setup.py to the Scripts or bin directory of your Python distribution
    # setup.py needs a function to which it's scripts can be linked.
    errors = run(interactive=False)
    if errors:
        sys.exit(1)
