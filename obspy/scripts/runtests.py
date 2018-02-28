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
    (https://www.gnu.org/copyleft/lesser.html)

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

(7) Report test results to https://tests.obspy.org/::

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
import traceback
import types
import unittest
import warnings
from argparse import ArgumentParser

import numpy as np
import requests

import obspy
from obspy.core.compatibility import urlparse
from obspy.core.util import ALL_MODULES, DEFAULT_MODULES, NETWORK_MODULES
from obspy.core.util.misc import MatplotlibBackend
from obspy.core.util.testing import MODULE_TEST_SKIP_CHECKS
from obspy.core.util.version import get_git_version


HARD_DEPENDENCIES = [
    "future", "numpy", "scipy", "matplotlib", "lxml.etree", "setuptools",
    "sqlalchemy", "decorator", "requests"]
OPTIONAL_DEPENDENCIES = [
    "flake8", "pyimgur", "pyproj", "pep8-naming", "m2crypto", "shapefile",
    "mpl_toolkits.basemap", "mock", "pyflakes", "geographiclib", "cartopy"]
DEPENDENCIES = HARD_DEPENDENCIES + OPTIONAL_DEPENDENCIES

PSTATS_HELP = """
Call "python -m pstats obspy.pstats" for an interactive profiling session.

The following commands will produce the same output as shown above:
  sort cumulative
  stats obspy. 20

Type "help" to see all available options.
"""

# Set legacy printing for numpy so the doctests work regardless of the numpy
# version.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass


HOSTNAME = platform.node().split('.', 1)[0]


def _get_suites(verbosity=1, names=[]):
    """
    The ObsPy test suite.
    """
    # Construct the test suite from the given names. Modules
    # need not be imported before in this case
    suites = {}
    ut = unittest.TestLoader()
    status = True
    import_failures = {}
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
        except Exception:
            status = False
            msg = (">>> Cannot import test suite for module obspy.%s due "
                   "to:" % name)
            msg += "\n" + "-" * len(msg)
            # Extract traceback from the exception.
            exc_info = sys.exc_info()
            stack = traceback.extract_stack()
            tb = traceback.extract_tb(exc_info[2])
            full_tb = stack[:-1] + tb
            exc_line = traceback.format_exception_only(*exc_info[:2])
            tb = "".join(traceback.format_list(full_tb))
            tb += "\n"
            tb += "".join(exc_line)
            info = msg + "\n" + tb
            import_failures[name] = info
            if verbosity:
                print(msg)
                print(tb)
        else:
            suites[name] = ut.suiteClass(suite)
    return suites, status, import_failures


def _create_report(ttrs, timetaken, log, server, hostname, sorted_tests,
                   ci_url=None, pr_url=None, import_failures=None):
    """
    If `server` is specified without URL scheme, 'https://' will be used as a
    default.
    """
    # import additional libraries here to speed up normal tests
    from future import standard_library
    with standard_library.hooks():
        import urllib.parse
    import codecs
    from xml.etree import ElementTree
    from xml.sax.saxutils import escape
    if import_failures is None:
        import_failures = {}
    timestamp = int(time.time())
    result = {'timestamp': timestamp}
    result['slowest_tests'] = [("%0.3fs" % dt, "%s" % desc)
                               for (desc, dt) in sorted_tests[:20]]
    result['timetaken'] = timetaken
    if log:
        try:
            data = codecs.open(log, 'r', encoding='UTF-8').read()
            result['install_log'] = escape(data)
        except Exception:
            print("Cannot open log file %s" % log)
    # get ObsPy module versions
    result['obspy'] = {}
    tests = 0
    errors = 0
    failures = 0
    skipped = 0
    try:
        installed = get_git_version()
    except Exception:
        installed = ''
    result['obspy']['installed'] = installed
    for module in sorted(ALL_MODULES):
        result['obspy'][module] = {}
        result['obspy'][module]['installed'] = installed
        # add a failed-to-import test module to report with an error
        if module in import_failures:
            result['obspy'][module]['timetaken'] = 0
            result['obspy'][module]['tested'] = True
            result['obspy'][module]['tests'] = 1
            # can't say how many tests would have been in that suite so just
            # leave 0
            result['obspy'][module]['skipped'] = 0
            result['obspy'][module]['failures'] = {}
            result['obspy'][module]['errors'] = {
                'f%s' % (errors): import_failures[module]}
            tests += 1
            errors += 1
            continue
        if module not in ttrs:
            continue
        # test results
        ttr = ttrs[module]
        result['obspy'][module]['timetaken'] = ttr.__dict__['timetaken']
        result['obspy'][module]['tested'] = True
        result['obspy'][module]['tests'] = ttr.testsRun
        skipped += len(ttr.skipped)
        result['obspy'][module]['skipped'] = len(ttr.skipped)
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
        if module == "pep8-naming":
            module_ = "pep8ext_naming"
        else:
            module_ = module
        temp = module_.split('.')
        try:
            mod = __import__(module_,
                             fromlist=[native_str(temp[1:])])
        except ImportError:
            version_ = '---'
        else:
            try:
                version_ = mod.__version__
            except AttributeError:
                version_ = '???'
        result['dependencies'][module] = version_
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
        except Exception:
            result['platform'][func] = ''
    # set node name to hostname if set
    result['platform']['node'] = hostname
    # post only the first part of the node name (only applies to MacOS X)
    try:
        result['platform']['node'] = result['platform']['node'].split('.')[0]
    except Exception:
        pass
    # test results
    result['tests'] = tests
    result['errors'] = errors
    result['failures'] = failures
    result['skipped'] = skipped
    # try to append info on skipped tests:
    result['skipped_tests_details'] = []
    try:
        for module, testresult_ in ttrs.items():
            if testresult_.skipped:
                for skipped_test, skip_message in testresult_.skipped:
                    result['skipped_tests_details'].append(
                        (module, skipped_test.__module__,
                         skipped_test.__class__.__name__,
                         skipped_test._testMethodName, skip_message))
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        print("\n".join(traceback.format_exception(exc_type, exc_value,
                                                   exc_tb)))
        result['skipped_tests_details'] = []

    if ci_url is not None:
        result['ciurl'] = ci_url
    if pr_url is not None:
        result['prurl'] = pr_url

    # generate XML document
    def _dict2xml(doc, result):
        for key, value in result.items():
            key = key.split('(')[0].strip()
            if isinstance(value, dict):
                child = ElementTree.SubElement(doc, key)
                _dict2xml(child, value)
            elif value is not None:
                if isinstance(value, (str, native_str)):
                    ElementTree.SubElement(doc, key).text = value
                elif isinstance(value, (str, native_str)):
                    ElementTree.SubElement(doc, key).text = str(value, 'utf-8')
                else:
                    ElementTree.SubElement(doc, key).text = str(value)
            else:
                ElementTree.SubElement(doc, key)
    root = ElementTree.Element("report")
    _dict2xml(root, result)
    xml_doc = ElementTree.tostring(root)
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
        'modules': len(ttrs) + len(import_failures),
        'xml': xml_doc
    })
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain"}
    url = server
    if not urlparse(url).scheme:
        url = "https://" + url
    response = requests.post(url=url, headers=headers,
                             data=params.encode('UTF-8'))
    # get the response
    if response.status_code == 200:
        report_url = response.json().get('url', server)
        print('Your test results have been reported and are available at: '
              '{}\nThank you!'.format(report_url))
    # handle errors
    else:
        print("Error: Could not sent a test report to %s." % (server))
        print(response.reason)


class _TextTestResult(unittest.TextTestResult):
    """
    A test result class that can print formatted text results to a stream.
    """
    timer = []

    def startTest(self, test):  # noqa
        self.start = time.time()
        super(_TextTestResult, self).startTest(test)

    def stopTest(self, test):  # noqa
        super(_TextTestResult, self).stopTest(test)
        self.timer.append((test, time.time() - self.start))


def _recursive_skip(test_suite, msg):
    """
    Helper method to recursively skip all tests aggregated in `test_suite`
    with the the specified message.

    :type test_suite: unittest.TestSuite
    :type msg: str
    :param msg: Reason for unconditionally skipping the tests.
    """
    def _custom_skip_test(test_case):
        test_case.skipTest(msg)

    if isinstance(test_suite, unittest.TestSuite):
        for obj in test_suite:
            _recursive_skip(obj, msg)
    elif isinstance(test_suite, unittest.TestCase):
        # overwrite setUp method
        test_suite.setUp = types.MethodType(_custom_skip_test, test_suite)
    else:
        raise NotImplementedError()


class _TextTestRunner(unittest.TextTestRunner):
    """
    A test runner class that displays results in textual form.
    """
    resultclass = _TextTestResult

    def __init__(self, timeit=False, *args, **kwargs):
        super(_TextTestRunner, self).__init__(*args, **kwargs)
        self.timeit = timeit

    def run(self, suites):
        """
        Run the given test case or test suite.
        """
        results = {}
        time_taken = 0
        keys = sorted(suites.keys())
        for key in keys:
            test = suites[key]
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
            result = self._makeResult()
            start = time.time()
            test(result)
            stop = time.time()
            results[key] = result
            total = stop - start
            results[key].__dict__['timetaken'] = total
            if self.timeit:
                self.stream.writeln('')
                self.stream.write("obspy.%s: " % (key))
                num = test.countTestCases()
                try:
                    avg = float(total) / num
                except Exception:
                    avg = 0
                msg = '%d tests in %.3fs (average of %.4fs per test)'
                self.stream.writeln(msg % (num, total, avg))
                self.stream.writeln('')
            time_taken += total
        runs = 0
        faileds = 0
        erroreds = 0
        was_successful = True
        if self.verbosity:
            self.stream.writeln()
        for result in results.values():
            failed, errored = map(len, (result.failures, result.errors))
            faileds += failed
            erroreds += errored
            if not result.wasSuccessful():
                was_successful = False
                result.printErrors()
            runs += result.testsRun
        if self.verbosity:
            self.stream.writeln(unittest.TextTestResult.separator2)
            self.stream.writeln("Ran %d test%s in %.3fs" %
                                (runs, runs != 1 and "s" or "", time_taken))
            self.stream.writeln()
        if not was_successful:
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


def run_tests(verbosity=1, tests=None, report=False, log=None,
              server="tests.obspy.org", test_all_modules=False, timeit=False,
              interactive=False, slowest=0, exclude=[], tutorial=False,
              hostname=HOSTNAME, ci_url=None, pr_url=None):
    """
    This function executes ObsPy test suites.

    :type verbosity: int, optional
    :param verbosity: Run tests in verbose mode (``0``=quiet, ``1``=normal,
        ``2``=verbose, default is ``1``).
    :type tests: list of str
    :param tests: List of submodules for which test suites should be run
        (e.g. ``['io.mseed', 'io.sac']``).  If no suites are specified, all
        non-networking submodules' test suites will be run.
    :type report: bool, optional
    :param report: Submits a test report if enabled (default is ``False``).
    :type log: str, optional
    :param log: Filename of install log file to append to report.
    :type server: str, optional
    :param server: Report server URL (default is ``"tests.obspy.org"``).
    """
    if tests is None:
        tests = []
    print("Running {}, ObsPy version '{}'".format(__file__, obspy.__version__))
    if test_all_modules:
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
    suites, status, import_failures = _get_suites(verbosity, tests)
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
        except Exception:
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
        if 'y' in var:
            report = True
    if report:
        _create_report(ttr, total_time, log, server, hostname, sorted_tests,
                       ci_url, pr_url, import_failures)
    # make obspy-runtests exit with 1 if a test suite could not be added,
    # indicating failure
    if status is False:
        errors += 1
    if errors:
        return errors


def run(argv=None, interactive=True):
    MatplotlibBackend.switch_backend("AGG", sloppy=False)
    parser = ArgumentParser(prog='obspy-runtests',
                            description='A command-line program that runs all '
                                        'ObsPy tests.')
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + get_git_version())
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='quiet mode')
    parser.add_argument('--raise-all-warnings', action='store_true',
                        help='All warnings are raised as exceptions when this '
                             'flag is set. Only for debugging purposes.')

    # filter options
    filter = parser.add_argument_group('Module Filter',
                                       'Providing no modules will test all '
                                       'ObsPy modules which do not require an '
                                       'active network connection.')
    filter.add_argument('-a', '--all', action='store_true',
                        dest='test_all_modules',
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
    report.add_argument('--ci-url', default=None, dest="ci_url",
                        help='URL to Continuous Integration job page.')
    report.add_argument('--pr-url', default=None,
                        dest="pr_url", help='Github (Pull Request) URL.')

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
        np.seterr(all='warn')
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
    # whether to raise any warning that's appearing
    if args.raise_all_warnings:
        # raise all NumPy warnings
        np.seterr(all='raise')
        # raise user and deprecation warnings
        warnings.simplefilter("error", UserWarning)
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

    # All arguments are used by the test runner and should not interfere
    # with any other module that might also parse them, e.g. flake8.
    sys.argv = sys.argv[:1]

    return run_tests(verbosity, args.tests, report, args.log, args.server,
                     args.test_all_modules, args.timeit, interactive, args.n,
                     exclude=args.exclude, tutorial=args.tutorial,
                     hostname=args.hostname, ci_url=args.ci_url,
                     pr_url=args.pr_url)


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
        Profile.run('from obspy.scripts.runtests import run; run()',
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
