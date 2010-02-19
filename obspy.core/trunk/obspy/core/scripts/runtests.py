#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy Test Suite Module.

All tests in ObsPy are located in the tests directory of the each specific
module. The __init__.py of the tests directory itself as well as every test
file located in the tests directory has a function called suite, which is
executed using this script. Running the script with the verbose keyword exposes
the names of all available test cases.

Examples
--------
(1) Run all tests::

    python runtests.py

    or

    >>> import obspy.core
    >>> obspy.core.runTests()  # DOCTEST: +SKIP

(2) Verbose output::

    python runtests.py -v

    or

    >>> import obspy.core
    >>> obspy.core.runTests(verbosity=2)"  # DOCTEST: +SKIP

(3) Run tests of module :mod:`obspy.mseed`::

    python runtests.py obspy.mseed.tests.suite 

    or as shortcut::

    python runtests.py mseed

(4) Run a certain test case::

    python runtests.py obspy.core.tests.test_stats.StatsTestCase.test_init

    or

    >>> import obspy.core
    >>> tests = ['obspy.core.tests.test_stats.StatsTestCase.test_init']
    >>> obspy.core.runTests(verbosity=2, tests=tests)  # DOCTEST: +SKIP
"""

from optparse import OptionParser
import unittest
import os


DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan', 'sh']
ALL_MODULES = DEFAULT_MODULES + ['fissures', 'arclink', 'seishub']
DEPENDENCIES = ['numpy', 'scipy', 'matplotlib', 'lxml.etree', '_omnipy']


def _getSuite(verbosity=1, tests=[]):
    """
    The obspy test suite.
    """
    if tests == []:
        names = ['obspy.%s.tests.suite' % d for d in DEFAULT_MODULES]
    else:
        names = []
        # Search for short cuts in tests, if there are no short cuts,
        # names variables is equal to tests variable
        for test in tests:
            if test in ALL_MODULES:
                test = 'obspy.%s.tests.suite' % test
            names.append(test)
    # Construct the test suite from the given names. Modules
    # need not be imported before in this case
    suites = []
    ut = unittest.TestLoader()
    for name in names:
        try:
            suites.append(ut.loadTestsFromName(name, None))
        except Exception, e:
            if verbosity:
                print e
                print "Cannot import test suite for module %s" % name
    return ut.suiteClass(suites)


def _createReport(ttr, log, server):
    # import additional libraries here to speed up normal tests
    import httplib
    import urllib
    from urlparse import urlparse
    import time
    import platform
    from xml.etree import ElementTree as etree
    timestamp = int(time.time())
    result = {'timestamp':timestamp}
    if log:
        try:
            result['install_log': open(log, 'r').read()]
        except:
            print "Cannot open log file"
    # get ObsPy module versions
    result['obspy'] = {}
    for module in ALL_MODULES:
        try:
            mod = __import__('obspy.' + module, fromlist='obspy')
            result['obspy'][module] = mod.__version__
        except:
            result['obspy'][module] = ''
    # get dependencies
    result['dependencies'] = {}
    for module in DEPENDENCIES:
        temp = module.split('.')
        try:
            mod = __import__(module, fromlist=temp[1:])
            if module == '_omnipy':
                result['dependencies'][module] = mod.coreVersion()
            else:
                result['dependencies'][module] = mod.__version__
        except:
            result['dependencies'][module] = ''
    # get system / environment settings
    result['platform'] = {}
    for func in ['system', 'node', 'release', 'version', 'machine',
                 'processor', 'python_version', 'python_implementation',
                 'python_compiler', 'architecture']:
        try:
            temp = getattr(platform, func)()
            if isinstance(temp, tuple):
                temp = temp[0]
            result['platform'][func] = temp
        except:
            result['platform'][func] = ''
    # test results
    result['errors'] = {}
    for method, text in ttr.errors:
        result['errors'][str(method)] = text
    result['failures'] = {}
    for method, text in ttr.failures:
        result['errors'][str(method)] = text
    # generate XML document
    root = etree.Element("report")
    for key, value in result.iteritems():
        if isinstance(value, dict):
            child = etree.SubElement(root, key)
            for subkey, subvalue in value.iteritems():
                subkey = subkey.split('(')[0].strip()
                if subvalue:
                    etree.SubElement(child, subkey).text = str(subvalue)
                else:
                    etree.SubElement(child, subkey)
        elif value:
            etree.SubElement(root, key).text = str(value)
        else:
            etree.SubElement(root, key)
    xml_doc = etree.tostring(root, "UTF-8")
    print
    # send result to report server
    params = urllib.urlencode({
        'timestamp': timestamp,
        'system': result['platform']['system'],
        'python_version': result['platform']['python_version'],
        'architecture': result['platform']['architecture'],
        'failures': len(ttr.failures),
        'errors': len(ttr.errors),
        'xml': xml_doc
    })
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain"}
    conn = httplib.HTTPConnection(server)
    conn.request("POST", "/", params, headers)
    # get the response
    response = conn.getresponse()
    # handle redirect
    if response.status == 301:
        o = urlparse(response.msg['location'])
        conn = httplib.HTTPConnection(o.netloc)
        conn.request("POST", o.path, params, headers)
        # get the response
        response = conn.getresponse()
    # handle errors
    if response.status == 200:
        print "Test report has been sent to %s." % (server)
    else:
        print "Error: Could not sent a test report to %s." % (server)
        print response.reason


def runTests(verbosity=1, tests=[], report=False, log=None,
             server="tests.obspy.org"):
    """
    This function executes ObsPy test suites.

    Parameters
    ----------
    verbosity : [ 0 | 1 | 2 ], optional
        Run tests in verbose mode (0=quiet, 1=normal, 2=verbose, default is 1).
    tests : list of strings, optional
        Test suites to run. If no suite is given all installed tests suites
        will be started (default is a empty list).
        Example ['obspy.core.tests.suite']
    report : boolean, optional
        Submits a test report if enabled (default is False).
    log : string, optional
        Filename of install log file to append to report
    server : string, optional
        Report server URL (default is "tests.obspy.org").
    """
    suite = _getSuite(verbosity, tests)
    ttr = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    if report:
        _createReport(ttr, log, server)


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
    parser.add_option("-r", "--report", default=False,
                      action="store_true", dest="report",
                      help="submit a test report")
    parser.add_option("-u", "--server", default="tests.obspy.org",
                      type="string", dest="server",
                      help="report server (default is tests.obspy.org)")
    parser.add_option("-l", "--log", default=None,
                      type="string", dest="log",
                      help="append log file to test report")
    (options, _) = parser.parse_args()
    # set correct verbosity level
    if options.verbose:
        verbosity = 2
    elif options.quiet:
        verbosity = 0
    else:
        verbosity = 1
    # check for send report option or environmental settings
    if options.report or 'OBSPY_REPORT' in os.environ.keys():
        report = True
    else:
        report = False
    if 'OBSPY_REPORT_SERVER' in os.environ.keys():
        options.server = os.environ['OBSPY_REPORT_SERVER']
    runTests(verbosity, parser.largs, report, options.log, options.server)


if __name__ == "__main__":
    # It is not possible to add the code of main directly to here.
    # This script is automatically installed with name obspy-runtests by
    # setup.py to the Scripts or bin directory of your Python distribution
    # setup.py needs a function to which it's scripts can be linked.
    main()
