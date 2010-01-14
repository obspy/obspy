#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
import sys
import time
import unittest


DEFAULT_MODULES = ['core', 'gse2', 'mseed', 'sac', 'wav', 'signal', 'imaging',
                   'xseed', 'seisan']



def suite(options, modules=DEFAULT_MODULES):
    """
    The obspy test suite.
    """
    suite = unittest.TestSuite()
    for id in modules:
        if not '.' in id:
            module = 'obspy.%s.tests' % id
        else:
            module = id
        try:
            __import__(module)
        except ImportError, e:
            print "Cannot import test suite of module %s" % module
            print e
            time.sleep(0.5)
        else:
            suite.addTests(sys.modules[module].suite())
    return suite


def runTests(options, *args):
    """
    This function runs all available tests in obspy, from python
    """
    if options.verbose:
        unittest.TextTestRunner(verbosity=2).run(suite(options, *args))
    else:
        unittest.TextTestRunner(verbosity=1).run(suite(options, *args))


def main():
    usage = "usage: %prog [options] modules"
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose", default=False,
                      action="store_true", dest="verbose",
                      help="verbose mode")
    parser.add_option("-a", "--all", default=False,
                      action="store_true", dest="all",
                      help="Test all modules")
    (options, _) = parser.parse_args()
    if options.all:
        runTests(options)
        return
    if len(parser.largs) == 0:
        parser.print_help()
        return
    runTests(options, parser.largs)


if __name__ == "__main__":
    main()
