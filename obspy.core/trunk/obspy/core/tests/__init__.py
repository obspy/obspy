# -*- coding: utf-8 -*-

from obspy.core import util, stats, utcdatetime
from obspy.core.tests import test_stream, test_utcdatetime, test_trace, \
    test_stats
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    try:
        suite.addTest(doctest.DocTestSuite(util))
        suite.addTest(doctest.DocTestSuite(stats))
        suite.addTest(doctest.DocTestSuite(utcdatetime))
    except:
        pass
    suite.addTest(test_utcdatetime.suite())
    suite.addTest(test_stats.suite())
    suite.addTest(test_trace.suite())
    suite.addTest(test_stream.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
