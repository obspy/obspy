# -*- coding: utf-8 -*-

import obspy
from obspy.core import util, utcdatetime, trace, stream
from obspy.core.tests import test_stream, test_utcdatetime, test_trace, \
    test_stats, test_waveform_plugins, test_preview, test_util
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    try:
        suite.addTest(doctest.DocTestSuite(util))
        suite.addTest(doctest.DocTestSuite(utcdatetime))
        suite.addTest(doctest.DocTestSuite(trace))
        suite.addTest(doctest.DocTestSuite(stream))
    except:
        pass
    suite.addTest(test_utcdatetime.suite())
    suite.addTest(test_stats.suite())
    suite.addTest(test_trace.suite())
    suite.addTest(test_stream.suite())
    suite.addTest(test_waveform_plugins.suite())
    suite.addTest(test_preview.suite())
    suite.addTest(test_util.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
