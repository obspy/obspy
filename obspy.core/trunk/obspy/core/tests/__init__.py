# -*- coding: utf-8 -*-

import obspy
from obspy.core import util, utcdatetime, trace, stream, __init__
from obspy.core.tests import test_stream, test_utcdatetime, test_trace, \
    test_stats, test_waveform_plugins, test_preview, test_util, test_ascii
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    for module in [util, utcdatetime, trace, stream, __init__]:
        try:
            suite.addTest(doctest.DocTestSuite(module))
        except:
            pass
    suite.addTest(test_utcdatetime.suite())
    suite.addTest(test_stats.suite())
    suite.addTest(test_trace.suite())
    suite.addTest(test_stream.suite())
    suite.addTest(test_waveform_plugins.suite())
    suite.addTest(test_preview.suite())
    suite.addTest(test_util.suite())
    suite.addTest(test_ascii.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
