# -*- coding: utf-8 -*-

import obspy.core
from obspy.core.tests import test_stream, test_utcdatetime, test_trace, \
    test_stats, test_waveform_plugins, test_preview, test_util, test_ascii
import doctest
import unittest
import warnings


def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(obspy.core))
    for module in [obspy.core, obspy.core.util, obspy.core.utcdatetime,
                   obspy.core.trace, obspy.core.stream]:
        try:
            suite.addTest(doctest.DocTestSuite(module))
        except Exception, e:
            warnings.warn(str(e))
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
