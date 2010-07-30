# -*- coding: utf-8 -*-

from obspy.signal import invsim, trigger, util, __init__
from obspy.signal.tests import test_invsim, test_freqattributes, test_filter, \
    test_rotate, test_trigger, test_util, test_cpxtrace, test_hoctavbands, \
    test_polarization
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_invsim.suite())
    for module in [__init__, invsim, trigger, util]:
        try:
            suite.addTest(doctest.DocTestSuite(module))
        except:
            pass
    suite.addTest(test_filter.suite())
    suite.addTest(test_rotate.suite())
    suite.addTest(test_trigger.suite())
    suite.addTest(test_util.suite())
    suite.addTest(test_cpxtrace.suite())
    suite.addTest(test_freqattributes.suite())
    suite.addTest(test_hoctavbands.suite())
    suite.addTest(test_polarization.suite())
    suite.addTest(test_invsim.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
