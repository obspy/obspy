# -*- coding: utf-8 -*-

from obspy.signal import invsim, trigger
from obspy.signal.tests import test_invsim, test_filter, test_rotate, \
    test_trigger
import doctest
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_invsim.suite())
    try:
        suite.addTest(doctest.DocTestSuite(invsim))
        suite.addTest(doctest.DocTestSuite(trigger))
    except:
        pass
    suite.addTest(test_filter.suite())
    suite.addTest(test_rotate.suite())
    suite.addTest(test_trigger.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
