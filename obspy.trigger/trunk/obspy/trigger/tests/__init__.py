# -*- coding: utf-8 -*-

from obspy.trigger import trigger
from obspy.trigger.tests import test_trigger
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    try:
        suite.addTest(doctest.DocTestSuite(trigger))
    except:
        pass
    suite.addTest(test_trigger.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
