# -*- coding: utf-8 -*-

from obspy.wav.tests import test_core
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_core.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
