# -*- coding: utf-8 -*-

from obspy.imaging.tests import test_beachball
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_beachball.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')