# -*- coding: utf-8 -*-

from obspy.db.tests import test_util
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_util.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
