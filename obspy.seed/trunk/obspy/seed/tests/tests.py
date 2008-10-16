# -*- coding: utf-8 -*-

import doctest
import unittest

from obspy.seed.tests import test_utils
from obspy.seed.tests import test_fields


def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocFileSuite('test_blockette010.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette011.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette012.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette030.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette031.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette033.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette034.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette050.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette051.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette052.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette053.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette054.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette057.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette058.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette061.txt'))
    suite.addTest(test_utils.suite())
    suite.addTest(test_fields.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')