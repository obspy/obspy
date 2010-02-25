# -*- coding: utf-8 -*-

from obspy.xseed.tests import test_blockettes, test_fields, test_utils, \
    test_parser
import unittest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_blockettes.suite())
    suite.addTest(test_utils.suite())
    suite.addTest(test_fields.suite())
    suite.addTest(test_parser.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

