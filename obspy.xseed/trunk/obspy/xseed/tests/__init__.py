# -*- coding: utf-8 -*-

import unittest

from obspy.xseed.tests import test_utils
from obspy.xseed.tests import test_fields
from obspy.xseed.tests import test_blockettes


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_blockettes.suite())
    suite.addTest(test_utils.suite())
    suite.addTest(test_fields.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

