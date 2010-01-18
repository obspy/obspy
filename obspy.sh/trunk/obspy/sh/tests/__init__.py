# -*- coding: utf-8 -*-

import obspy, unittest
from obspy.sh.tests import test_asc


def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_asc.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
