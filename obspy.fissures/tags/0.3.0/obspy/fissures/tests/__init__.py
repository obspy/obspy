# -*- coding: utf-8 -*-

from obspy.fissures.tests import test_client
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_client.suite())
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
