# -*- coding: utf-8 -*-

from obspy.seishub.tests import test_client
from obspy.seishub import client
import unittest
import doctest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(client))
    suite.addTest(test_client.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
