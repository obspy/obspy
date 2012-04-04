# -*- coding: utf-8 -*-
from obspy.core.quakeml import readQuakeML
import os
import unittest


class QuakeMLTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.quakeml
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

    def test_readQuakeML(self):
        """
        """
        # iris
        catalog = readQuakeML(self.iris_xml)
        self.assertEquals(len(catalog), 2)
        # print catalog
        # neries
        catalog = readQuakeML(self.neries_xml)
        self.assertEquals(len(catalog), 3)
        # print catalog


def suite():
    return unittest.makeSuite(QuakeMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
