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
        self.assertEquals(catalog[0].id,
                          'smi:www.iris.edu/ws/event/query?eventId=3279407')
        self.assertEquals(catalog[1].id,
                          'smi:www.iris.edu/ws/event/query?eventId=2318174')
        # neries
        catalog = readQuakeML(self.neries_xml)
        self.assertEquals(len(catalog), 3)
        self.assertEquals(catalog[0].id,
                          'quakeml:eu.emsc/event/20120404_0000041')
        self.assertEquals(catalog[1].id,
                          'quakeml:eu.emsc/event/20120404_0000038')
        self.assertEquals(catalog[2].id,
                          'quakeml:eu.emsc/event/20120404_0000039')


def suite():
    return unittest.makeSuite(QuakeMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
