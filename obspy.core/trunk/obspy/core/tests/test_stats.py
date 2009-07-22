# -*- coding: utf-8 -*-

from obspy.core import Stats
import copy
import unittest


class StatsTestCase(unittest.TestCase):
    """
    Test suite for L{obspy.core.Stats}.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_deepcopy(self):
        """
        Tests initialization from a given time string.
        """
        stats = Stats()
        stats.network = 'BW'
        stats['station'] = 'ROTZ'
        x = stats.keys()
        x.sort()
        y = copy.deepcopy(x)[0:3]
        self.assertEquals(y, ['network', 'station'])


def suite():
    return unittest.makeSuite(StatsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
