# -*- coding: utf-8 -*-

from StringIO import StringIO
from obspy.core import Stats, Stream, Trace
import copy
import pickle
import unittest


class StatsTestCase(unittest.TestCase):
    """
    Test suite for L{obspy.core.Stats}.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        """
        Init tests.
        """
        stats = Stats({'test': 'muh'})
        self.assertEquals(stats.test, 'muh')
        self.assertEquals(stats['test'], 'muh')
        self.assertTrue('test' in stats)
        self.assertTrue('test' in stats.__dict__)

    def test_deepcopy(self):
        """
        Tests deepcopy method from L{obspy.core.Stats}.
        """
        stats = Stats()
        stats.network = 'BW'
        stats['station'] = 'ROTZ'
        stats2 = copy.deepcopy(stats)
        stats.network = 'CZ'
        stats.station = 'RJOB'
        self.assertEquals(stats2.network, 'BW')
        self.assertEquals(stats2.station, 'ROTZ')
        self.assertEquals(stats.network, 'CZ')
        self.assertEquals(stats.station, 'RJOB')

    def test_update(self):
        """
        Tests update method of AttribDict.
        """
        x = Stats({'a': 5})
        self.assertTrue('a' in dir(x))
        x.update({'b': 5})
        self.assertTrue('b' in dir(x))
        y = {'a':5}
        y.update({'b':5})
        x = Stats(y)
        self.assertTrue('b' in dir(x))

    def test_simpleStats(self):
        """
        Various setter and getter tests.
        """
        stats = Stats()
        stats.test = 1
        self.assertEquals(stats.test, 1)
        self.assertEquals(stats['test'], 1)
        stats['test2'] = 2
        self.assertEquals(stats.test2, 2)
        self.assertEquals(stats['test2'], 2)
        stats['test'] = 2
        self.assertEquals(stats.test, 2)
        self.assertEquals(stats['test'], 2)
        stats.test2 = 1
        self.assertEquals(stats.test2, 1)
        self.assertEquals(stats['test2'], 1)

    def test_nestedStats(self):
        """
        Various setter and getter tests.
        """
        stats = Stats()
        stats.test = Stats()
        stats.test['test2'] = 'muh'
        self.assertEquals(stats.test.test2, 'muh')
        self.assertEquals(stats.test['test2'], 'muh')
        self.assertEquals(stats['test'].test2, 'muh')
        self.assertEquals(stats['test']['test2'], 'muh')
        stats.test['test2'] = 'maeh'
        self.assertEquals(stats.test.test2, 'maeh')
        self.assertEquals(stats.test['test2'], 'maeh')
        self.assertEquals(stats['test'].test2, 'maeh')
        self.assertEquals(stats['test']['test2'], 'maeh')

    def test_bugfix_setStats(self):
        """
        Test related to issue #4.
        """
        st = Stream([Trace()])
        st += st
        # change stats attributes
        st[0].stats.station = 'AAA'
        st[1].stats['station'] = 'BBB'
        self.assertEquals(st[0].stats.station, 'AAA')
        self.assertEquals(st[0].stats['station'], 'AAA')
        self.assertEquals(st[1].stats['station'], 'BBB')
        self.assertEquals(st[1].stats.station, 'BBB')

    def test_bugfix_setStats2(self):
        """
        Second test related to issue #4.
        """
        st = Stream([Trace([], {'station': 'BGLD'})])
        self.assertEquals(st[0].stats.station, 'BGLD')
        self.assertEquals(st[0].stats['station'], 'BGLD')
        st[0].stats.station = 'AAA'
        self.assertEquals(st[0].stats.station, 'AAA')
        self.assertEquals(st[0].stats['station'], 'AAA')
        st = st + st
        self.assertEquals(st[0].stats.station, 'AAA')
        self.assertEquals(st[0].stats['station'], 'AAA')
        st[0].stats.station = 'BBB'
        self.assertEquals(st[0].stats.station, 'BBB')
        self.assertEquals(st[0].stats['station'], 'BBB')

    def test_bugfix_setStats3(self):
        """
        Third test related to issue #4.
        """
        st = Stream([Trace([], {'station': 'BGLD'})])
        self.assertEquals(st[0].stats.station, 'BGLD')
        st = st + st
        st[0].stats.station = 'AAA'
        st = st + st
        st[3].stats.station = 'BBB'
        for key, value in {0:'AAA', 1:'BGLD', 2:'AAA', 3:'BBB'}.iteritems():
            self.assertEquals(st[key].stats.station, value)
            self.assertEquals(st[key].stats['station'], value)
            self.assertEquals(st[key].stats.get('station'), value)
            self.assertTrue(value in st[key].stats.values())

    def test_pickleStats(self):
        """
        Pickling L{obspy.core.Stats} objects. Test case for issue #10
        """
        stats = Stats()
        stats.muh = 1
        stats['maeh'] = 'hallo'
        picklestring = pickle.dumps(stats)
        fp = StringIO(picklestring)
        stats2 = pickle.load(fp)
        self.assertEquals(stats, stats2)


def suite():
    return unittest.makeSuite(StatsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
