# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats
from obspy.core.util import AttribDict
import copy
import pickle
import unittest
import warnings


class StatsTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.Stats.
    """

    def test_init(self):
        """
        Init tests.
        """
        stats = Stats({'test': 'muh'})
        stats['other1'] = {'test1': '1'}
        stats['other2'] = AttribDict({'test2': '2'})
        stats['other3'] = 'test3'
        self.assertEqual(stats.test, 'muh')
        self.assertEqual(stats['test'], 'muh')
        self.assertEqual(stats.other1.test1, '1')
        self.assertEqual(stats.other1.__class__, AttribDict)
        self.assertEqual(len(stats.other1), 1)
        self.assertEqual(stats.other2.test2, '2')
        self.assertEqual(stats.other2.__class__, AttribDict)
        self.assertEqual(len(stats.other2), 1)
        self.assertEqual(stats.other3, 'test3')
        self.assertTrue('test' in stats)
        self.assertTrue('test' in stats.__dict__)

    def test_deepcopy(self):
        """
        Tests deepcopy method of Stats object.
        """
        stats = Stats()
        stats.network = 'BW'
        stats['station'] = 'ROTZ'
        stats['other1'] = {'test1': '1'}
        stats['other2'] = AttribDict({'test2': '2'})
        stats['other3'] = 'test3'
        stats2 = copy.deepcopy(stats)
        stats.network = 'CZ'
        stats.station = 'RJOB'
        self.assertEqual(stats2.__class__, Stats)
        self.assertEqual(stats2.network, 'BW')
        self.assertEqual(stats2.station, 'ROTZ')
        self.assertEqual(stats2.other1.test1, '1')
        self.assertEqual(stats2.other1.__class__, AttribDict)
        self.assertEqual(len(stats2.other1), 1)
        self.assertEqual(stats2.other2.test2, '2')
        self.assertEqual(stats2.other2.__class__, AttribDict)
        self.assertEqual(len(stats2.other2), 1)
        self.assertEqual(stats2.other3, 'test3')
        self.assertEqual(stats.network, 'CZ')
        self.assertEqual(stats.station, 'RJOB')

    def test_update(self):
        """
        Tests update method of Stats object.
        """
        x = Stats({'a': 5})
        self.assertTrue('a' in dir(x))
        x.update({'b': 5})
        self.assertTrue('b' in dir(x))
        y = {'a': 5}
        y.update({'b': 5})
        x = Stats(y)
        self.assertTrue('b' in dir(x))

    def test_simpleStats(self):
        """
        Various setter and getter tests.
        """
        stats = Stats()
        stats.test = 1
        self.assertEqual(stats.test, 1)
        self.assertEqual(stats['test'], 1)
        stats['test2'] = 2
        self.assertEqual(stats.test2, 2)
        self.assertEqual(stats['test2'], 2)
        stats['test'] = 2
        self.assertEqual(stats.test, 2)
        self.assertEqual(stats['test'], 2)
        stats.test2 = 1
        self.assertEqual(stats.test2, 1)
        self.assertEqual(stats['test2'], 1)

    def test_nestedStats(self):
        """
        Various setter and getter tests.
        """
        # 1
        stats = Stats()
        stats.test = dict()
        stats.test['test2'] = 'muh'
        self.assertEqual(stats.test.test2, 'muh')
        self.assertEqual(stats.test['test2'], 'muh')
        self.assertEqual(stats['test'].test2, 'muh')
        self.assertEqual(stats['test']['test2'], 'muh')
        stats.test['test2'] = 'maeh'
        self.assertEqual(stats.test.test2, 'maeh')
        self.assertEqual(stats.test['test2'], 'maeh')
        self.assertEqual(stats['test'].test2, 'maeh')
        self.assertEqual(stats['test']['test2'], 'maeh')
        # 2 - multiple initialization
        stats = Stats({'muh': 'meah'})
        stats2 = Stats(Stats(Stats(stats)))
        self.assertEqual(stats2.muh, 'meah')
        # 3 - check conversion to AttribDict
        stats = Stats()
        stats.sub1 = {'muh': 'meah'}
        stats.sub2 = AttribDict({'muh2': 'meah2'})
        stats2 = Stats(stats)
        self.assertTrue(isinstance(stats.sub1, AttribDict))
        self.assertTrue(isinstance(stats.sub2, AttribDict))
        self.assertEqual(stats2.sub1.muh, 'meah')
        self.assertEqual(stats2.sub2.muh2, 'meah2')

    def test_bugfix_setStats(self):
        """
        Test related to issue #4.
        """
        st = Stream([Trace()])
        st += st
        # change stats attributes
        st[0].stats.station = 'AAA'
        st[1].stats['station'] = 'BBB'
        self.assertEqual(st[0].stats.station, 'BBB')
        self.assertEqual(st[0].stats['station'], 'BBB')
        self.assertEqual(st[1].stats['station'], 'BBB')
        self.assertEqual(st[1].stats.station, 'BBB')

    def test_bugfix_setStats2(self):
        """
        Second test related to issue #4.
        """
        st = Stream([Trace(header={'station': 'BGLD'})])
        self.assertEqual(st[0].stats.station, 'BGLD')
        self.assertEqual(st[0].stats['station'], 'BGLD')
        st[0].stats.station = 'AAA'
        self.assertEqual(st[0].stats.station, 'AAA')
        self.assertEqual(st[0].stats['station'], 'AAA')
        st = st + st
        self.assertEqual(st[0].stats.station, 'AAA')
        self.assertEqual(st[0].stats['station'], 'AAA')
        st[0].stats.station = 'BBB'
        self.assertEqual(st[0].stats.station, 'BBB')
        self.assertEqual(st[0].stats['station'], 'BBB')

    def test_bugfix_setStats3(self):
        """
        Third test related to issue #4.
        """
        st = Stream([Trace(header={'station': 'BGLD'})])
        self.assertEqual(st[0].stats.station, 'BGLD')
        st = st + st
        st[0].stats.station = 'AAA'
        st = st + st
        st[3].stats.station = 'BBB'
        # changed in rev. 1625: adding streams doesn't deepcopy
        # therefore all traces in the test stream are identical
        # (python list behavior)
        for tr in st:
            self.assertTrue(tr == st[0])
            self.assertEqual(tr.stats.station, 'BBB')
            self.assertEqual(tr.stats['station'], 'BBB')
            self.assertEqual(tr.stats.get('station'), 'BBB')
            self.assertTrue('BBB' in tr.stats.values())

    def test_pickleStats(self):
        """
        Test pickling Stats objects. Test case for issue #10.
        """
        stats = Stats()
        stats.muh = 1
        stats['maeh'] = 'hallo'
        # ASCII
        temp = pickle.dumps(stats, protocol=0)
        stats2 = pickle.loads(temp)
        self.assertEqual(stats, stats2)
        # old binary
        temp = pickle.dumps(stats, protocol=1)
        stats2 = pickle.loads(temp)
        self.assertEqual(stats, stats2)
        # new binary
        temp = pickle.dumps(stats, protocol=2)
        stats2 = pickle.loads(temp)
        self.assertEqual(stats, stats2)

    def test_setCalib(self):
        """
        Test to prevent setting a calibration factor of 0
        """
        x = Stats()
        # this should work
        x.update({'calib': 1.23})
        self.assertTrue(x.calib, 1.23)
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            # 1
            self.assertRaises(UserWarning, x.__setitem__, 'calib', 0)
            # 2
            self.assertRaises(UserWarning, x.update, {'calib': 0})
        # calib value should nevertheless be set to 0
        self.assertTrue(x.calib, 0)

    def test_compare_with_dict(self):
        """
        Checks if Stats is still comparable to a dict object.
        """
        adict = {
            'network': '', 'sampling_rate': 1.0, 'test': 1, 'station': '',
            'location': '', 'starttime': UTCDateTime(1970, 1, 1, 0, 0),
            'delta': 1.0, 'calib': 1.0, 'npts': 0,
            'endtime': UTCDateTime(1970, 1, 1, 0, 0), 'channel': ''}
        ad = Stats(adict)
        self.assertEqual(ad, adict)
        self.assertEqual(adict, ad)


def suite():
    return unittest.makeSuite(StatsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
