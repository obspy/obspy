# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import pickle
import unittest
import warnings

try:
    import IPython  # NOQA
except ImportError:
    HAS_IPYTHON_PRETTY = False
    IPYTHON_IMPORT_MSG = 'IPython not installed'
else:
    try:
        from IPython.lib.pretty import pretty
    except ImportError:
        HAS_IPYTHON_PRETTY = False
        IPYTHON_IMPORT_MSG = 'Could not import IPython.lib.pretty.pretty'
    else:
        HAS_IPYTHON_PRETTY = True
        IPYTHON_IMPORT_MSG = ''

from obspy import Stream, Trace, UTCDateTime
from obspy.core import Stats, read
from obspy.core.util import AttribDict


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
        self.assertIn('test', stats)
        self.assertIn('test', stats.__dict__)

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
        self.assertIn('a', dir(x))
        x.update({'b': 5})
        self.assertIn('b', dir(x))
        y = {'a': 5}
        y.update({'b': 5})
        x = Stats(y)
        self.assertIn('b', dir(x))

    def test_simple_stats(self):
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

    def test_nested_stats(self):
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

    def test_bugfix_set_stats(self):
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

    def test_bugfix_set_stats_2(self):
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

    def test_bugfix_set_stats_3(self):
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
            self.assertEqual(tr, st[0])
            self.assertEqual(tr.stats.station, 'BBB')
            self.assertEqual(tr.stats['station'], 'BBB')
            self.assertEqual(tr.stats.get('station'), 'BBB')
            self.assertIn('BBB', tr.stats.values())

    def test_pickle_stats(self):
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

    def test_set_calib(self):
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

    @unittest.skipIf(not HAS_IPYTHON_PRETTY, IPYTHON_IMPORT_MSG)
    def test_repr_pretty(self):
        """
        Test _repr_pretty_ method of Stats and underlying AttribDict.
        """
        # we test two different stats/attribdict objects to make sure it works
        # when supplying objects with different keys in them (as a regression
        # test for previous commit d55c7221fe4b4a983ed3551bd18589f23c0b22c4)
        st = read()
        expected_stats = [
            "        network: 'BW'",
            "        station: 'RJOB'",
            "       location: ''",
            "        channel: 'EHZ'",
            '      starttime: 2009-08-24T00:20:03.000000Z',
            '        endtime: 2009-08-24T00:20:32.990000Z',
            '  sampling_rate: 100.0',
            '          delta: 0.01',
            '           npts: 3000',
            '          calib: 1.0',
            '   back_azimuth: 100.0',
            '    inclination: 30.0',
            '       response: Channel Response',
            '                 \tFrom M/S (Velocity in Meters Per Second) to '
            'COUNTS (Digital Counts)',
            '                 \tOverall Sensitivity: 2.5168e+09 defined at '
            '0.020 Hz',
            '                 \t4 stages:',
            '                 \t\tStage 1: PolesZerosResponseStage from M/S '
            'to V, gain: 1500',
            '                 \t\tStage 2: CoefficientsTypeResponseStage from '
            'V to COUNTS, gain: 1.67785e+06',
            '                 \t\tStage 3: FIRResponseStage from COUNTS to '
            'COUNTS, gain: 1',
            '                 \t\tStage 4: FIRResponseStage from COUNTS to '
            'COUNTS, gain: 1',
            ]
        expected_attribdict = [
            "AttribDict({'back_azimuth': 100.0,",
            "            'calib': 1.0,",
            "            'channel': 'EHZ',",
            "            'delta': 0.01,",
            "            'endtime': 2009-08-24T00:20:32.990000Z,",
            "            'inclination': 30.0,",
            "            'location': '',",
            "            'network': 'BW',",
            "            'npts': 3000,",
            "            'response': Channel Response",
            '                      \tFrom M/S (Velocity in Meters Per Second) '
            'to COUNTS (Digital Counts)',
            '                      \tOverall Sensitivity: 2.5168e+09 defined '
            'at 0.020 Hz',
            '                      \t4 stages:',
            '                      \t\tStage 1: PolesZerosResponseStage from '
            'M/S to V, gain: 1500',
            '                      \t\tStage 2: CoefficientsTypeResponseStage '
            'from V to COUNTS, gain: 1.67785e+06',
            '                      \t\tStage 3: FIRResponseStage from COUNTS '
            'to COUNTS, gain: 1',
            '                      \t\tStage 4: FIRResponseStage from COUNTS '
            'to COUNTS, gain: 1,',
            "            'sampling_rate': 100.0,",
            "            'starttime': 2009-08-24T00:20:03.000000Z,",
            "            'station': 'RJOB'})",
            ]
        actual_stats = pretty(st[0].stats).splitlines()
        self.assertEqual(expected_stats, actual_stats)
        actual_attribdict = pretty(
            AttribDict(st[0].stats.__dict__)).splitlines()
        self.assertEqual(expected_attribdict, actual_attribdict)
        # OK, now for Stats/AttribDict with different keys in it
        st = read('/path/to/test.sac', format='SAC')
        expected_stats = [
            "        network: ''",
            "        station: 'STA'",
            "       location: ''",
            "        channel: 'Q'",
            '      starttime: 1978-07-18T08:00:10.000000Z',
            '        endtime: 1978-07-18T08:01:49.000000Z',
            '  sampling_rate: 1.0',
            '          delta: 1.0',
            '           npts: 100',
            '          calib: 1.0',
            "        _format: 'SAC'",
            "            sac: AttribDict({'b': 10.0,",
            "                             'delta': 1.0,",
            "                             'depmax': 1.0,",
            "                             'depmen': 8.3446501e-08,",
            "                             'depmin': -1.0,",
            "                             'e': 109.0,",
            "                             'iftype': 1,",
            "                             'kcmpnm': 'Q       ',",
            "                             'kevnm': 'FUNCGEN: SINE   ',",
            "                             'kstnm': 'STA     ',",
            "                             'lcalda': 1,",
            "                             'leven': 1,",
            "                             'lovrok': 1,",
            "                             'lpspol': 0,",
            "                             'npts': 100,",
            "                             'nvhdr': 6,",
            "                             'nzhour': 8,",
            "                             'nzjday': 199,",
            "                             'nzmin': 0,",
            "                             'nzmsec': 0,",
            "                             'nzsec': 0,",
            "                             'nzyear': 1978,",
            "                             'unused23': 0})",
            ]
        expected_attribdict = [
            "AttribDict({'b': 10.0,",
            "            'delta': 1.0,",
            "            'depmax': 1.0,",
            "            'depmen': 8.3446501e-08,",
            "            'depmin': -1.0,",
            "            'e': 109.0,",
            "            'iftype': 1,",
            "            'kcmpnm': 'Q       ',",
            "            'kevnm': 'FUNCGEN: SINE   ',",
            "            'kstnm': 'STA     ',",
            "            'lcalda': 1,",
            "            'leven': 1,",
            "            'lovrok': 1,",
            "            'lpspol': 0,",
            "            'npts': 100,",
            "            'nvhdr': 6,",
            "            'nzhour': 8,",
            "            'nzjday': 199,",
            "            'nzmin': 0,",
            "            'nzmsec': 0,",
            "            'nzsec': 0,",
            "            'nzyear': 1978,",
            "            'unused23': 0})",
            ]
        actual_stats = pretty(st[0].stats).splitlines()
        self.assertEqual(expected_stats, actual_stats)
        actual_attribdict = pretty(st[0].stats['sac']).splitlines()
        self.assertEqual(expected_attribdict, actual_attribdict)


def suite():
    return unittest.makeSuite(StatsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
