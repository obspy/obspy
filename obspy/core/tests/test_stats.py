# -*- coding: utf-8 -*-
import copy
import io
import pickle
import pytest
import warnings

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core import Stats
from obspy.core.util.testing import WarningsCapture
from obspy.core.util import AttribDict


class TestStats:
    """
    Test suite for obspy.core.util.Stats.
    """
    nslc = ['network', 'station', 'location', 'channel']

    def test_init(self):
        """
        Init tests.
        """
        stats = Stats({'test': 'muh'})
        stats['other1'] = {'test1': '1'}
        stats['other2'] = AttribDict({'test2': '2'})
        stats['other3'] = 'test3'
        assert stats.test == 'muh'
        assert stats['test'] == 'muh'
        assert stats.other1.test1 == '1'
        assert stats.other1.__class__ == AttribDict
        assert len(stats.other1) == 1
        assert stats.other2.test2 == '2'
        assert stats.other2.__class__ == AttribDict
        assert len(stats.other2) == 1
        assert stats.other3 == 'test3'
        assert 'test' in stats
        assert 'test' in stats.__dict__

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
        assert stats2.__class__ == Stats
        assert stats2.network == 'BW'
        assert stats2.station == 'ROTZ'
        assert stats2.other1.test1 == '1'
        assert stats2.other1.__class__ == AttribDict
        assert len(stats2.other1) == 1
        assert stats2.other2.test2 == '2'
        assert stats2.other2.__class__ == AttribDict
        assert len(stats2.other2) == 1
        assert stats2.other3 == 'test3'
        assert stats.network == 'CZ'
        assert stats.station == 'RJOB'

    def test_update(self):
        """
        Tests update method of Stats object.
        """
        x = Stats({'a': 5})
        assert 'a' in dir(x)
        x.update({'b': 5})
        assert 'b' in dir(x)
        y = {'a': 5}
        y.update({'b': 5})
        x = Stats(y)
        assert 'b' in dir(x)

    def test_simple_stats(self):
        """
        Various setter and getter tests.
        """
        stats = Stats()
        stats.test = 1
        assert stats.test == 1
        assert stats['test'] == 1
        stats['test2'] = 2
        assert stats.test2 == 2
        assert stats['test2'] == 2
        stats['test'] = 2
        assert stats.test == 2
        assert stats['test'] == 2
        stats.test2 = 1
        assert stats.test2 == 1
        assert stats['test2'] == 1

    def test_nested_stats(self):
        """
        Various setter and getter tests.
        """
        # 1
        stats = Stats()
        stats.test = dict()
        stats.test['test2'] = 'muh'
        assert stats.test.test2 == 'muh'
        assert stats.test['test2'] == 'muh'
        assert stats['test'].test2 == 'muh'
        assert stats['test']['test2'] == 'muh'
        stats.test['test2'] = 'maeh'
        assert stats.test.test2 == 'maeh'
        assert stats.test['test2'] == 'maeh'
        assert stats['test'].test2 == 'maeh'
        assert stats['test']['test2'] == 'maeh'
        # 2 - multiple initialization
        stats = Stats({'muh': 'meah'})
        stats2 = Stats(Stats(Stats(stats)))
        assert stats2.muh == 'meah'
        # 3 - check conversion to AttribDict
        stats = Stats()
        stats.sub1 = {'muh': 'meah'}
        stats.sub2 = AttribDict({'muh2': 'meah2'})
        stats2 = Stats(stats)
        assert isinstance(stats.sub1, AttribDict)
        assert isinstance(stats.sub2, AttribDict)
        assert stats2.sub1.muh == 'meah'
        assert stats2.sub2.muh2 == 'meah2'

    def test_bugfix_set_stats(self):
        """
        Test related to issue #4.
        """
        st = Stream([Trace()])
        st += st
        # change stats attributes
        st[0].stats.station = 'AAA'
        st[1].stats['station'] = 'BBB'
        assert st[0].stats.station == 'BBB'
        assert st[0].stats['station'] == 'BBB'
        assert st[1].stats['station'] == 'BBB'
        assert st[1].stats.station == 'BBB'

    def test_bugfix_set_stats_2(self):
        """
        Second test related to issue #4.
        """
        st = Stream([Trace(header={'station': 'BGLD'})])
        assert st[0].stats.station == 'BGLD'
        assert st[0].stats['station'] == 'BGLD'
        st[0].stats.station = 'AAA'
        assert st[0].stats.station == 'AAA'
        assert st[0].stats['station'] == 'AAA'
        st = st + st
        assert st[0].stats.station == 'AAA'
        assert st[0].stats['station'] == 'AAA'
        st[0].stats.station = 'BBB'
        assert st[0].stats.station == 'BBB'
        assert st[0].stats['station'] == 'BBB'

    def test_bugfix_set_stats_3(self):
        """
        Third test related to issue #4.
        """
        st = Stream([Trace(header={'station': 'BGLD'})])
        assert st[0].stats.station == 'BGLD'
        st = st + st
        st[0].stats.station = 'AAA'
        st = st + st
        st[3].stats.station = 'BBB'
        # changed in rev. 1625: adding streams doesn't deepcopy
        # therefore all traces in the test stream are identical
        # (python list behavior)
        for tr in st:
            assert tr == st[0]
            assert tr.stats.station == 'BBB'
            assert tr.stats['station'] == 'BBB'
            assert tr.stats.get('station') == 'BBB'
            assert 'BBB' in tr.stats.values()

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
        assert stats == stats2
        # old binary
        temp = pickle.dumps(stats, protocol=1)
        stats2 = pickle.loads(temp)
        assert stats == stats2
        # new binary
        temp = pickle.dumps(stats, protocol=2)
        stats2 = pickle.loads(temp)
        assert stats == stats2
        # SOH channels sampling_rate & delta == 0. for #1989
        stats.sampling_rate = 0
        pickle.loads(pickle.dumps(stats, protocol=0))
        pickle.loads(pickle.dumps(stats, protocol=1))
        pickle.loads(pickle.dumps(stats, protocol=2))

    def test_set_calib(self):
        """
        Test to prevent setting a calibration factor of 0
        """
        x = Stats()
        # this should work
        x.update({'calib': 1.23})
        assert x.calib, 1.23
        # this raises UserWarning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            # 1
            with pytest.raises(UserWarning):
                x.__setitem__('calib', 0)
            # 2
            with pytest.raises(UserWarning):
                x.update({'calib': 0})
        # calib value should nevertheless be set to 0
        assert x.calib, 0

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
        assert ad == adict
        assert adict == ad

    def test_delta_zero(self):
        """
        Make sure you can set delta = 0. for #1989
        """
        stat = Stats()
        stat.delta = 0

    def test_non_str_in_nscl_raise_warning(self):
        """
        Ensure assigning a non-str value to network, station, location, or
        channel issues a warning, then casts value into str. See issue # 1995
        """
        stats = Stats()

        for val in self.nslc:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('default')
                setattr(stats, val, 42)
            # make sure a warning was issued
            assert len(w) == 1
            exp_str = 'Attribute "%s" must be of type ' % val
            assert exp_str in str(w[-1].message)
            # make sure the value was cast to a str
            new_val = getattr(stats, val)
            assert new_val == '42'

    def test_nscl_cannot_be_none(self):
        """
        Ensure the nslc values can't be assigned to None but rather None
        gets converted to a str
        """
        stats = Stats()
        for val in self.nslc:
            with pytest.warns(UserWarning):
                setattr(stats, val, None)
            assert getattr(stats, val) == 'None'

    @pytest.mark.filterwarnings('ignore:Attribute')
    def test_casted_stats_nscl_writes_to_mseed(self):
        """
        Ensure a Stream object that has had its nslc types cast to str can
        still be written.
        """
        st = Stream(traces=read()[0])

        # Get a new stats object with just the basic items in it
        stats_items = set(Stats())
        new_stats = Stats()
        new_stats.__dict__.update({x: st[0].stats[x] for x in stats_items})
        with WarningsCapture():
            new_stats.network = 1
            new_stats.station = 1.1
        new_stats.channel = 'Non'
        st[0].stats = new_stats
        # try writing stream to bytes buffer
        bio = io.BytesIO()
        st.write(bio, 'mseed')
        bio.seek(0)
        # read bytes and compare
        stt = read(bio)
        # remove _mseed so streams can compare equal
        stt[0].stats.pop('mseed')
        del stt[0].stats._format  # format gets added upon writing
        assert st == stt

    @pytest.mark.filterwarnings('ignore:Attribute')
    def test_different_string_types(self):
        """
        Test the various types of strings found in the wild get converted to
        native_str type.
        """
        nbytes = bytes('HHZ', 'utf8')
        the_strs = ['HHZ', nbytes, u'HHZ']

        stats = Stats()

        for a_str in the_strs:
            for nslc in self.nslc:
                setattr(stats, nslc, a_str)
                assert isinstance(getattr(stats, nslc), str)

    def test_component(self):
        """
        Test setting and getting of component.
        """
        stats = Stats()
        # Channel with 3 characters
        stats.channel = 'HHZ'
        assert stats.component == 'Z'
        stats.component = 'L'
        assert stats.component == 'L'
        assert stats.channel == 'HHL'
        stats['component'] = 'Q'
        assert stats['component'] == 'Q'
        assert stats.channel == 'HHQ'
        # Channel with 1 character as component
        stats.channel = 'N'
        stats.component = 'E'
        assert stats.channel == 'E'
        assert stats.component == 'E'
        # Channel with 0 characters
        stats.channel = ''
        assert stats.component == ''
        stats.component = 'Z'
        assert stats.channel == 'Z'
        # Components must be single character
        stats.channel = 'HHZ'
        with pytest.raises(ValueError):
            stats.component = ''
        assert stats.channel == 'HHZ'
        with pytest.raises(ValueError):
            stats.component = 'ZZ'
        assert stats.channel == 'HHZ'
