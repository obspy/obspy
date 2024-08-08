import io
import os.path
from tempfile import gettempdir

import numpy as np
from obspy import read_events
from obspy.core.util import NamedTemporaryFile
import obspy.io.csv.core as iocsv
import pytest
from obspy.core.util import get_example_file


def test_csv():
    events = read_events()
    with NamedTemporaryFile(suffix='.csv') as ft:
        events.write(ft.name, 'CSV')
        events2 = read_events(ft.name)
    assert len(events2) == len(events)
    assert events2[0].origins[0].time == events[0].origins[0].time
    with NamedTemporaryFile(suffix='.csv') as ft:
        events.write(ft.name, 'CSV', depth_in_km=True)
        events2 = read_events(ft.name)
    assert len(events2) == len(events)
    assert events2[0].origins[0].time == events[0].origins[0].time


def test_csv_reading_external_catalog():
    names = 'year mon day hour minu sec _ lat lon dep mag id'
    incomplete_names = 'year mon day hour minu sec _ lat lon dep'
    fname = '/path/to/external.csv'
    assert not iocsv._is_csv(fname)
    events = read_events(fname, 'CSV', skipheader=1, names=names)
    events2 = read_events(fname, 'CSV', skipheader=1, names=incomplete_names)
    assert len(events) == 1
    assert str(events[0].origins[0].time) == '2023-05-06T19:55:01.300000Z'
    assert len(events[0].magnitudes) == 1
    assert len(events2) == 1
    assert str(events2[0].origins[0].time) == '2023-05-06T19:55:01.300000Z'
    assert len(events2[0].magnitudes) == 0


def test_csv_incomplete_catalog():
    events = read_events()
    del events[0].magnitudes[0].magnitude_type
    events[1].magnitudes = []
    events[1].preferred_magnitude_id = None
    events[1].origins[0].depth = None
    events[2].origins = []
    events[2].preferred_origin_id = None
    with NamedTemporaryFile(suffix='.csv') as ft:
        with pytest.warns(Warning, match='.*event 2012'):
            events.write(ft.name, 'CSV')
        events2 = read_events(ft.name, 'CSV')
    assert len(events2) == 2
    assert events2[0].origins[0].time == events[0].origins[0].time
    assert events2[1].origins[0].depth is None
    assert events2[1].origins[0].time == events[1].origins[0].time

    fname = get_example_file('incomplete.csv')
    assert iocsv._is_csv(fname)
    events = read_events(fname)
    assert len(events) == 2
    assert events[0].origins[0].depth is None
    assert len(events[0].magnitudes) == 0
    assert events[1].origins[0].depth is None
    assert len(events[1].magnitudes) == 1
    assert events[1].magnitudes[0].mag == 10.0


def test_csv_custom_fmt():
    events = read_events()
    with NamedTemporaryFile(suffix='.csv') as ft:
        fname = ft.name
        events.write(fname, 'CSV', fields='{lat:.5f} {lon:.5f}')
        assert not iocsv._is_csv(fname)
        data = np.genfromtxt(fname, names=True, delimiter=',')
        assert len(data) == 3
        assert len(data[0]) == 2


def test_csv_empty():
    assert not iocsv._is_csv(b'')
    empty_cat = iocsv._read_csv(b'')
    assert len(empty_cat) == 0


def test_csz(check_compression=False):
    events = read_events('/path/to/example.pha')
    tempdir = gettempdir()
    fname = os.path.join(tempdir, 'obbspycsv_testfile.csz')
    with NamedTemporaryFile(suffix='.csz') as ft:
        fname = ft.name

        def _test_write_read(events, **kw):
            events.write(fname, 'CSZ', **kw)
            assert iocsv._is_csz(fname)
            events2 = read_events(fname, check_compression=check_compression)
            assert len(events2) == len(events)
            for ev1, ev2 in zip(events, events2):
                assert len(ev2.origins[0].arrivals) == \
                    len(ev1.origins[0].arrivals)
                assert len(ev2.picks) == \
                    len(ev1.picks)
        _test_write_read(events)
        _test_write_read(events, compression=False)
        try:
            import zlib  # noqa: F401
        except ImportError:
            pass
        else:
            _test_write_read(events, compression=True, compresslevel=6)
        # test with missing origin and waveformid
        events[1].origins = []
        events[0].picks[0].waveform_id = None
        with pytest.warns(Warning,
                          match='The object with identity|No.*found'):
            events.write(fname, 'CSZ')
        assert iocsv._is_csz(fname)
        events2 = read_events(fname, check_compression=check_compression)
        assert len(events2) == 1
        assert (len(events2[0].origins[0].arrivals) ==
                len(events[0].origins[0].arrivals))
        assert len(events2[0].picks) == len(events[0].picks)
        assert events2[0].picks[0].waveform_id is None


def test_csz_without_picks(check_compression=False):
    events = read_events()
    with NamedTemporaryFile(suffix='.csz') as ft:
        fname = ft.name
        events.write(fname, 'CSZ')
        assert iocsv._is_csz(fname)
        events2 = read_events(fname, check_compression=check_compression)
        assert events2[0]._format == 'CSZ'
        assert len(events2) == len(events)


def test_csz_without_check_compression_parameters():
    test_csz(check_compression=True)
    test_csz_without_picks(check_compression=True)


def test_load_csv():
    events = read_events()
    with NamedTemporaryFile(suffix='.csv') as ft:
        events.write(ft.name, 'CSV')
        t = iocsv.load_csv(ft.name)
    assert t['mag'][0] == 4.4
    with NamedTemporaryFile(suffix='.csz') as ft:
        events.write(ft.name, 'CSZ')
        t = iocsv.load_csv(ft.name)
    assert t['mag'][0] == 4.4
    t = iocsv._events2array(events)
    assert t['mag'][0] == 4.4


def test_load_csv_incomplete_catalog():
    events = read_events()
    del events[0].magnitudes[0].magnitude_type
    events[1].magnitudes = []
    events[1].preferred_magnitude_id = None
    events[2].origins = []
    events[2].preferred_origin_id = None
    with NamedTemporaryFile(suffix='.csv') as ft:
        with pytest.warns(Warning, match='No.*found'):
            events.write(ft.name, 'CSV')
        t = iocsv.load_csv(ft.name)
    assert len(t) == 2
    assert np.isnan(t['mag'][1])


def test_load_csv_some_cols():
    events = read_events()
    with NamedTemporaryFile(suffix='.csv') as ft:
        fields = '{lat:.6f} {lon:.6f} {mag:.2f}'
        events.write(ft.name, 'CSV', fields=fields)
        t = iocsv.load_csv(ft.name)
        t2 = iocsv.load_csv(ft.name, only=['mag'])
        t3 = iocsv.load_csv(ft.name, skipheader=1, names={2: 'mag'})
    assert t['mag'][0] == 4.4
    assert t2['mag'][0] == 4.4
    assert t3['mag'][0] == 4.4
    assert 'mag' in t2.dtype.names
    assert 'lat' not in t2.dtype.names
    assert 'mag' in t3.dtype.names
    assert 'lat' not in t3.dtype.names


def test_events2array():
    events = read_events()
    t = iocsv._events2array(events)
    assert t['mag'][0] == 4.4


def test_eventtxt():
    fname = get_example_file('events.txt')
    assert iocsv._is_eventtxt(fname)
    events = read_events(fname)
    arr = iocsv.load_eventtxt(fname)
    assert len(events) == 2
    assert str(events[0].origins[0].time) == '2012-04-11T08:38:37.000000Z'
    assert events[0].origins[0].creation_info.author == 'ISC'
    assert events[0].magnitudes[0].creation_info.author == 'GCMT'
    assert events[0].event_descriptions[0].text == 'SUMATRA'
    assert len(events[0].magnitudes) == 1
    assert events[1].origins[0].creation_info is None
    assert events[1].magnitudes[0].creation_info is None
    assert list(arr['mag']) == [8.6, 8.5]
    with open(fname) as f:
        eventtxt = f.read()
    with io.StringIO() as f:
        events.write(f, 'EVENTTXT')
        assert f.getvalue() == eventtxt
