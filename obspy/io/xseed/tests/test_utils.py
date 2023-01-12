# -*- coding: utf-8 -*-
from obspy import UTCDateTime
from obspy.io.xseed.utils import datetime_2_string, to_tag
from obspy.io.xseed.core import _is_resp


class TestUtils():
    """
    Utils test suite.
    """
    def test_to_tag(self):
        name = "Hello World"
        assert "hello_world" == to_tag(name)

    def test_datetime_to_string(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        assert datetime_2_string(dt) == "2008,358,01:30:22.1234"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 98765)
        assert datetime_2_string(dt) == "2008,358,01:30:22.0987"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 1234)
        assert datetime_2_string(dt) == "2008,358,01:30:22.0012"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123)
        assert datetime_2_string(dt) == "2008,358,01:30:22.0001"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 9)
        assert datetime_2_string(dt) == "2008,358,01:30:22.0000"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 21)
        assert datetime_2_string(dt) == "2008,358,01:30:21.0000"
        dt = UTCDateTime(2008, 12, 23, 0o1, 0, 0, 0)
        assert datetime_2_string(dt) == "2008,358,01:00:00.0000"
        dt = UTCDateTime(2008, 12, 23)
        assert datetime_2_string(dt) == "2008,358"

    def test_datetime_to_string_compact(self):
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22, 123456)
        assert datetime_2_string(dt, True) == "2008,358,01:30:22.1234"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30, 22)
        assert datetime_2_string(dt, True) == "2008,358,01:30:22"
        dt = UTCDateTime(2008, 12, 23, 0o1, 30)
        assert datetime_2_string(dt, True) == "2008,358,01:30"
        dt = UTCDateTime(2008, 12, 23, 0o1)
        assert datetime_2_string(dt, True) == "2008,358,01"
        dt = UTCDateTime(2008, 12, 23)
        assert datetime_2_string(dt, True) == "2008,358"

    def test_is_resp(self, root):
        """
        Checks _is_resp() routine on all files in signal/tests/data.
        """
        signal_test_files = (root / 'signal' / 'tests' / 'data').glob('*')
        # List of files that are actually RESP files, all other files are
        # considered non-RESP files
        resp_filenames = [
            "IUANMO.resp",
            "RESP.NZ.CRLZ.10.HHZ",
            "RESP.NZ.CRLZ.10.HHZ.mac",
            "RESP.NZ.CRLZ.10.HHZ.windows",
            "RESP.OB.AAA._.BH_"]
        for filename in signal_test_files:
            got = _is_resp(filename)
            expected = filename.name in resp_filenames
            assert got == expected, \
                "_is_resp() returns %s for file %s" % (got, filename)
