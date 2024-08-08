# -*- coding: utf-8 -*-
"""
The sac.core test suite.
"""
import copy
import io
import warnings

import numpy as np

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core.util import NamedTemporaryFile, CatchAndAssertWarnings
from obspy.core import AttribDict
from obspy.io.sac import SacError, SACTrace, SacIOError
from obspy.io.sac.core import (_is_sac, _is_sac_xy, _read_sac, _read_sac_xy,
                               _write_sac, _write_sac_xy)
from obspy.io.sac.util import utcdatetime_to_sac_nztimes
import pytest


class TestCore():
    """
    Test cases for sac core interface
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        # directory where the test files are located
        self.file = testdata['test.sac']
        self.filexy = testdata['testxy.sac']
        self.filebe = testdata['test.sac.swap']
        self.fileseis = testdata['seism.sac']
        self.file_notascii = testdata['non_ascii.sac']
        self.file_encode = testdata['test_encode.sac']
        self.testdata = np.array(
            [-8.74227766e-08, -3.09016973e-01,
             -5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
             -1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
             -5.87784529e-01, -3.09016049e-01], dtype=np.float32)

    def test_read_via_obspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        assert tr.stats['station'] == 'STA'
        assert tr.stats.npts == 100
        assert tr.stats['sampling_rate'] == 1.0
        assert tr.stats.get('channel') == 'Q'
        assert tr.stats.starttime.timestamp == 269596810.0
        assert tr.stats.sac.get('nvhdr') == 6
        assert tr.stats.sac.b == 10.0
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_read_write_via_obspy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SAC')
            tr1 = read(tempfile)[0]
        np.testing.assert_array_equal(tr.data, tr1.data)
        # this tests failed because SAC calculates the seismogram's
        # mean value in single precision and python in double
        # precision resulting in different values. The following line
        # is therefore just a fix until we have come to a conclusive
        # solution how to handle the two different approaches
        tr1.stats.sac['depmen'] = tr.stats.sac['depmen']
        assert tr == tr1

    def test_read_xy_write_xy_via_obspy(self):
        """
        Write/Read files via L{obspy.Stream}
        """
        tr = read(self.filexy, format='SACXY')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SACXY')
            tr1 = read(tempfile)[0]

        tr.stats.pop('sac', None)
        tr1.stats.pop('sac', None)

        assert tr == tr1

    def test_read_write_xy_via_obspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC')[0]
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SACXY')
            tr1 = read(tempfile)[0]
        assert tr1.stats['station'] == 'STA'
        assert tr1.stats.npts == 100
        assert tr1.stats['sampling_rate'] == 1.0
        assert tr1.stats.get('channel') == 'Q'
        assert tr1.stats.starttime.timestamp == 269596810.0
        assert tr1.stats.sac.get('nvhdr') == 6
        assert tr1.stats.sac.b == 10.0
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr1.data[0:10])

    def test_read_big_endian_via_obspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.filebe, format='SAC')[0]
        assert tr.stats['station'] == 'STA'
        assert tr.stats.npts == 100
        assert tr.stats['sampling_rate'] == 1.0
        assert tr.stats.get('channel') == 'Q'
        assert tr.stats.starttime.timestamp == 269596810.0
        assert tr.stats.sac.get('nvhdr') == 6
        assert tr.stats.sac.b == 10.0
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             tr.data[0:10])

    def test_swap_bytes_via_obspy(self):
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            trbe = read(self.filebe, format='SAC')[0]
            trbe.write(tempfile, format='SAC', byteorder='<')
            tr = read(tempfile, format='SAC')[0]
            trle = read(self.file, format='SAC')[0]
            assert tr.stats.station == trle.stats.station
            assert tr.stats.npts == trle.stats.npts
            assert tr.stats.delta == trle.stats.delta
            assert tr.stats.sac.b == trle.stats.sac.b
            np.testing.assert_array_almost_equal(tr.data[0:10],
                                                 trle.data[0:10])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            trle = read(self.file, format='SAC')[0]
            trle.write(tempfile, format='SAC', byteorder='>')
            tr = read(tempfile, format='SAC')[0]
            trbe = read(self.filebe, format='SAC')[0]
            assert tr.stats.station == trbe.stats.station
            assert tr.stats.npts == trbe.stats.npts
            assert tr.stats.delta == trbe.stats.delta
            assert tr.stats.sac.b == trbe.stats.sac.b
            np.testing.assert_array_almost_equal(tr.data[0:10],
                                                 trbe.data[0:10])

    def test_read_head_via_obspy(self):
        """
        Read files via L{obspy.Stream}
        """
        tr = read(self.file, format='SAC', headonly=True)[0]
        assert tr.stats['station'] == 'STA'
        assert tr.stats.npts == 100
        assert tr.stats['sampling_rate'] == 1.0
        assert tr.stats.get('channel') == 'Q'
        assert tr.stats.starttime.timestamp == 269596810.0
        assert tr.stats.sac.get('nvhdr') == 6
        assert tr.stats.sac.b == 10.0
        assert str(tr.data) == '[]'

    def test_write_via_obspy(self):
        """
        Writing artificial files via L{obspy.Stream}
        """
        st = Stream(traces=[Trace(header={'sac': {}}, data=self.testdata)])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st.write(tempfile, format='SAC')
            tr = read(tempfile)[0]
        np.testing.assert_array_almost_equal(self.testdata, tr.data)

    def test_set_version(self):
        """
        Tests if SAC version is set when writing
        """
        np.random.seed(815)
        st = Stream([Trace(data=np.random.randn(1000))])
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st.write(tempfile, format="SAC")
            st2 = read(tempfile, format="SAC")
        assert st2[0].stats['sac'].nvhdr == 6

    def test_read_and_write_via_obspy(self):
        """
        Read and Write files via L{obspy.Stream}
        """
        # read trace
        tr = read(self.file)[0]
        # write comparison trace
        st2 = Stream()
        st2.traces.append(Trace())
        tr2 = st2[0]
        tr2.data = copy.deepcopy(tr.data)
        tr2.stats = copy.deepcopy(tr.stats)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            st2.write(tempfile, format='SAC')
            # read comparison trace
            tr3 = read(tempfile)[0]
        # check if equal
        assert tr3.stats['station'] == tr.stats['station']
        assert tr3.stats.npts == tr.stats.npts
        assert tr.stats['sampling_rate'] == tr.stats['sampling_rate']
        assert tr.stats.get('channel') == tr.stats.get('channel')
        assert tr.stats.get('starttime') == tr.stats.get('starttime')
        assert tr.stats.sac.get('nvhdr') == tr.stats.sac.get('nvhdr')
        np.testing.assert_equal(tr.data, tr3.data)

    def test_convert_to_sac(self):
        """
        Test that an obspy trace is correctly written to SAC.
        All the header variables which are tagged as required by
        https://ds.iris.edu/files/sac-manual/manual/file_format.html
        are controlled in this test
        """
        # setUp is called before every test, not only once at the
        # beginning, that is we allocate the data just here
        # generate artificial mseed data
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'location': '00',
                'channel': 'BHZ', 'calib': 1.0, 'sampling_rate': 40.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 43400)}
        data = np.random.randint(0, 5000, 11947).astype(np.int32)
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            st.write(tmpfile, format="SAC")
            st2 = read(tmpfile, format="SAC")
        # check all the required entries (see url in docstring)
        assert st2[0].stats.starttime == st[0].stats.starttime
        assert st2[0].stats.npts == st[0].stats.npts
        assert st2[0].stats.sac.nvhdr == 6
        assert round(abs(st2[0].stats.sac.b-0.000400), 7) == 0
        # compare with correct digit size (nachkommastellen)
        assert round(abs(
            (0.0004 + (st[0].stats.npts - 1) *
             st[0].stats.delta) / st2[0].stats.sac.e-1.0), 7) == 0
        assert st2[0].stats.sac.iftype == 1
        assert st2[0].stats.sac.leven == 1
        assert round(abs(st2[0].stats.sampling_rate /
                         st[0].stats.sampling_rate-1.0), 7) == 0

    def test_iztype11(self, testdata):
        # test that iztype 11 is read correctly
        sod_file = testdata['dis.G.SCZ.__.BHE_short']
        tr = read(sod_file)[0]
        with open(sod_file, "rb") as fh:
            sac = SACTrace.read(fh)
        t1 = tr.stats.starttime - float(tr.stats.sac.b)
        t2 = sac.reftime
        assert round(abs(t1.timestamp-t2.timestamp), 5) == 0
        # see that iztype is written correctly
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format="SAC")
            with open(tempfile, "rb") as fh:
                sac2 = SACTrace.read(fh)
        assert sac2._header['iztype'] == 11
        assert round(abs(tr.stats.sac.b-sac2.b), 7) == 0
        assert round(abs(t2.timestamp-sac2.reftime.timestamp), 5) == 0

    def test_default_values(self):
        tr = read(self.file)[0]
        assert tr.stats.calib == 1.0
        assert tr.stats.location == ''
        assert tr.stats.network == ''

    def test_reference_time(self):
        """
        Test case for bug #107. The SAC reference time is specified by the
        iztype. However it seems no matter what iztype is given, the
        starttime of the seismogram is calculated by adding the B header
        (in seconds) to the SAC reference time.
        """
        tr = read(self.fileseis)[0]
        # see that starttime is set correctly (#107)
        assert round(abs(tr.stats.sac.iztype-9), 7) == 0
        assert round(abs(tr.stats.sac.b-9.4599991), 7) == 0
        assert tr.stats.starttime == UTCDateTime("1981-03-29 10:38:23.459999")
        # check that if we rewrite the file, nothing changed
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            tr.write(tmpfile, format="SAC")
            tr2 = read(tmpfile)[0]
            assert tr.stats.station == tr2.stats.station
            assert tr.stats.npts == tr2.stats.npts
            assert tr.stats.delta == tr2.stats.delta
            assert tr.stats.starttime == tr2.stats.starttime
            assert tr.stats.sac.b == tr2.stats.sac.b
            np.testing.assert_array_equal(tr.data, tr2.data)
        # test some more entries, I can see from the plot
        assert tr.stats.station == "CDV"
        assert tr.stats.channel == "Q"

    def test_undefined_b(self):
        """
        Test that an undefined B value (-12345.0) is not messing up the
        starttime
        """
        # read in the test file an see that sac reference time and
        # starttime of seismogram are correct
        tr = read(self.file)[0]
        assert tr.stats.starttime.timestamp == 269596810.0
        assert tr.stats.sac.b == 10.0
        with open(self.file, 'rb') as fh:
            sac_ref_time = SACTrace.read(fh).reftime
        assert sac_ref_time.timestamp == 269596800.0
        # change b to undefined and write (same case as if b == 0.0)
        # now sac reference time and reftime of seismogram must be the
        # same
        tr.stats.sac.b = -12345.0
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            tr.write(tmpfile, format="SAC")
            tr2 = read(tmpfile)[0]
            assert tr2.stats.starttime.timestamp == 269596810.0
            assert tr2.stats.sac.b == 10.0
            with open(tmpfile, "rb") as fh:
                sac_ref_time2 = SACTrace.read(fh).reftime
        assert sac_ref_time2.timestamp == 269596800.0

    def test_issue_156(self):
        """
        Test case for issue #156.
        """
        # 1
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SAC')
            st = read(sac_file)
        assert st[0].stats.delta == 0.01
        assert st[0].stats.sampling_rate == 100.0
        # 2
        tr = Trace()
        tr.stats.delta = 0.005
        tr.data = np.arange(0, 2000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SAC')
            st = read(sac_file)
        assert st[0].stats.delta == 0.005
        assert st[0].stats.sampling_rate == 200.0

    def test_write_sac_xy_with_minimum_stats(self):
        """
        Write SACXY with minimal stats header, no inhereted from SAC file
        """
        tr = Trace()
        tr.stats.delta = 0.01
        tr.data = np.arange(0, 3000)
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr.write(sac_file, 'SACXY')
            st = read(sac_file)
        assert st[0].stats.delta == 0.01
        assert st[0].stats.sampling_rate == 100.0

    def test_not_used_but_given_headers(self):
        """
        Test case for #188
        """
        tr1 = read(self.file)[0]
        not_used = ['xminimum', 'xmaximum', 'yminimum', 'ymaximum',
                    'unused6', 'unused7', 'unused8', 'unused9', 'unused10',
                    'unused11', 'unused12']
        for i, header_value in enumerate(not_used):
            tr1.stats.sac[header_value] = i
        with NamedTemporaryFile() as tf:
            sac_file = tf.name
            tr1.write(sac_file, 'SAC')
            tr2 = read(sac_file)[0]
        for i, header_value in enumerate(not_used):
            assert int(tr2.stats.sac[header_value]) == i

    def test_writing_micro_seconds(self):
        """
        Test case for #194. Check that microseconds are written to
        the SAC header b
        """
        np.random.seed(815)
        head = {'network': 'NL', 'station': 'HGN', 'channel': 'BHZ',
                'sampling_rate': 200.0,
                'starttime': UTCDateTime(2003, 5, 29, 2, 13, 22, 999999)}
        data = np.random.randint(0, 5000, 100).astype(np.int32)
        st = Stream([Trace(header=head, data=data)])
        # write them as SAC
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            st.write(tmpfile, format="SAC")
            st2 = read(tmpfile, format="SAC")
        # check all the required entries (see url in docstring)
        assert st2[0].stats.starttime == st[0].stats.starttime
        assert round(abs(st2[0].stats.sac.b-0.000999), 7) == 0

    def test_null_terminated_strings(self, testdata):
        """
        Test case for #374. Check that strings stop at the position
        of null termination '\x00'
        """
        null_file = testdata['null_terminated.sac']
        tr = read(null_file)[0]
        assert tr.stats.station == 'PIN1'
        assert tr.stats.network == 'GD'
        assert tr.stats.channel == 'LYE'

    def test_write_small_trace(self):
        """
        Tests writing Traces containing 0, 1, 2, 3, 4 samples only.
        """
        for format in ['SAC', 'SACXY']:
            for num in range(5):
                tr = Trace(data=np.arange(num))
                with NamedTemporaryFile() as tf:
                    tempfile = tf.name
                    tr.write(tempfile, format=format)
                    # test results
                    st = read(tempfile, format=format)
                assert len(st) == 1
                np.testing.assert_array_equal(tr.data, st[0].data)

    def test_issue390(self):
        """
        Read all SAC headers if debug_headers flag is enabled.
        """
        # 1 - binary SAC
        tr = read(self.file, headonly=True, debug_headers=True)[0]
        assert tr.stats.sac.nzyear == 1978
        assert tr.stats.sac.nzjday == 199
        assert tr.stats.sac.nzhour == 8
        assert tr.stats.sac.nzmin == 0
        assert tr.stats.sac.nzsec == 0
        assert tr.stats.sac.nzmsec == 0
        assert tr.stats.sac.delta == 1.0
        assert tr.stats.sac.scale == -12345.0
        assert tr.stats.sac.npts == 100
        assert tr.stats.sac.knetwk == '-12345  '
        assert tr.stats.sac.kstnm == 'STA     '
        assert tr.stats.sac.kcmpnm == 'Q       '
        # 2 - ASCII SAC
        tr = read(self.filexy, headonly=True, debug_headers=True)[0]
        assert tr.stats.sac.nzyear == -12345
        assert tr.stats.sac.nzjday == -12345
        assert tr.stats.sac.nzhour == -12345
        assert tr.stats.sac.nzmin == -12345
        assert tr.stats.sac.nzsec == -12345
        assert tr.stats.sac.nzmsec == -12345
        assert tr.stats.sac.delta == 1.0
        assert tr.stats.sac.scale == -12345.0
        assert tr.stats.sac.npts == 100
        assert tr.stats.sac.knetwk == '-12345  '
        assert tr.stats.sac.kstnm == 'sta     '
        assert tr.stats.sac.kcmpnm == 'Q       '

    def test_read_with_fsize(self, testdata):
        """
        Testing fsize option on read()
        """
        # reading sac file with wrong file size should raise error
        longer_file = testdata['seism-longer.sac']
        shorter_file = testdata['seism-shorter.sac']
        # default
        with pytest.raises(SacError):
            read(longer_file)
        with pytest.raises(SacError):
            read(shorter_file)
        # fsize=True
        with pytest.raises(SacError):
            read(longer_file, fsize=True)
        with pytest.raises(SacError):
            read(shorter_file, fsize=True)
        # using fsize=False should not work for shorter file
        # (this is not supported by SAC) ...
        with pytest.raises(SacIOError):
            read(shorter_file, fsize=False)
        # ...but it should work for longer file
        tr = read(longer_file, fsize=False, debug_headers=True)[0]
        # checking trace
        assert tr.stats.sac.nzyear == 1981
        assert tr.stats.sac.nzjday == 88
        assert tr.stats.sac.nzhour == 10
        assert tr.stats.sac.nzmin == 38
        assert tr.stats.sac.nzsec == 14
        assert tr.stats.sac.nzmsec == 0
        # we should never test equality for float values:
        assert abs(tr.stats.sac.delta - 0.01) <= 1e-9
        assert tr.stats.sac.scale == -12345.0
        assert tr.stats.sac.npts == 998
        assert tr.stats.sac.knetwk == '-12345  '
        assert tr.stats.sac.kstnm == 'CDV     '
        assert tr.stats.sac.kcmpnm == 'Q       '

    def test_read_sac_from_bytes_io(self):
        """
        Tests reading from a BytesIO object.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)

            # Attempt to read from it.
            tr = _read_sac(buf)[0]

        # Open file normally and make sure the results are identical.
        tr2 = _read_sac(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_read_sac_from_open_file(self):
        """
        Tests reading from an open file.
        """
        with open(self.file, "rb") as fh:
            # Attempt to read from it.
            tr = _read_sac(fh)[0]

        # Open file normally and make sure the results are identical.
        tr2 = _read_sac(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_read_write_bytes_io(self):
        """
        Tests reading and writing to and from BytesIO.
        """
        st = _read_sac(self.file)
        with io.BytesIO() as buf:
            _write_sac(st, buf)
            buf.seek(0, 0)
            # Attempt to read from it.
            st2 = _read_sac(buf)

        tr = st[0]
        tr2 = st2[0]
        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_read_write_open_file(self):
        st = _read_sac(self.file)

        with NamedTemporaryFile() as tf_out:
            _write_sac(st, tf_out)
            tf_out.seek(0, 0)
            st2 = _read_sac(tf_out)

        tr = st[0]
        tr2 = st2[0]
        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_writing_to_obj_with_multiple_traces_fails(self):
        """
        Writing to a buf with multiple trace objects should fail. The SAC
        format cannot deal with that.
        """
        st = read()
        with io.BytesIO() as fh:
            with pytest.raises(ValueError):
                st.write(fh, format="sac")

    def test_writing_to_io_string_io_fails(self):
        """
        Writing to io.StringIO should fail on all platforms.
        """
        st = read()[:1]
        with io.StringIO() as fh:
            with pytest.raises(ValueError):
                st.write(fh, format="sac")

    def test_read_via_obspy_from_bytes_io(self):
        """
        Read sac files from a BytesIO object via ObsPy.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)

            # Attempt to read from it.
            tr = read(buf)[0]

        # Open file normally and make sure the results are identical.
        tr2 = read(self.file)[0]
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_write_via_obspy_to_bytes_io(self):
        """
        Read sac files from a BytesIO object via ObsPy.
        """
        tr = read(self.file)[0]
        with io.BytesIO() as buf:
            tr.write(buf, format="sac")
            buf.seek(0, 0)
            # Attempt to read from it.
            tr2 = read(buf)[0]

        # depmen is different as it is actually calculated on the fly.
        del tr.stats.sac.depmen
        del tr2.stats.sac.depmen
        np.testing.assert_array_equal(tr.data, tr2.data)
        assert tr == tr2

    def test_read_xy_write_xy_from_bytes_io(self):
        """
        Reading/writing XY sac files from/to io.BytesIO. It's alphanumeric
        so bytes should also do the trick.
        """
        # Original.
        st = _read_sac_xy(self.filexy)

        with io.BytesIO() as fh:
            _write_sac_xy(st, fh)
            fh.seek(0, 0)
            st2 = _read_sac_xy(fh)

        st[0].stats.pop('sac', None)
        st2[0].stats.pop('sac', None)

        assert st == st2

    def test_read_xy_write_xy_from_open_file_binary_mode(self):
        """
        Reading/writing XY sac files to open files in binary mode.
        """
        # Original.
        st = _read_sac_xy(self.filexy)

        with NamedTemporaryFile() as tf:
            _write_sac_xy(st, tf)
            tf.seek(0, 0)
            st2 = _read_sac_xy(tf)

        st[0].stats.pop('sac', None)
        st2[0].stats.pop('sac', None)

        assert st == st2

    def test_is_sac_bytes_io(self):
        """
        Tests the _is_sac function for BytesIO objects.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            assert _is_sac(buf)

        # Should naturally fail for an XY file.
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            assert not _is_sac(buf)

    def test_is_sac_string_io_raises(self):
        """
        Should raise a ValueError.
        """
        with io.StringIO() as buf:
            buf.write("abcdefghijklmnopqrstuvwxyz")
            buf.seek(0, 0)
            with pytest.raises(ValueError):
                _is_sac(buf)

    def test_is_sac_open_file(self):
        """
        Tests the _is_sac function for open files.
        """
        with open(self.file, "rb") as fh:
            assert _is_sac(fh)

    def test_is_sacxy_bytes_io(self):
        """
        Tests the _is_sac_xy function for BytesIO objects.
        """
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            assert _is_sac_xy(buf)

        # Should naturally fail for a normal sac file.
        with io.BytesIO() as buf:
            # Read file to BytesIO.
            with open(self.file, "rb") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            assert not _is_sac_xy(buf)

    def test_is_sacxy_string_io_raises(self):
        """
        Tests the _is_sac_xy function for StringIO objects where it should
        raise. I/O is binary only.
        """
        with io.StringIO() as buf:
            # Read file to BytesIO.
            with open(self.filexy, "rt") as fh:
                buf.write(fh.read())
            buf.seek(0, 0)
            with pytest.raises(ValueError):
                _is_sac_xy(buf)

    def test_is_sacxy_open_file_binary_mode(self):
        """
        Tests the _is_sac_xy function for open files in binary mode.
        """
        with open(self.filexy, "rb") as fh:
            assert _is_sac_xy(fh)

        with open(__file__, "rb") as fh:
            assert not _is_sac_xy(fh)

    def test_is_sacxy_open_file_text_mode_fails(self):
        """
        Tests that the _is_sac_xy function for open files in text mode fails.
        """
        with open(self.filexy, "rt") as fh:
            with pytest.raises(ValueError):
                _is_sac_xy(fh)

    def test_writing_to_file_like_objects_with_obspy(self):
        """
        Very simple test just executing a common operation and making sure
        it does not fail.
        """
        st = read()[:1]

        with io.BytesIO() as fh:
            st.write(fh, format="sac")

        with io.BytesIO() as fh:
            st.write(fh, format="sacxy")

        # Will fail if the stream contains more than one trace.
        st = read()
        assert len(st) > 1
        with io.BytesIO() as fh:
            with pytest.raises(ValueError):
                st.write(fh, format="sac")

        with io.BytesIO() as fh:
            with pytest.raises(ValueError):
                st.write(fh, format="sacxy")

    def test_valid_sac_from_minimal_existing_sac_header(self):
        """
        An incomplete manually-produced SAC header should still produce a
        valid SAC file, including values from the ObsPy header.  Issue 1204.
        """
        tr = Trace(np.arange(100))
        t = UTCDateTime()
        tr.stats.starttime = t
        tr.stats.station = 'AAA'
        tr.stats.network = 'XX'
        tr.stats.channel = 'BHZ'
        tr.stats.location = '00'

        tr.stats.sac = AttribDict()
        tr.stats.sac.iztype = 9
        tr.stats.sac.nvhdr = 6
        tr.stats.sac.leven = 1
        tr.stats.sac.lovrok = 1
        tr.stats.sac.iftype = 1
        tr.stats.sac.stla = 1.
        tr.stats.sac.stlo = 2.

        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SAC')
            tr1 = read(tempfile)[0]

        # starttime made its way to SAC file
        nztimes, microsecond = utcdatetime_to_sac_nztimes(t)
        assert tr1.stats.sac.nzyear == nztimes['nzyear']
        assert tr1.stats.sac.nzjday == nztimes['nzjday']
        assert tr1.stats.sac.nzhour == nztimes['nzhour']
        assert tr1.stats.sac.nzmin == nztimes['nzmin']
        assert tr1.stats.sac.nzsec == nztimes['nzsec']
        assert tr1.stats.sac.nzmsec == nztimes['nzmsec']
        assert tr1.stats.sac.kstnm == 'AAA'
        assert tr1.stats.sac.knetwk == 'XX'
        assert tr1.stats.sac.kcmpnm == 'BHZ'
        assert tr1.stats.sac.khole == '00'
        assert tr1.stats.sac.iztype == 9
        assert tr1.stats.sac.nvhdr == 6
        assert tr1.stats.sac.leven == 1
        assert tr1.stats.sac.lovrok == 1
        assert tr1.stats.sac.iftype == 1
        assert tr1.stats.sac.stla == 1.0
        assert tr1.stats.sac.stlo == 2.0

    def test_merge_sac_obspy_headers(self):
        """
        Test that manually setting a set of SAC headers not related
        to validity or reference time on Trace.stats.sac is properly merged
        with the Trace.stats header. Issue 1285.
        """
        tr = Trace(data=np.arange(30))
        o = 10.0
        tr.stats.sac = {'o': o}

        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SAC')
            tr1 = read(tempfile)[0]

        assert tr1.stats.starttime == tr.stats.starttime
        assert tr1.stats.sac.o == o

    def test_decimate_resample(self):
        """
        Test that ObsPy Trace resampling and decimation is properly reflected
        in the SAC file.
        """
        tr = read(self.file, format='SAC')[0]
        tr.decimate(2)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr.write(tempfile, format='SAC')
            tr1 = read(tempfile)[0]
        assert tr1.stats.sac.npts == tr.stats.sac.npts / 2
        assert tr1.stats.sac.delta == tr.stats.sac.delta * 2

    def test_invalid_header_field(self):
        """
        Given a SAC file on disk, when it is read and an invalid header is
        appended to the stats.sac dictionary, then the invalid header should be
        ignored (user given a warning) and the written file should be the same
        as the original.
        """
        tr = read(self.file, format='SAC')[0]

        with io.BytesIO() as buf:
            tr.write(buf, format='SAC')
            buf.seek(0, 0)

            tr.stats.sac.AAA = 10.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                with io.BytesIO() as buf1:
                    tr.write(buf1, format='SAC')
                    assert 'Ignored' in str(w[-1].message)
                    buf1.seek(0, 0)

                    assert buf.read() == buf1.read()

    def test_not_ascii(self):
        """
        Read file with non-ascii and null-termination characters.
        See ObsPy issue #1432
        """
        tr = read(self.file_notascii, format='SAC')[0]
        assert tr.stats.station == 'ALS'
        assert tr.stats.channel == 'HHE'
        assert tr.stats.network == ''

    def test_sac_booleans_from_trace(self):
        """
        SAC booleans "lcalda" and "lpspol" should be "False" and "True",
        respectively, by default when converting from a "Trace".
        """
        tr = Trace()
        sac = SACTrace.from_obspy_trace(tr)
        assert not sac.lcalda
        assert sac.lpspol

    def test_sac_file_from_new_header(self):
        """
        Writing to disk a new Trace object shouldn't ignore custom header
        fields, if an arrival time is set. See ObsPy issue #1519
        """
        tr = Trace(np.zeros(1000))
        tr.stats.delta = 0.01
        tr.stats.station = 'XXX'
        tr.stats.sac = {'stla': 10., 'stlo': -5., 'a': 12.34}
        with io.BytesIO() as tf:
            tr.write(tf, format='SAC')
            tf.seek(0)
            tr1 = read(tf)[0]
        assert round(abs(tr1.stats.sac.stla-10.), 4) == 0
        assert round(abs(tr1.stats.sac.stlo--5.), 4) == 0
        assert round(abs(tr1.stats.sac.a-12.34), 5) == 0

    def test_always_sac_reftime(self):
        """
        Writing a SAC file from a .stats.sac with no reference time should
        still write a SAC file with a reference time.
        """
        reftime = UTCDateTime('2010001')
        a = 12.34
        b = 0.0
        tr = Trace(np.zeros(1000))
        tr.stats.delta = 0.01
        tr.stats.station = 'XXX'
        tr.stats.starttime = reftime
        tr.stats.sac = {}
        tr.stats.sac['a'] = a
        tr.stats.sac['b'] = b
        with io.BytesIO() as tf:
            tr.write(tf, format='SAC')
            tf.seek(0)
            tr1 = read(tf)[0]
        assert tr1.stats.starttime == reftime
        assert round(abs(tr1.stats.sac.a-a), 5) == 0
        assert tr1.stats.sac.b == b

    def test_wrong_encoding(self):
        """
        Read SAC file with wrong encoding
        """
        tr0 = read(self.file_encode)[0]
        assert tr0.stats.get('channel') == '????????'

    def test_encoding_flag(self):
        """
        Test passing encoding flag through obspy.read
        """
        tr0 = read(self.file_encode, encoding='cp1252')[0]
        assert tr0.stats.get('channel') == 'ÇÏÿÿÇÏÿÿ'

    def test_write_keep_sac_header_false(self):
        """
        We expect a Trace that came from a SAC file and trimmed to write a SAC
        file with iztype 'ib' and 'b = 0'.

        https://github.com/obspy/obspy/issues/2760#issuecomment-735555420

        "If you observe the stats.starttime is not the same that the
        stats.sac information. The stats.sac information is the same that the
        original file. If you write this SAC file and open with sac iris, the
        information of the file is the stats.sac ."

        1. build the SAC file, and write it to file
        2. read it and trim it in ObsPy
        3. compare the trimmed header

        """
        hdr = {
            "delta": 0.1,
            "depmin": -55782.0,
            "depmax": 54891.0,
            "scale": 1.0,
            "b": 0.0,
            "e": 87004.1,
            "stla": -22.85915,
            "stlo": -69.80658,
            "stel": 1407.0,
            "stdp": 0.0,
            "depmen": 140.41512,
            "cmpaz": 90.0,
            "cmpinc": 0.0,
            "nzyear": 2007,
            "nzjday": 365,
            "nzhour": 23,
            "nzmin": 49,
            "nzsec": 55,
            "nzmsec": 900,
            "nvhdr": 6,
            "npts": 870041,
            "iftype": 'itime',
            "iztype": 'ib',
            "leven": 1,
            "lpspol": 1,
            "lovrok": 1,
            "lcalda": 0,
            "kstnm": "ALG",
            "kcmpnm": "HHE",
            "knetwk": "Y9",
        }
        data = np.zeros(hdr['npts'], dtype=np.float32)
        data[-1] = hdr['depmin']
        data[-2] = hdr['depmax']
        sac = SACTrace(data=data, **hdr)
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac.write(tempfile)
            tr0 = read(tempfile, format='SAC')[0]

        tr_trimmed = tr0.trim(
            starttime=UTCDateTime(2008, 1, 1, 0, 0),
            endtime=UTCDateTime(2008, 1, 1, 23, 59, 59, 800000),
            fill_value=None,
            nearest_sample=True,
            pad=False,
            )

        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            tr_trimmed.write(tempfile, format='SAC', keep_sac_header=False)
            tr_trimmed_read = read(tempfile, format='SAC')[0]

        # the SAC iztype was 'ib', so the reference time of the written/read
        # SAC file should be the first sample time of the trimmed trace.
        assert tr_trimmed_read.stats.sac['nzyear'] == \
            tr_trimmed.stats.starttime.year
        assert tr_trimmed_read.stats.sac['nzjday'] == \
            tr_trimmed.stats.starttime.julday
        assert tr_trimmed_read.stats.sac['nzhour'] == \
            tr_trimmed.stats.starttime.hour
        assert tr_trimmed_read.stats.sac['nzmin'] == \
            tr_trimmed.stats.starttime.minute
        assert tr_trimmed_read.stats.sac['nzsec'] == \
            tr_trimmed.stats.starttime.second
        assert tr_trimmed_read.stats.sac['nzmsec'] == \
            tr_trimmed.stats.starttime.microsecond * 1000

    def test_sampling_rate_float_issue(self, testdata):
        """
        Test for rounding issues when reading SAC which stores sampling rate as
        a single precision floating point represntation of the sample spacing
        in seconds. See #3408.
        """
        path = testdata['sample_spacing_rounding_issue.sac']
        msg = (r'Sample spacing read from SAC file \(0.040000003 when rounded '
               r'to nanoseconds\) was rounded of to microsecond precision '
               r'\(0.040000000\) to avoid floating point issues when '
               r'converting to sampling rate \(see #3408\)')
        # test with rounding (default)
        for func in (read, _read_sac):
            with CatchAndAssertWarnings(expected=[(UserWarning, msg)]):
                tr = func(path, format="SAC")[0]
            assert abs(25.0 - tr.stats.sampling_rate) < 1e-6
        # test without rounding
        for func in (read, _read_sac):
            tr = func(path, format="SAC", round_sampling_interval=False)[0]
        assert abs(25.0 - tr.stats.sampling_rate) > 1e-6
