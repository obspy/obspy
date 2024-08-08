# -*- coding: utf-8 -*-
import io
import datetime
import warnings
import random

import numpy as np

from obspy import UTCDateTime, read
from obspy.core.util import NamedTemporaryFile
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

from .. import header as _hd
from ..sactrace import SACTrace
from ..util import SacHeaderError, SacHeaderTimeError
import pytest


class TestSACTrace():
    """
    Test suite for obspy.io.sac.sactrace
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        self.file = testdata['test.sac']
        self.filexy = testdata['testxy.sac']
        self.filebe = testdata['test.sac.swap']
        self.fileseis = testdata['seism.sac']
        self.testdata = np.array(
            [-8.74227766e-08, -3.09016973e-01,
             -5.87785363e-01, -8.09017122e-01, -9.51056600e-01,
             -1.00000000e+00, -9.51056302e-01, -8.09016585e-01,
             -5.87784529e-01, -3.09016049e-01], dtype=np.float32)

    def test_read_binary(self):
        """
        Tests for SACTrace binary file read
        """
        sac = SACTrace.read(self.file, byteorder='little')
        assert sac.npts == 100
        assert sac.kstnm == 'STA'
        assert sac.delta == 1.0
        assert sac.kcmpnm == 'Q'
        assert sac.reftime.datetime == datetime.datetime(1978, 7, 18, 8, 0)
        assert sac.nvhdr == 6
        assert sac.b == 10.0
        assert round(abs(sac.depmen-9.0599059e-8), 7) == 0
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             sac.data[0:10])

    def test_read_binary_headonly(self):
        """
        A headonly read should return readable headers and data is None
        """
        sac = SACTrace.read(self.file, byteorder='little', headonly=True)
        assert sac.data is None
        assert sac.npts == 100
        assert sac.depmin == -1.0
        assert round(abs(sac.depmen-8.344650e-8), 7) == 0
        assert sac.depmax == 1.0

    def test_read_sac_byteorder(self):
        """
        A read should fail if the byteorder is wrong
        """
        with pytest.raises(IOError):
            SACTrace.read(self.filebe, byteorder='little')
        with pytest.raises(IOError):
            SACTrace.read(self.file, byteorder='big')
        # a SACTrace should show the correct byteorder
        sac = SACTrace.read(self.filebe, byteorder='big')
        assert sac.byteorder == 'big'
        sac = SACTrace.read(self.file, byteorder='little')
        assert sac.byteorder == 'little'
        # a SACTrace should autodetect the correct byteorder
        sac = SACTrace.read(self.file)
        assert sac.byteorder == 'little'
        sac = SACTrace.read(self.filebe)
        assert sac.byteorder == 'big'

    def test_write_sac(self):
        """
        A trace you've written and read in again should look the same as the
        one you started with.
        """
        sac1 = SACTrace.read(self.file, byteorder='little')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac1.write(tempfile, byteorder='little')
            sac2 = SACTrace.read(tempfile, byteorder='little')
        np.testing.assert_array_equal(sac1.data, sac2.data)
        assert sac1._header == sac2._header

    def test_write_binary_headonly(self):
        """
        A trace you've written headonly should only modify the header of an
        existing file, and fail if the file doesn't exist.
        """
        # make a sac trace
        sac = SACTrace.read(self.file, byteorder='little')
        # write it all to temp file
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            sac.write(tempfile, byteorder='little')
            # read it headonly and modify the header
            # modify the data, too, and verify it didn't get written
            sac2 = SACTrace.read(tempfile, headonly=True, byteorder='little')
            sac2.kcmpnm = 'xyz'
            sac2.b = 7.5
            sac2.data = np.array([1.5, 2e-3, 17], dtype=np.float32)
            # write it again (write over)
            sac2.write(tempfile, headonly=True, byteorder='little')
            # read it all and compare
            sac3 = SACTrace.read(tempfile, byteorder='little')
        assert sac3.kcmpnm == 'xyz'
        assert sac3.b == 7.5
        np.testing.assert_array_equal(sac3.data, sac.data)

        # ...and fail if the file doesn't exist
        sac = SACTrace.read(self.file, headonly=True, byteorder='little')
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
        with pytest.raises(IOError):
            sac.write(tempfile, headonly=True, byteorder='little')

    def test_read_sac_ascii(self):
        """
        Read an ASCII SAC file.
        """
        sac = SACTrace.read(self.filexy, ascii=True)
        assert sac.npts == 100
        assert sac.kstnm == 'sta'
        assert sac.delta == 1.0
        assert sac.kcmpnm == 'Q'
        assert sac.nvhdr == 6
        assert sac.b == 10.0
        assert round(abs(sac.depmen-9.4771387e-08), 7) == 0
        np.testing.assert_array_almost_equal(self.testdata[0:10],
                                             sac.data[0:10])

    def test_reftime(self):
        """
        A SACTrace.reftime should be created correctly from a file's nz-times
        """
        sac = SACTrace.read(self.fileseis)
        assert sac.reftime == UTCDateTime('1981-03-29T10:38:14.000000Z')
        # changes to a reftime should be reflected in the nz times and reftime
        nzsec, nzmsec = sac.nzsec, sac.nzmsec
        sac.reftime = sac.reftime + 2.5
        assert sac.nzsec == nzsec + 2
        assert sac.nzmsec == nzmsec + 500
        assert sac.reftime == UTCDateTime('1981-03-29T10:38:16.500000Z')
        # changes in the nztimes should be reflected reftime
        sac.nzyear = 2001
        assert sac.reftime.year == 2001

    def test_reftime_relative_times(self):
        """
        Changes in the reftime shift all relative time headers
        """
        sac = SACTrace.read(self.fileseis)
        a, b, t1 = sac.a, sac.b, sac.t1
        sac.reftime -= 10.0
        assert round(abs(sac.a - (a + 10.0)), 5) == 0
        assert round(abs(sac.b - (b + 10.0)), 7) == 0
        assert round(abs(sac.t1 - (t1 + 10.0)), 7) == 0
        # changes in the reftime should push remainder microseconds to the
        # relative time headers, and milliseconds to the nzmsec
        sac = SACTrace(b=5.0, t1=20.0)
        b, t1, nzmsec = sac.b, sac.t1, sac.nzmsec
        sac.reftime += 1.2e-3
        assert sac.nzmsec == nzmsec + 1
        assert round(abs(sac.b - (b - 1.0e-3)), 6) == 0
        assert round(abs(sac.t1 - (t1 - 1.0e-3)), 5) == 0

    def test_lcalda(self):
        """
        Test that distances are set when geographic information is present and
        lcalda is True, and that they're not set when geographic information
        is missing or lcalca is false.
        """
        stla, stlo, evla, evlo = -35.0, 100, 42.5, -37.5
        meters, az, baz = gps2dist_azimuth(evla, evlo, stla, stlo)
        km = meters / 1000.0
        gcarc = kilometer2degrees(km)

        # distances are set when lcalda True and all distance values set
        sac = SACTrace(lcalda=True, stla=stla, stlo=stlo, evla=evla, evlo=evlo)
        assert round(abs(sac.az-az), 4) == 0
        assert round(abs(sac.baz-baz), 4) == 0
        assert round(abs(sac.dist-km), 4) == 0
        assert round(abs(sac.gcarc-gcarc), 4) == 0
        # distances are not set when lcalda False and all distance values set
        sac = SACTrace(lcalda=False, stla=stla, stlo=stlo, evla=evla,
                       evlo=evlo)
        assert sac.az is None
        assert sac.baz is None
        assert sac.dist is None
        assert sac.gcarc is None
        # distances are not set when lcalda True, not all distance values set
        sac = SACTrace(lcalda=True, stla=stla)
        assert sac.az is None
        assert sac.baz is None
        assert sac.dist is None
        assert sac.gcarc is None
        # exception raised when set_distances is forced but not all distances
        # values are set. NOTE: still have a problem when others are "None".
        sac = SACTrace(lcalda=True, stla=stla)
        with pytest.raises(SacHeaderError):
            sac._set_distances(force=True)

    def test_propagate_modified_stats_strings_to_sactrace(self):
        """
        If you build a SACTrace from an ObsPy Trace that has certain string
        headers mismatched between the Stats header and an existing Stats.sac
        header, channel and kcmpnm for example, the resulting SACTrace values
        should come from Stats. Addresses GitHub issue #1457.
        """
        tr = read(self.fileseis)[0]
        # modify the header values by adding a single meaningless character
        for sachdr, statshdr in [('kstnm', 'station'), ('knetwk', 'network'),
                                 ('kcmpnm', 'channel'), ('khole', 'location')]:
            modified_value = tr.stats[statshdr] + '1'
            tr.stats[statshdr] = modified_value
            sac = SACTrace.from_obspy_trace(tr)
            assert getattr(sac, sachdr) == modified_value

    def test_reftime_incomplete(self):
        """
        Replacement for SACTrace._from_arrays doctest which raises UserWarning
        """
        sac = SACTrace._from_arrays()
        assert sac.lcalda
        assert not sac.leven
        assert not sac.lovrok
        assert not sac.lpspol
        assert sac.iztype is None
        with pytest.raises(SacHeaderTimeError):
            getattr(sac, 'reftime')
        # raises "UserWarning: Reference time information incomplete"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            str(sac)
            assert len(w) == 1
            assert "Reference time information incomplete" in str(w[0])

    def test_floatheader(self):
        """
        Test standard SACTrace float headers using the floatheader descriptor.
        """
        sac = SACTrace()
        for hdr in ('delta', 'scale', 'odelta', 'internal0', 'stel', 'stdp',
                    'evdp', 'mag', 'user0', 'user1', 'user2', 'user3', 'user4',
                    'user5', 'user6', 'user7', 'user8', 'user9', 'dist', 'az',
                    'baz', 'gcarc', 'cmpaz', 'cmpinc'):
            floatval = random.random()

            # setting value
            setattr(sac, hdr, floatval)
            assert round(
                abs(sac._hf[_hd.FLOATHDRS.index(hdr)]-floatval), 7) == 0
            # getting value
            assert round(abs(getattr(sac, hdr)-floatval), 7) == 0
            # setting None produces null value
            setattr(sac, hdr, None)
            assert round(
                abs(sac._hf[_hd.FLOATHDRS.index(hdr)]-_hd.FNULL), 7) == 0
            # getting existing null values return None
            sac._hf[_hd.FLOATHDRS.index(hdr)] = _hd.FNULL
            assert getattr(sac, hdr) is None
            # __doc__ on class and instance
            assert getattr(SACTrace, hdr).__doc__ == _hd.DOC.get(hdr)
            # self.assertEqual(getattr(sac, hdr).__doc__, _hd.DOC.get(hdr]))
            # TODO: I'd like to find a way for this to work:-(
            # TODO: factor __doc__ tests out into one test for all headers

    def test_relative_time_headers(self):
        """
        Setting relative time headers will work with UTCDateTime objects.
        """
        # TODO: ultimately, _all_ children of floatheader (this one, geosetter,
        #   etc.) should be tested in test_floatheader for normal setting, and
        #   only special behaviour will happen here.
        utc = UTCDateTime(year=1970, month=1, day=1, minute=15, second=10,
                          microsecond=0)
        sac = SACTrace(nzyear=utc.year, nzjday=utc.julday, nzhour=utc.hour,
                       nzmin=utc.minute, nzsec=utc.second, nzmsec=0)
        for hdr in ('b', 'a', 'o', 'f', 't0', 't1', 't2', 't3', 't4', 't5',
                    't6', 't7', 't8', 't9'):
            offset_float = random.uniform(-1, 1)
            offset_utc = utc + offset_float
            setattr(sac, hdr, offset_utc)
            assert round(
                abs(sac._hf[_hd.FLOATHDRS.index(hdr)]-offset_float), 5) == 0

    def test_int_headers(self):
        """
        Test standard SACTrace int headers using the intheader descriptor.
        """
        sac = SACTrace()
        for hdr in ('nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec',
                    'nvhdr', 'norid', 'nevid', 'nwfid', 'iinst', 'istreg',
                    'ievreg', 'iqual', 'unused23'):

            intval = random.randint(-10, 10)

            # setting value
            setattr(sac, hdr, intval)
            assert sac._hi[_hd.INTHDRS.index(hdr)] == intval
            # getting value
            assert getattr(sac, hdr) == intval
            # setting None produces null value
            setattr(sac, hdr, None)
            assert sac._hi[_hd.INTHDRS.index(hdr)] == _hd.INULL
            # getting existing null values return None
            sac._hi[_hd.INTHDRS.index(hdr)] = _hd.INULL
            assert getattr(sac, hdr) is None
            # __doc__ on class and instance
            assert getattr(SACTrace, hdr).__doc__ == _hd.DOC.get(hdr)

    def test_bool_headers(self):
        sac = SACTrace()
        for hdr in ('leven', 'lpspol', 'lovrok', 'lcalda'):
            # getting existing null values return None
            sac._hi[_hd.INTHDRS.index(hdr)] = _hd.INULL
            assert getattr(sac, hdr) is None

            for boolval in (True, False, 0, 1):
                setattr(sac, hdr, boolval)
                assert sac._hi[_hd.INTHDRS.index(hdr)] == int(boolval)
                assert getattr(sac, hdr) == bool(boolval)

    def test_enumheader(self):
        sac = SACTrace()
        # set all the `iztype` reference headers, so that it won't fail
        for idx, hdr in enumerate(['b', 'o', 'a', 'f'] +
                                  ['t' + str(i) for i in range(10)]):
            setattr(sac, hdr, idx)
        for enumhdr, accepted_vals in _hd.ACCEPTED_VALS.items():
            if enumhdr != 'iqual':
                # iqual is allowed to be integer, not an enumerated str
                for accepted_val in accepted_vals:
                    accepted_int = _hd.ENUM_VALS[accepted_val]

                    sac._hi[_hd.INTHDRS.index(enumhdr)] = accepted_int
                    assert getattr(sac, enumhdr) == accepted_val

                    setattr(sac, enumhdr, accepted_val)
                    assert sac._hi[_hd.INTHDRS.index(enumhdr)] == \
                        accepted_int

    def test_string_headers(self):
        sac = SACTrace()
        for hdr in ('kstnm', 'khole', 'ko', 'ka', 'kt0', 'kt1', 'kt2', 'kt3',
                    'kt4', 'kt5', 'kt6', 'kt7', 'kt8', 'kt9', 'kf', 'kuser0',
                    'kuser1', 'kuser2', 'kcmpnm', 'knetwk', 'kdatrd',
                    'kinst'):

            strval = hdr

            # normal get/set
            setattr(sac, hdr, strval)
            assert sac._hs[_hd.STRHDRS.index(hdr)].decode() == strval
            assert getattr(sac, hdr) == strval

            # null get/set
            sac._hs[_hd.STRHDRS.index(hdr)] = _hd.SNULL
            assert getattr(sac, hdr) is None

            # get/set value too long
            too_long = "{}_1234567890".format(hdr)
            # will raise "UserWarning: Alphanumeric headers longer than 8
            # characters are right-truncated"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', UserWarning)
                setattr(sac, hdr, too_long)
                assert len(w) == 1
                assert w[0].category == UserWarning
                assert 'Alphanumeric headers longer than 8' in str(w[0])
            assert sac._hs[_hd.STRHDRS.index(hdr)].decode() == too_long[:8]
            assert getattr(sac, hdr) == too_long[:8].strip()

            # docstring
            assert getattr(SACTrace, hdr).__doc__ == _hd.DOC.get(hdr)

    def test_kevnm(self):
        sac = SACTrace()
        # test kevnm (kevnm + kevnm2)
        kevnm = '1234567890123456'
        kevnm1, kevnm2 = kevnm[:8], kevnm[8:]

        sac.kevnm = kevnm
        assert sac._hs[_hd.STRHDRS.index('kevnm')].decode() == kevnm1
        assert sac._hs[_hd.STRHDRS.index('kevnm2')].decode() == kevnm2
        assert sac.kevnm == kevnm

        sac._hs[_hd.STRHDRS.index('kevnm')] = _hd.SNULL
        sac._hs[_hd.STRHDRS.index('kevnm2')] = _hd.SNULL
        assert sac.kevnm is None

        assert SACTrace.kevnm.__doc__ == _hd.DOC.get('kevnm')

    def test_data_headers(self):
        """
        Headers that depend on the data vector should return values operating
        on the data, or fall back to stored header values if data is absent.
        """
        data = np.random.ranf(10).astype(np.float32)
        npts = 4
        depmax = 3.0
        depmen = 2.0
        depmin = 1.0
        sac = SACTrace(depmin=depmin, depmen=depmen, depmax=depmax, npts=npts,
                       data=data)

        for hdr, func in [('depmin', min), ('depmen', np.mean),
                          ('depmax', max), ('npts', len)]:
            # getting value
            assert getattr(sac, hdr) == func(data)

            with pytest.raises(AttributeError):
                # can't set value on write-only attribute
                setattr(sac, hdr, func(data))

        # headers fall back to stored value when data is None
        sac.data = None
        for hdr, value in [('depmin', depmin), ('depmen', depmen),
                           ('depmax', depmax), ('npts', npts)]:
            assert getattr(sac, hdr) == value

    def test_char_header_padding(self):
        """
        SAC binary file header fields should be padded with blanks, not NULs,
        or NUL-terminated.
        """

        dat = np.zeros(8)

        sac = SACTrace(
            kstnm='TEST',
            delta=1,
            kcmpnm='Z',
            kevnm='Bug test',
            npts=len(dat),
            data=dat
        )
        with io.BytesIO() as bio:
            sac.write(bio, byteorder='big')
            bio.seek(0)
            result = bio.read()
        # kstnm is at offset 0x1b8 in the file
        assert result[0x1b8:(0x1b8+8)] == b'TEST    '
        # kcmpnm is at offset 0x258 in the file
        assert result[0x258:(0x258+8)] == b'Z       '
