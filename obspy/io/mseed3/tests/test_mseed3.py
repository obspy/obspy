# -*- coding: utf-8 -*-
"""
Tests for obspy.io.mseed3.
"""

import io
import os

import numpy as np
import pytest

from obspy import Stream, Trace, UTCDateTime, read
from obspy.core import Stats
from obspy.io.mseed3.core import _is_mseed3, _read_mseed3, _write_mseed3

# --- Expected values for shared reference files --------------------------

# A 500-point sine that expands with index; the first 4 truncated-int
# samples are 0, 6, 10, 10 across encodings (the libmseed reference set).
SINE_INT_FIRST4 = [0, 6, 10, 10]
SINE_INT_LAST4 = [-833501145, -866584864, -556206270, 0]

# Expected number of samples per channel for testdata-3channel-signal
THREECH_NPTS = 4200
THREECH_IDS = ("IU.COLA.00.LH1", "IU.COLA.00.LH2", "IU.COLA.00.LHZ")
THREECH_START = UTCDateTime("2010-02-27T06:50:00.069539Z")

# First/last samples decoded from a single record of LH1 in the
# 3-channel file.
THREECH_LH1_FIRST4 = [-502676, -504105, -507491, -506991]


def _make_trace(data, **stats_overrides):
    """Build a Trace with sensible defaults for write tests."""
    stats = Stats()
    stats.network = "XX"
    stats.station = "TEST"
    stats.location = ""
    stats.channel = "BHZ"
    stats.sampling_rate = 40.0
    stats.starttime = UTCDateTime("2012-05-12T00:00:00")
    stats.npts = len(data)
    for k, v in stats_overrides.items():
        setattr(stats, k, v)
    return Trace(data=np.asarray(data), header=stats)


class TestIsMSEED3:
    """Format-detection probe (_is_mseed3)."""

    def test_detects_v3_file(self, testdata):
        assert _is_mseed3(testdata["testdata-3channel-signal.mseed3"]) is True

    def test_detects_v2_file(self, testdata):
        assert _is_mseed3(testdata["testdata-3channel-signal.mseed2"]) is True

    def test_rejects_non_mseed_file(self, tmp_path):
        bogus = tmp_path / "not_mseed.bin"
        bogus.write_bytes(b"this is not miniSEED data, definitely not")
        assert _is_mseed3(str(bogus)) is False

    def test_rejects_short_bytes(self):
        assert _is_mseed3(b"abc") is False

    def test_accepts_bytes_input(self, testdata):
        data = testdata["testdata-3channel-signal.mseed3"].read_bytes()
        assert _is_mseed3(data) is True
        assert _is_mseed3(bytearray(data)) is True
        assert _is_mseed3(memoryview(data)) is True

    def test_accepts_file_like(self, testdata):
        with open(testdata["testdata-3channel-signal.mseed3"], "rb") as fh:
            assert _is_mseed3(fh) is True

    def test_rejects_missing_file(self, tmp_path):
        assert _is_mseed3(str(tmp_path / "does-not-exist.mseed3")) is False


class TestReadMSEED3:
    """Reading via _read_mseed3 and the obspy.read plugin entry point."""

    # ---- basic 3-channel parse (v3 + v2) ----

    def _check_threech(self, st):
        assert len(st) == 3
        ids = tuple(tr.id for tr in st)
        assert ids == THREECH_IDS
        for tr in st:
            assert tr.stats.sampling_rate == 1.0
            assert tr.stats.npts == THREECH_NPTS
            assert tr.stats.starttime == THREECH_START
            assert tr.data.dtype == np.int32
            assert tr.stats.mseed3.source_id.startswith("FDSN:IU_COLA_")

    def test_read_v3_three_channel(self, testdata):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        self._check_threech(st)
        assert st[0].data[:4].tolist() == THREECH_LH1_FIRST4

    def test_read_v2_three_channel(self, testdata):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed2"])
        self._check_threech(st)
        assert st[0].data[:4].tolist() == THREECH_LH1_FIRST4

    def test_read_via_obspy_format_kwarg(self, testdata):
        st = read(str(testdata["testdata-3channel-signal.mseed3"]), format="MSEED3")
        self._check_threech(st)

    def test_read_via_obspy_autodetect(self, testdata):
        # The v3 file must be detected when no ``format=`` is supplied.
        st = read(str(testdata["reference-testdata-int32.mseed3"]))
        assert len(st) == 1
        assert st[0].stats.npts == 500
        assert st[0].id == "XX.TEST..BHZ"

    # ---- v3 encodings -------------------------------------------------

    def test_read_v3_int32(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-int32.mseed3"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-4:].tolist() == SINE_INT_LAST4

    def test_read_v3_int16(self, testdata):
        # int16-encoded data are returned as int32.
        st = _read_mseed3(testdata["reference-testdata-int16.mseed3"])
        tr = st[0]
        # Only ~220 samples fit into 16-bit dynamic range.
        assert tr.stats.npts == 220
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4

    def test_read_v3_float32(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-float32.mseed3"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.float32
        assert tr.data[0] == 0.0
        # Compare against truncated integer-form of the sine series
        assert np.allclose(
            tr.data[:4], [0.0, 6.109208, 10.246826, 10.609957], atol=1e-4
        )

    def test_read_v3_float64(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-float64.mseed3"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.float64
        assert np.allclose(tr.data[:4], [0.0, 6.109208, 10.246826, 10.609957])

    def test_read_v3_steim1(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim1.mseed3"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-4:].tolist() == SINE_INT_LAST4

    def test_read_v3_steim2(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim2.mseed3"])
        tr = st[0]
        # Steim-2 reference file holds one fewer sample: the original
        # trailing 0 falls outside Steim-2's 30-bit difference range and
        # is dropped by the encoder. Only verify the surviving tail.
        assert tr.stats.npts == 499
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-3:].tolist() == SINE_INT_LAST4[:-1]

    def test_read_v3_text(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-text.mseed3"])
        tr = st[0]
        assert tr.id == "XX.TEST..LOG"
        assert tr.data.dtype == np.dtype("|S1")
        text = b"".join(tr.data.tolist()).decode("utf-8", errors="replace")
        assert text.startswith("I've seen things")
        assert "Time to die." in text

    # ---- v2 encodings -------------------------------------------------

    def test_read_v2_int32(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-int32.mseed2"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-4:].tolist() == SINE_INT_LAST4

    def test_read_v2_int16(self, testdata):
        # int16-encoded data are returned as int32.
        st = _read_mseed3(testdata["reference-testdata-int16.mseed2"])
        tr = st[0]
        # Only ~220 samples fit into 16-bit dynamic range.
        assert tr.stats.npts == 220
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4

    def test_read_v2_float32(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-float32.mseed2"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.float32
        assert tr.data[0] == 0.0
        # Compare against truncated integer-form of the sine series
        assert np.allclose(
            tr.data[:4], [0.0, 6.109208, 10.246826, 10.609957], atol=1e-4
        )

    def test_read_v2_float64(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-float64.mseed2"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.float64
        assert np.allclose(tr.data[:4], [0.0, 6.109208, 10.246826, 10.609957])

    def test_read_v2_steim1(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim1.mseed2"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-4:].tolist() == SINE_INT_LAST4

    def test_read_v2_steim2(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim2.mseed2"])
        tr = st[0]
        # Steim-2 reference file holds one fewer sample: the original
        # trailing 0 falls outside Steim-2's 30-bit difference range and
        # is dropped by the encoder. Only verify the surviving tail.
        assert tr.stats.npts == 499
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-3:].tolist() == SINE_INT_LAST4[:-1]

    # Little-endian Steim-1 and Steim-2 files are supported even though
    # they are not part of the official SEED standard.
    def test_read_v2_steim1_le(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim1-LE.mseed2"])
        tr = st[0]
        assert tr.stats.npts == 500
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-4:].tolist() == SINE_INT_LAST4

    def test_read_v2_steim2_le(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-steim2-LE.mseed2"])
        tr = st[0]
        # Steim-2 reference file holds one fewer sample: the original
        # trailing 0 falls outside Steim-2's 30-bit difference range and
        # is dropped by the encoder. Only verify the surviving tail.
        assert tr.stats.npts == 499
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == SINE_INT_FIRST4
        assert tr.data[-3:].tolist() == SINE_INT_LAST4[:-1]

    def test_read_v2_text(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-text.mseed2"])
        tr = st[0]
        assert tr.id == "XX.TEST..LOG"
        assert tr.data.dtype == np.dtype("|S1")
        text = b"".join(tr.data.tolist()).decode("utf-8", errors="replace")
        assert text.startswith("I've seen things")
        assert "Time to die." in text

    # ---- Legacy v2 encodings (read only) ---------------------------------

    def test_read_v2_cdsn(self, testdata):
        st = _read_mseed3(testdata["testdata-encoding-CDSN.mseed2"])
        tr = st[0]
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == [-96, -87, -100, -128]
        assert tr.data[-4:].tolist() == [-205, -205, -196, -185]

    def test_read_v2_dwwssn(self, testdata):
        st = _read_mseed3(testdata["testdata-encoding-DWWSSN.mseed2"])
        tr = st[0]
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == [6, 5, 1, -9]
        assert tr.data[-4:].tolist() == [66, 71, 77, 76]

    def test_read_v2_geoscope16_3exp(self, testdata):
        st = _read_mseed3(
            testdata["testdata-encoding-GEOSCOPE-16bit-3exp-encoded.mseed2"]
        )
        tr = st[0]
        assert tr.data.dtype == np.float32
        assert np.allclose(tr.data[:4], [-1.0625, -1.078125, -1.078125, -1.078125])
        assert np.allclose(
            tr.data[-4:], [-1.1640625, -1.1640625, -1.1640625, -1.1640625]
        )

    def test_read_v2_sro(self, testdata):
        st = _read_mseed3(testdata["testdata-encoding-SRO.mseed2"])
        tr = st[0]
        assert tr.data.dtype == np.int32
        assert tr.data[:4].tolist() == [39, 42, 32, 1]
        assert tr.data[-4:].tolist() == [-45, -31, -26, -36]

    # ---- exotic time / sampling-rate values ---------------------------

    def test_read_v3_oddrate(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-oddrate.mseed3"])
        assert st[0].stats.sampling_rate == 1080.0

    def test_read_v2_oddrate(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-oddrate.mseed2"])
        assert st[0].stats.sampling_rate == 1080.0

    def test_read_nsec_precision(self, testdata):
        # The "nsec" reference has a sub-microsecond starttime offset.
        st = _read_mseed3(testdata["reference-testdata-nsec.mseed3"])
        # Round-trip via .ns to preserve nanosecond precision.
        assert st[0].stats.starttime.ns == 1336780800123456789

    def test_read_v3_olden(self, testdata):
        # Pre-epoch / very-old start time
        st = _read_mseed3(testdata["reference-testdata-olden.mseed3"])
        assert st[0].stats.starttime.year == 1964

    def test_read_v2_olden(self, testdata):
        st = _read_mseed3(testdata["reference-testdata-olden.mseed2"])
        assert st[0].stats.starttime.year == 1964

    def test_read_v2_no_blockette_1000(self, testdata):
        # With no Blockette 1000, the record length is inferred from the data
        # structure (next header or end of file) and the encoding falls back to STEIM1.
        st = _read_mseed3(testdata["testdata-no-blockette1000-steim1.mseed2"])
        assert st[0].stats.npts == 7312
        assert st[0].data.dtype == np.int32
        assert st[0].data[:4].tolist() == [337, 396, 454, 503]
        assert st[0].data[-4:].tolist() == [226, 175, 116, 70]

    def test_read_v2_unapplied_time_correction(self, testdata):
        st = _read_mseed3(testdata["testdata-unapplied-timecorrection.mseed2"])
        # The unapplied 1.0 second time correction is applied to the starttime
        assert st[0].stats.starttime == UTCDateTime("2003-05-29T02:13:23.043400Z")

    # ---- headonly -----------------------------------------------------

    def test_read_headonly(self, testdata):
        st = _read_mseed3(
            testdata["reference-testdata-headeronly.mseed2"], headonly=True
        )
        assert len(st) == 1
        assert st[0].id == "XX.TEST..SOH"
        assert st[0].stats.npts == 0
        assert len(st[0].data) == 0

        st = _read_mseed3(
            testdata["reference-testdata-headeronly.mseed3"], headonly=True
        )
        assert len(st) == 1
        assert st[0].id == "XX.TEST..SOH"
        assert st[0].stats.npts == 0
        assert len(st[0].data) == 0

        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"], headonly=True)
        assert len(st) == 3
        for tr in st:
            # stats.npts reports the *recorded* sample count, but no
            # data samples were unpacked.
            assert tr.stats.npts == THREECH_NPTS
            assert len(tr.data) == 0

    # ---- twopass ------------------------------------------------------

    def test_read_twopass_matches_default(self, testdata):
        st1 = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        st2 = _read_mseed3(testdata["testdata-3channel-signal.mseed3"], twopass=True)
        assert len(st1) == len(st2)
        for tr1, tr2 in zip(st1, st2):
            assert tr1.id == tr2.id
            np.testing.assert_array_equal(tr1.data, tr2.data)
        # twopass populates the per-segment record count
        for tr in st2:
            assert tr.stats.mseed3.number_of_records > 0

    # ---- filtering ----------------------------------------------------

    def test_sourceid_filter(self, testdata):
        st = _read_mseed3(
            testdata["testdata-3channel-signal.mseed3"], sourceid="FDSN:IU_COLA_*_L_H_Z"
        )
        assert len(st) == 1
        assert st[0].id == "IU.COLA.00.LHZ"

    def test_sourcename_full_nslc(self, testdata):
        st = _read_mseed3(
            testdata["testdata-3channel-signal.mseed3"], sourcename="IU.COLA.00.LHZ"
        )
        assert len(st) == 1
        assert st[0].id == "IU.COLA.00.LHZ"

    def test_sourcename_front_anchored_wildcard(self, testdata):
        st = _read_mseed3(
            testdata["testdata-3channel-signal.mseed3"], sourcename="IU.COLA.00.*"
        )
        assert len(st) == 3

    def test_sourcename_and_sourceid_mutually_exclusive(self, testdata):
        with pytest.raises(ValueError, match="Cannot specify both"):
            _read_mseed3(
                testdata["testdata-3channel-signal.mseed3"],
                sourceid="FDSN:*",
                sourcename="IU.*",
            )

    def test_time_window_trim(self, testdata):
        start = UTCDateTime("2010-02-27T06:50:30")
        end = UTCDateTime("2010-02-27T06:50:40")
        st = _read_mseed3(
            testdata["testdata-3channel-signal.mseed3"], starttime=start, endtime=end
        )
        assert len(st) == 3
        for tr in st:
            # Trim uses ``nearest_sample`` semantics so the returned span
            # may overshoot by < 1 sample interval on either side.
            sample_delta = tr.stats.delta
            assert tr.stats.starttime >= start - sample_delta
            assert tr.stats.endtime <= end + sample_delta
            # And the window should actually be narrowed.
            assert tr.stats.npts < THREECH_NPTS

    # ---- alternative input types --------------------------------------

    def test_read_from_bytes(self, testdata):
        data = testdata["testdata-3channel-signal.mseed3"].read_bytes()
        st = _read_mseed3(data)
        self._check_threech(st)

    def test_read_from_memoryview(self, testdata):
        data = testdata["testdata-3channel-signal.mseed3"].read_bytes()
        st = _read_mseed3(memoryview(data))
        self._check_threech(st)

    def test_read_from_bytesio(self, testdata):
        data = testdata["testdata-3channel-signal.mseed3"].read_bytes()
        st = _read_mseed3(io.BytesIO(data))
        self._check_threech(st)

    def test_read_from_open_file(self, testdata):
        with open(testdata["testdata-3channel-signal.mseed3"], "rb") as fh:
            st = _read_mseed3(fh)
        self._check_threech(st)

    def test_read_from_pathlib(self, testdata):
        # ``testdata['name']`` already returns a ``pathlib.Path``.
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        self._check_threech(st)

    # ---- multi-record, mixed lengths ----------------------------------

    def test_read_mixed_lengths_v3(self, testdata):
        st = _read_mseed3(testdata["testdata-oneseries-mixedlengths-mixedorder.mseed3"])
        assert len(st) == 1
        tr = st[0]
        assert tr.id == "XX.TEST.00.LHZ"
        assert tr.stats.npts == 3952
        assert tr.data[:4].tolist() == [-231946, -228438, -223155, -221231]

    def test_read_mixed_lengths_v2(self, testdata):
        st = _read_mseed3(testdata["testdata-oneseries-mixedlengths-mixedorder.mseed2"])
        assert len(st) == 1
        assert st[0].stats.npts == 3952

    # ---- mixformat ----------------------------------------------------

    def test_read_mix_format_versions(self, testdata):
        """
        A single buffer containing both miniSEED v2 and v3 records (the
        same series encoded in each format) must be readable. pymseed
        handles both versions through one parser; concatenating the two
        files should yield two traces with identical metadata and
        samples.
        """
        v2_path = testdata["testdata-oneseries-mixedlengths-mixedorder.mseed2"]
        v3_path = testdata["testdata-oneseries-mixedlengths-mixedorder.mseed3"]
        v2_bytes = v2_path.read_bytes()
        v3_bytes = v3_path.read_bytes()

        # Sanity: each file alone reads as one trace.
        v2_alone = _read_mseed3(v2_bytes)
        v3_alone = _read_mseed3(v3_bytes)
        assert len(v2_alone) == 1
        assert len(v3_alone) == 1
        np.testing.assert_array_equal(v2_alone[0].data, v3_alone[0].data)

        for label, combined in [
            ("v2+v3", v2_bytes + v3_bytes),
            ("v3+v2", v3_bytes + v2_bytes),
        ]:
            st = _read_mseed3(combined)
            assert len(st) == 2, f"{label}: expected 2 traces, got {len(st)}"
            for tr in st:
                assert tr.id == "XX.TEST.00.LHZ"
                assert tr.stats.npts == 3952
                assert tr.stats.starttime == v2_alone[0].stats.starttime
                assert tr.stats.sampling_rate == v2_alone[0].stats.sampling_rate
                np.testing.assert_array_equal(tr.data, v2_alone[0].data)

    # ---- details -------------------------------------------------------

    def test_read_details_homogeneous_v3(self, testdata):
        """details=True populates per-record stats for a 3-channel v3 file."""
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"], details=True)
        assert len(st) == 3
        expected_nrec = [36, 35, 36]
        for tr, nrec in zip(st, expected_nrec):
            assert tr.stats.mseed3.number_of_records == nrec
            assert tr.stats.mseed3.timing_qualities == [100]
            assert tr.stats.mseed3.publication_versions == [4]
            assert tr.stats.mseed3.encodings == ["STEIM-2 integer compression"]

    def test_read_details_heterogeneous_v2(self, testdata):
        """details=True collects one entry per run of identical values.
        The timing-quality file has 101 records each with a distinct quality
        (0..100), so the result has 101 entries covering all values."""
        st = _read_mseed3(testdata["timingquality.mseed"], details=True)
        assert len(st) == 1
        tr = st[0]
        assert tr.id == "BW.BGLD..EHE"
        assert tr.stats.mseed3.number_of_records == 101
        # All 101 records have distinct qualities — none are adjacent repeats —
        # so the full set 0..100 must appear (in file order, not sorted).
        assert set(tr.stats.mseed3.timing_qualities) == set(range(101))
        assert len(tr.stats.mseed3.timing_qualities) == 101
        assert tr.stats.mseed3.publication_versions == [2]
        assert tr.stats.mseed3.encodings == ["STEIM-1 integer compression"]

    def test_read_details_disabled_omits_keys(self, testdata):
        """details=False (default) must not populate the four extra keys."""
        _DETAIL_KEYS = {
            "number_of_records",
            "timing_qualities",
            "publication_versions",
            "encodings",
        }
        st_no_details = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        for tr in st_no_details:
            present = set(tr.stats.mseed3.keys())
            assert present == {"source_id"}, f"Expected only 'source_id', got {present}"

        st_details = _read_mseed3(
            testdata["testdata-3channel-signal.mseed3"], details=True
        )
        for tr in st_details:
            present = set(tr.stats.mseed3.keys())
            assert _DETAIL_KEYS.issubset(present), (
                f"Missing detail keys: {_DETAIL_KEYS - present}"
            )

    # ---- errors -------------------------------------------------------

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises((IOError, OSError)):
            _read_mseed3(str(tmp_path / "nope.mseed3"))


class TestWriteMSEED3:
    """Writing via _write_mseed3 + read/write roundtrips."""

    # ---- roundtrips ---------------------------------------------------

    def test_roundtrip_int32_default(self, testdata, tmp_path):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out))
        st2 = _read_mseed3(str(out))
        assert len(st) == len(st2)
        for tr1, tr2 in zip(st, st2):
            assert tr1.id == tr2.id
            assert tr1.stats.sampling_rate == tr2.stats.sampling_rate
            assert tr1.stats.starttime == tr2.stats.starttime
            np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_roundtrip_float32(self, testdata, tmp_path):
        st = _read_mseed3(testdata["reference-testdata-float32.mseed3"])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out))
        st2 = _read_mseed3(str(out))
        assert st2[0].data.dtype == np.float32
        np.testing.assert_array_equal(st[0].data, st2[0].data)

    def test_roundtrip_float64(self, testdata, tmp_path):
        st = _read_mseed3(testdata["reference-testdata-float64.mseed3"])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out))
        st2 = _read_mseed3(str(out))
        assert st2[0].data.dtype == np.float64
        np.testing.assert_array_equal(st[0].data, st2[0].data)

    def test_roundtrip_format_version_2(self, testdata, tmp_path):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        out = tmp_path / "out.mseed2"
        _write_mseed3(st, str(out), format_version=2)
        st2 = _read_mseed3(str(out))
        assert len(st) == len(st2)
        for tr1, tr2 in zip(st, st2):
            np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_write_int16_promotes_to_int32_on_read(self, tmp_path):
        data = np.array([1, -2, 3, 4, -5, 6, 100, -200], dtype=np.int16)
        st = Stream(traces=[_make_trace(data, sampling_rate=100.0, channel="HHZ")])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out), encoding="INT16")
        st2 = _read_mseed3(str(out))
        assert st2[0].data.dtype == np.int32
        np.testing.assert_array_equal(st2[0].data, data.astype(np.int32))

    def test_write_int64_downcasts_when_in_int32_range(self, tmp_path):
        data = np.array([1, 2, 3, -1000000, 1000000], dtype=np.int64)
        st = Stream(traces=[_make_trace(data, sampling_rate=100.0, channel="HHZ")])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out))
        st2 = _read_mseed3(str(out))
        assert st2[0].data.dtype == np.int32
        np.testing.assert_array_equal(st2[0].data, data.astype(np.int32))

    def test_write_int64_overflow_raises(self):
        data = np.array([2**40], dtype=np.int64)
        st = Stream(traces=[_make_trace(data, sampling_rate=100.0, channel="HHZ")])
        with pytest.raises(ValueError, match="int64"):
            _write_mseed3(st, "/tmp/should_not_be_written.mseed3")

    def test_write_unsupported_dtype_raises(self):
        data = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        st = Stream(traces=[_make_trace(data, sampling_rate=100.0, channel="HHZ")])
        with pytest.raises(ValueError, match="Unsupported data type"):
            _write_mseed3(st, "/tmp/should_not_be_written.mseed3")

    # ---- encoding aliases & validation --------------------------------

    @pytest.mark.parametrize("encoding", ["STEIM1", "STEIM2", 10, 11, "INT32", 3])
    def test_write_encoding_aliases(self, testdata, tmp_path, encoding):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        out = tmp_path / f"out_{encoding}.mseed3"
        _write_mseed3(st, str(out), encoding=encoding)
        st2 = _read_mseed3(str(out))
        for tr1, tr2 in zip(st, st2):
            np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_write_invalid_encoding_raises(self, testdata):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        with pytest.raises(ValueError, match="Unsupported encoding"):
            _write_mseed3(st, "/tmp/x.mseed3", encoding="NOT_AN_ENCODING")

    # ---- alternate destinations ---------------------------------------

    def test_write_to_bytesio(self, testdata):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        buf = io.BytesIO()
        _write_mseed3(st, buf)
        assert buf.tell() > 0
        st2 = _read_mseed3(buf.getvalue())
        assert len(st2) == len(st)
        for tr1, tr2 in zip(st, st2):
            np.testing.assert_array_equal(tr1.data, tr2.data)

    def test_write_max_record_length(self, testdata, tmp_path):
        # Smaller record length should yield more (but still valid) records.
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        small = tmp_path / "small.mseed3"
        large = tmp_path / "large.mseed3"
        _write_mseed3(st, str(small), max_record_length=512)
        _write_mseed3(st, str(large), max_record_length=4096)
        assert os.path.getsize(small) > 0
        assert os.path.getsize(large) > 0
        # The reassembled data must be identical regardless of record size.
        st_small = _read_mseed3(str(small))
        st_large = _read_mseed3(str(large))
        for a, b in zip(st_small, st_large):
            np.testing.assert_array_equal(a.data, b.data)

    def test_write_overwrite_truncates(self, testdata, tmp_path):
        st = _read_mseed3(testdata["testdata-3channel-signal.mseed3"])
        out = tmp_path / "out.mseed3"
        _write_mseed3(st, str(out))
        size_first = os.path.getsize(str(out))
        # Re-write with overwrite=True; file should not grow.
        _write_mseed3(st, str(out), overwrite=True)
        assert os.path.getsize(str(out)) == size_first
