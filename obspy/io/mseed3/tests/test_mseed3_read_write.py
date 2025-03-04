import json
from tempfile import NamedTemporaryFile

from simplemseed import FDSNSourceId, MSeed3Record, MSeed3Header
from obspy.io.mseed3.core import  (
    ObsPyMSEED3DataOverflowError,
    MSEED_STATS_KEY,
    PUB_VER_KEY,
)

from obspy import read, Stream, Trace
import numpy as np
import pytest
from io import BytesIO


class TestMSEED3ReadingAndWriting:
    """
    Test everything related to the general reading and writing of mseed3
    files.
    """

    def test_ref_data(self, datapath):
        testdir = datapath / "reference-data"
        for path in testdir.glob("*.mseed3"):
            if path.name == "reference-text.mseed3":
                # 0 not decompressible, can't return empty stream so???
                continue
            jsonpath = path.parent.joinpath(path.name[:-6] + "json")
            stream = read(path, format="MSEED3")
            assert len(stream) == 1
            trace = stream[0]
            with open(jsonpath, "r") as injson:
                jsonrec = json.load(injson)[0]

            assert jsonrec["SampleRate"] == trace.stats.sampling_rate
            assert jsonrec["PublicationVersion"] == trace.stats.get(PUB_VER_KEY)
            sid = FDSNSourceId.fromNslc(
                trace.stats.network,
                trace.stats.station,
                trace.stats.location,
                trace.stats.channel,
            )
            assert jsonrec["SID"] == str(sid)
            if "ExtraHeaders" in jsonrec:
                assert MSEED_STATS_KEY in trace.stats
                assert jsonrec["ExtraHeaders"] == trace.stats.mseed3
            if jsonrec["DataLength"] > 0:
                assert jsonrec["SampleCount"] == len(trace)
                jsondata = jsonrec["Data"]
                assert len(jsondata) == len(trace)
                for i in range(len(jsondata)):
                    assert jsondata[i] == trace[i]

    def test_read_file_via_obspy(self, datapath):
        """
        Read file test via L{obspy.core.Stream}.
        """
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        assert len(stream) == 6
        assert stream[0].stats.network == "CO"
        assert stream[0].stats.station == "BIRD"
        assert stream[0].stats.location == "00"
        assert stream[0].stats.channel == "HHE"
        assert str(stream[0].data) == "[ 401  630  750 ..., 1877 1019 1659]"
        # This is controlled by the stream[0].data attribute.
        assert stream[0].stats.npts == 3000
        assert stream[0].data.dtype == np.int32

    def test_write_int32(self, datapath):
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        outfile = datapath / "bird_jsc_recomp_int32.ms3"
        stream.write(outfile, format="MSEED3", encoding="INT32")
        redo = read(outfile, format="MSEED3")
        self._check_bird_jsc(stream, redo)
        assert stream[0].data.dtype == np.int32
        assert redo[0].data.dtype == np.int32

    def test_write_float32(self, datapath):
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            stream.write(outfile, format="MSEED3", encoding="FLOAT32")
            redo = read(outfile, format="MSEED3")
        self._check_bird_jsc(stream, redo)
        assert redo[0].data.dtype == np.float32

    def test_write_float64(self, datapath):
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            stream.write(outfile, format="MSEED3", encoding="FLOAT64")
            redo = read(outfile, format="MSEED3")
        self._check_bird_jsc(stream, redo)
        assert redo[0].data.dtype == np.float64

    def test_write_steim1(self, datapath):
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            stream.write(outfile, format="MSEED3", encoding="STEIM1")
            redo = read(outfile, format="MSEED3")
        self._check_bird_jsc(stream, redo)
        assert redo[0].data.dtype == np.int32

    def test_write_steim2(self, datapath):
        testfile = datapath / "bird_jsc.ms3"
        stream = read(testfile, format="MSEED3")
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            stream.write(outfile, format="MSEED3", encoding="STEIM2")
            redo = read(outfile, format="MSEED3")
        self._check_bird_jsc(stream, redo)
        assert redo[0].data.dtype == np.int32

    def test_write_long_sid(self, datapath):
        net = "XX2025"
        sta = "BIGGYBIG"
        loc = "01234567"
        band = "L"
        s = "RQQ"
        subs = "Z"
        self._do_check_sid(datapath, net, sta, loc, band, s, subs)

    def test_write_no_loc_subsource(self, datapath):
        net = "XX2025"
        sta = "BIGGYBIG"
        loc = ""
        band = "L"
        s = "RQQ"
        subs = ""
        self._do_check_sid(datapath, net, sta, loc, band, s, subs)

    def _do_check_sid(self, datapath, net, sta, loc, band, s, subs):
        chanSourceId = f"FDSN:{net}_{sta}_{loc}_{band}_{s}_{subs}"
        # create fake
        data = np.fromfunction(
            lambda i: (i % 99 - 49), (100,), dtype=np.int32
        )
        header = MSeed3Header()
        header.starttime = "2024-01-02T15:13:55.123456Z"
        header.sampleRatePeriod = -1 # neg is period, so 1 sps
        identifier = FDSNSourceId.parse(chanSourceId)
        record = MSeed3Record(header, identifier, data)
        recordBytes = record.pack()
        with NamedTemporaryFile() as tf:
            outfile = tf.name
            with open(outfile, 'wb') as out:
                out.write(recordBytes)
            st = read(outfile, format="MSEED3")
            assert len(st) == 1
            tr = st[0]
            assert tr.stats.network == net
            assert tr.stats.station == sta
            assert tr.stats.location == loc
            assert tr.stats.channel == f"{band}_{s}_{subs}"
            st.write(outfile, format="MSEED3")
            redo = read(outfile, format="MSEED3")
            assert len(redo) == 1
            redotr = redo[0]
            assert redotr.stats.network == net
            assert redotr.stats.station == sta
            assert redotr.stats.location == loc
            assert redotr.stats.channel == f"{band}_{s}_{subs}"

    def _check_bird_jsc(self, stream_a, stream_b):
        assert len(stream_a) == 6
        assert len(stream_b) == 6
        for st_idx in range(len(stream_b)):
            assert stream_a[st_idx].stats.network == stream_b[st_idx].stats.network
            assert stream_a[st_idx].stats.station == stream_b[st_idx].stats.station
            assert stream_a[st_idx].stats.location == stream_b[st_idx].stats.location
            assert stream_a[st_idx].stats.channel == stream_b[st_idx].stats.channel
            assert len(stream_a[st_idx].data) == len(stream_b[st_idx].data)
            for i in range(len(stream_b[st_idx].data)):
                assert stream_a[st_idx].data[i] == stream_b[st_idx].data[i]

    def test_fail_steim_for_float(self, datapath):
        data = np.array([1.1, 2, 3], dtype="float32")
        tr = Trace(data)
        stream = Stream([tr])
        with pytest.raises(Exception):
            with NamedTemporaryFile() as tf:
                stream.write(tf.name, format="MSEED3", encoding="STEIM1")
        with pytest.raises(Exception):
            with NamedTemporaryFile() as tf:
                stream.write(tf.name, format="MSEED3", encoding="STEIM2")

    def test_guess_encoding(self, datapath):
        tr = Trace(np.array([1, 2, 3, -17], dtype="int64"))
        stream = Stream([tr])
        # guess output encoding
        with BytesIO() as buf:
            stream.write(buf, format="MSEED3")
            buf.seek(0)
            in_stream = read(buf)
        assert in_stream[0].data.dtype == np.int32
        assert np.array_equal(stream[0].data, in_stream[0].data)
        tr = Trace(np.array([1.1, 2.2, 3.3, -17.1], dtype="float32"))
        stream = Stream([tr])
        # guess output encoding
        with BytesIO() as buf:
            stream.write(buf, format="MSEED3")
            buf.seek(0)
            in_stream = read(buf)
        assert in_stream[0].data.dtype == np.float32
        assert np.array_equal(stream[0].data, in_stream[0].data)
        tr = Trace(np.array([1.1, 2.2, 3.3, -17.1, 2**55], dtype="float64"))
        stream = Stream([tr])
        # guess output encoding
        with BytesIO() as buf:
            stream.write(buf, format="MSEED3")
            buf.seek(0)
            in_stream = read(buf)
        assert in_stream[0].data.dtype == np.float64
        assert np.array_equal(stream[0].data, in_stream[0].data)

    def test_fail_overflow(self, datapath):
        x = 2**55
        data = [1, 2, -3, x, -1]
        # should fail as python int list to numpy array checks values can fit
        # doesn't fail in numpy2.2 as remove
        #with pytest.raises(OverflowError):
            # this doesn't error
            # np.array(data, dtype=np.int32)
            # nor this
            # np.array(data).astype(np.int32)
        data_i64 = np.array(data, dtype=np.int64)

        # numpy ndarray.astype() does not check values fitting for
        # int64 -> int32 conversion
        tr = Trace(data_i64)
        stream = Stream([tr])
        with pytest.raises(ObsPyMSEED3DataOverflowError):
            with NamedTemporaryFile() as tf:
                stream.write(tf.name, format="MSEED3", encoding="INT32")
