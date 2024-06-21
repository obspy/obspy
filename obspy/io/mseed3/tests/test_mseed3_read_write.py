import json

from simplemseed import FDSNSourceId

from obspy import read


class TestMSEED3ReadingAndWriting:
    """
    Test everything related to the general reading and writing of MiniSEED
    files.
    """

    def test_ref_data(self, datapath):
        testdir = datapath / "reference-data"
        for path in testdir.glob("*.mseed3"):
            if path.name == "reference-text.mseed3":
                # 0 not decompressible, can't return empty stream so???
                continue
            print(path)
            jsonpath = path.parent.joinpath(path.name[:-6] + "json")
            stream = read(path, format="MSEED3")
            assert len(stream) == 1
            trace = stream[0]
            with open(jsonpath, "r") as injson:
                jsonrec = json.load(injson)[0]

            assert (jsonrec["SampleRate"] == trace.stats.sampling_rate)
            assert (
                jsonrec["PublicationVersion"] ==
                trace.stats.mseed3.pubVer
            )
            sid = FDSNSourceId.fromNslc(
                trace.stats.network,
                trace.stats.station,
                trace.stats.location,
                trace.stats.channel,
            )
            assert jsonrec["SID"] == str(sid)
            if "ExtraHeaders" in jsonrec:
                assert "eh" in trace.stats.mseed3
                assert jsonrec["ExtraHeaders"] == trace.stats.mseed3.eh
            if jsonrec["DataLength"] > 0:
                assert jsonrec["SampleCount"] == len(trace)
                jsondata = jsonrec["Data"]
                assert len(jsondata) == len(trace)
                for i in range(len(jsondata)):
                    assert jsondata[i] == trace[i]

    """
    Test everything related to the general reading and writing of MiniSEED
    files.
    """

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
