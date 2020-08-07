"""
INGV DMX bindings to ObsPy core module.
"""
from tempfile import SpooledTemporaryFile

import numpy as np

from obspy import Stream, Trace, UTCDateTime
from obspy.core.compatibility import from_buffer
from obspy.core.util.attribdict import AttribDict


descript_trace_dtypes = np.dtype([("network", "4S"),
                                  ("st_name", "5S"),
                                  ("component", "1S"),
                                  ("insstype", np.int16),
                                  ("begintime", np.double),
                                  ("localtime", np.int16),
                                  ("datatype", "1S"),
                                  ("descriptor", "1S"),
                                  ("digi_by", np.int16),
                                  ("processed", np.int16),
                                  ("length", np.int32),
                                  ("rate", np.float32),
                                  ("mindata", np.float32),
                                  ("maxdata", np.float32),
                                  ("avenoise", np.float32),
                                  ("numclip", np.int32),
                                  ("timecorrect", np.double),
                                  ("rate_correct", np.float32),
                                  ])

structtag_dtypes = np.dtype([("sinc", "1S"),
                             ("machine", "1S"),
                             ("id_struct", np.int16),
                             ("len_struct", np.int32),
                             ("len_data", np.int32)])

types = {"s": np.uint16, "q": np.int16, "u": np.uint16, "i": np.int16,
         "2": np.int32, "l": np.int32, "r": np.uint16, "f": np.float32,
         "d": np.float64}


def readstructtag(fid):
    y = AttribDict()
    # avoid passing np.intXX down to SpooledTemporaryFile.read() since it
    # errors out on numpy integer types on at least Python 3.6, seems fixed in
    # Python 3.7
    # see https://ci.appveyor.com/project/obspy/obspy/
    #                  builds/29252080/job/9gr8bqkgr005523n#L742
    data = fid.read(int(structtag_dtypes.itemsize))
    data = from_buffer(data, structtag_dtypes)
    for (key, (fmt, size)) in structtag_dtypes.fields.items():
        if str(fmt).count("S") != 0:
            y[key] = data[key][0].decode('UTF-8')
        else:
            y[key] = data[key][0]
    return y


def readdescripttrace(fid):
    y = AttribDict()

    # avoid passing np.intXX down to SpooledTemporaryFile.read() since it
    # errors out on numpy integer types on at least Python 3.6, seems fixed in
    # Python 3.7
    # see https://ci.appveyor.com/project/obspy/obspy/
    #                  builds/29252080/job/9gr8bqkgr005523n#L742
    data = fid.read(int(descript_trace_dtypes.itemsize))
    data = from_buffer(data, descript_trace_dtypes)

    for (key, (fmt, size)) in descript_trace_dtypes.fields.items():
        if str(fmt).count("S") != 0:
            y[key] = data[key][0].decode('UTF-8')
        else:
            y[key] = data[key][0]

    return y


def readdata(fid, n, t):
    target = types[t]
    # avoid passing np.intXX down to SpooledTemporaryFile.read() since it
    # errors out on numpy integer types on at least Python 3.6, seems fixed in
    # Python 3.7
    # see https://ci.appveyor.com/project/obspy/obspy/
    #                  builds/29252080/job/9gr8bqkgr005523n#L742
    data = fid.read(int(np.dtype(target).itemsize * n))
    return from_buffer(data, target)


def _is_dmx(filename):
    try:
        with open(filename, "rb") as fid:
            while fid.read(12):  # we require at least 1 full structtag
                fid.seek(-12, 1)
                structtag = readstructtag(fid)
                if int(structtag.len_struct) + int(structtag.len_data) <= 0:
                    return False
                elif structtag.id_struct == 7:
                    descripttrace = readdescripttrace(fid)
                    UTCDateTime(descripttrace.begintime)
                    readdata(fid, descripttrace.length, descripttrace.datatype)
                    return True
                else:
                    fid.seek(
                        int(structtag.len_struct) + int(structtag.len_data), 1)
    except Exception:
        return False

    # If it reaches this point, this means no id_struct=7 has been found
    return False


def _read_dmx(filename, **kwargs):
    station = kwargs.get("station", None)

    traces = []
    with open(filename, "rb") as fid:
        content = fid.read()

    with SpooledTemporaryFile(mode='w+b') as fid:
        fid.write(content)
        fid.seek(0)

        while fid.read(12):  # we require at least 1 full structtag
            fid.seek(-12, 1)
            structtag = readstructtag(fid)
            if structtag.id_struct == 7:
                descripttrace = readdescripttrace(fid)
                if station is None or descripttrace.st_name.strip() == station:
                    data = readdata(fid, descripttrace.length,
                                    descripttrace.datatype)
                    tr = Trace(data=np.asarray(data))
                    tr.stats.network = descripttrace.network.strip()
                    tr.stats.station = descripttrace.st_name.strip()
                    tr.stats.channel = descripttrace.component
                    tr.stats.sampling_rate = descripttrace.rate
                    tr.stats.starttime = UTCDateTime(descripttrace.begintime)
                    tr.stats.dmx = AttribDict({"descripttrace": descripttrace,
                                               "structtag": structtag})
                    traces.append(tr)
                else:
                    fid.seek(int(structtag.len_data), 1)
            else:
                fid.seek(
                    int(structtag.len_struct) + int(structtag.len_data), 1)

    st = Stream(traces=traces)
    return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
