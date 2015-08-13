# -*- coding: utf-8 -*-
"""
Reading of the K-NET and KiK-net ascii format as defined on
http://www.kyoshin.bosai.go.jp.
"""

from obspy import UTCDateTime, Stream, Trace
from obspy.core.trace import Stats
# from obspy.core.trace import Trace
# from obspy.core.trace import Stats
from datetime import datetime
import re
import numpy as np


class KnetFormatError(Exception):
    pass

class KnetDataError(Exception):
    pass

def _is_knet_ascii(filename):
    with open(filename, 'rt') as f:
        first_string = f.read(11)
        # File has less than 7 characters
        if len(first_string) != 11:
            return False
        if first_string == 'Origin Time':
            return True
    return False

def _prep_hdr_line(name, line):
    if not line.startswith(name):
        raise KnetFormatError("Expected line to start with %s but got %s " \
                              % (name, line))
    else:
        return line.split()

def _read_knet_hdr(hdrlines):
    """
    Read the header values into a dictionary.
    @param hdlines: List of the first lines of a KNet ASCII file, not including the "Memo." line.
    @return: Dictionary of values containing most of the elements expected in a Stats object.
    """
    hdrdict = {'knet':{}}
    hdrnames = ['Origin Time', 'Lat.', 'Long.', 'Depth. (km)', 'Mag.',
                'Station Code', 'Station Lat.', 'Station Long.',
                'Station Height(m)', 'Record Time', 'Sampling Freq(Hz)',
                'Duration Time(s)', 'Dir.', 'Scale Factor', 'Max. Acc. (gal)',
                'Last Correction', 'Memo.']
    _i = 0
    # Event information
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S'))
    hdrdict['knet']['evot'] = dt

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    lat = float(flds[1])
    hdrdict['knet']['evla'] = lat

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    lon = float(flds[1])
    hdrdict['knet']['evlo'] = lon

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dp = float(flds[2])
    hdrdict['knet']['evdp'] = dp

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    mag = float(flds[1])
    hdrdict['knet']['mag'] = mag

    # Station information
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['station'] = flds[2]

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stla'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stlo'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['stel'] = float(flds[2])

    # Data information
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    # A 15 s delay is added to the record time by the
    # the K-NET and KiK-Net data logger"
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')) - 15.0
    hdrdict['starttime'] = dt

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    freqstr = flds[2]
    m = re.search('[0-9]*', freqstr)
    freq = int(m.group())
    delta = 1.0 / freq
    hdrdict['delta'] = delta
    hdrdict['sampling_rate'] = freq

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    hdrdict['knet']['duration'] = float(flds[2])

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    channel = flds[1].replace('-', '')
    kiknetcomps = {'1':'NS1', '2':'EW1', '3':'UD1',
                   '4':'NS2', '5':'EW2', '6':'UD2'}
    if channel.strip() in kiknetcomps.keys():  # kiknet directions are 1-6
        channel = kiknetcomps[channel.strip()]
    hdrdict['channel'] = channel

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    eqn = flds[2]
    num, denom = eqn.split('/')
    num = float(re.search('[0-9]*', num).group())
    denom = float(denom)
    # convert the calibration from gal to m/s^2
    hdrdict['calib'] = 0.01 * num / denom

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    acc = float(flds[3])
    hdrdict['knet']['accmax'] = acc

    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    dt = flds[2] + ' ' + flds[3]
    dt = UTCDateTime(datetime.strptime(dt, '%Y/%m/%d %H:%M:%S'))
    hdrdict['knet']['last correction'] = dt

    # The comment ('Memo') field is optional
    _i += 1
    flds = _prep_hdr_line(hdrnames[_i], hdrlines[_i])
    if len(flds) > 1:
        hdrdict['knet']['comment'] = ' '.join(flds[1:])

    if len(hdrlines) != _i + 1:
        raise KnetFormatError("Expected %d header lines but got %d" \
                              % (_i + 1, len(hdrlines)))
    return hdrdict

def _read_knet_ascii(filename, **kwargs):
    """
    Read a KNet ASCII file, and return an ObsPy Trace object, plus a dictionary of header values.
    @param knetfilename: String path to valid KNet ASCII file, as described here: http://www.kyoshin.bosai.go.jp/kyoshin/man/knetform_en.html
    @return: ObsPy Trace object, and a dictionary of some of the header values found in the input file.
    """
    data = []
    hdrdict = {}
    with open(filename, 'rt') as f:
        dataOn = False
        headerlines = []
        for line in f.readlines():
            if line.startswith('Memo'):
                headerlines.append(line)
                hdrdict = _read_knet_hdr(headerlines)
                dataOn = True
                continue
            if not dataOn:
                headerlines.append(line)
                continue
            if dataOn:
                parts = line.strip().split()
                mdata = [float(p) for p in parts]
                data = data + mdata

    # fill in the values usually expected in Stats as best we can
    hdrdict['npts'] = len(data)

    elapsed = float(hdrdict['npts']) / float(hdrdict['sampling_rate'])
    hdrdict['endtime'] = hdrdict['starttime'] + elapsed
    hdrdict['network'] = 'NIED'
    hdrdict['location'] = ''

    data = np.array(data)
    stats = Stats(hdrdict)
    trace = Trace(data, header=stats)
    return Stream([trace])

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
