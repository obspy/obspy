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

# a delay in the KNet data logger - subtract this from "record time"
KNET_TRIGGER_DELAY = 15.0

TIMEFMT = '%Y/%m/%d %H:%M:%S'

def _is_knet_ascii(filename):
    with open(filename, 'rt') as f:
        first_string = f.read(11)
        # File has less than 7 characters
        if len(first_string) != 11:
            return False
        if first_string == 'Origin Time':
            return True
    return False

def _read_knet_hdr(hdrlines):
    """
    Read the header values into a dictionary.
    @param hdlines: List of the first lines of a KNet ASCII file, not including the "Memo." line.
    @return: Dictionary of values containing most of the elements expected in a Stats object.
    """
    hdrdict = {}
    for line in hdrlines:
        if line.startswith('Station Code'):
            parts = line.split()
            hdrdict['station'] = parts[2]
        if line.startswith('Station Lat'):
            parts = line.split()
            hdrdict['lat'] = float(parts[2])
        if line.startswith('Station Long'):
            parts = line.split()
            hdrdict['lon'] = float(parts[2])
        if line.startswith('Station Height'):
            parts = line.split()
            hdrdict['height'] = float(parts[2])
        if line.startswith('Record Time'):
            parts = line.split()
            datestr = parts[2] + ' ' + parts[3]
            hdrdict['starttime'] = UTCDateTime(datetime.strptime(datestr, TIMEFMT)) - KNET_TRIGGER_DELAY
        if line.startswith('Sampling Freq'):
            parts = line.split()
            freqstr = parts[2]
            m = re.search('[0-9]*', freqstr)
            freq = int(m.group())
            delta = 1.0 / freq
            hdrdict['delta'] = delta
            hdrdict['sampling_rate'] = freq
        if line.startswith('Duration Time'):
            parts = line.split()
            duration = float(parts[2])
            hdrdict['duration'] = duration
        if line.startswith('Dir.'):
            parts = line.split()
            channel = parts[1].replace('-', '')
            kiknetcomps = {'1':'NS1', '2':'EW1', '3':'UD1',
                           '4':'NS2', '5':'EW2', '6':'UD2'}
            if channel.strip() in kiknetcomps.keys():  # kiknet directions are 1-6
                channel = kiknetcomps[channel.strip()]
            hdrdict['channel'] = channel
        if line.startswith('Scale Factor'):
            parts = line.split()
            eqn = parts[2]
            num, denom = eqn.split('/')
            num = float(re.search('[0-9]*', num).group())
            denom = float(denom)
            # convert the calibration from gal to m/s^2
            hdrdict['calib'] = 0.01 * num / denom
        if line.startswith('Max. Acc'):
            parts = line.split()
            acc = float(parts[3])
            hdrdict['accmax'] = acc
        hdrdict['units'] = 'acc'  # this will be in all of the headers I read
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

    # The Stats constructor appears to modify the fields in the input dictionary
    # - let's save a copy
    header = hdrdict.copy()

    data = np.array(data)
    stats = Stats(hdrdict)
    trace = Trace(data, header=stats)
    return Stream([trace])

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
