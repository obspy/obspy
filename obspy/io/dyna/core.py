# -*- coding: utf-8 -*-
"""
DYNA and ITACA bindings to ObsPy core module.

:copyright:
    The ITACA Development Team (itaca@mi.ingv.it)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, unicode_literals)
from future.builtins import *  # NOQA

from io import StringIO
import re

import numpy as np

from obspy.core import Stream, Trace, Stats, AttribDict, UTCDateTime


class NativeHeader:
    """
    Base class for handling of native headers
    """
    pass


class DynaHdr(NativeHeader):
    """
    Class to handle DYNA header
    """
    KEYS_DYNA_10 = [
        # event related
        "EVENT_NAME",
        "EVENT_ID",
        "EVENT_DATE_YYYYMMDD",
        "EVENT_TIME_HHMMSS",
        "EVENT_LATITUDE_DEGREE",
        "EVENT_LONGITUDE_DEGREE",
        "EVENT_DEPTH_KM",
        "HYPOCENTER_REFERENCE",
        "MAGNITUDE_W",
        "MAGNITUDE_W_REFERENCE",
        "MAGNITUDE_L",
        "MAGNITUDE_L_REFERENCE",
        "FOCAL_MECHANISM",
        # station related
        "NETWORK",
        "STATION_CODE",
        "STATION_NAME",
        "STATION_LATITUDE_DEGREE",
        "STATION_LONGITUDE_DEGREE",
        "STATION_ELEVATION_M",
        "LOCATION",
        "VS30_M/S",
        "SITE_CLASSIFICATION_EC8",
        "MORPHOLOGIC_CLASSIFICATION",
        # geometry
        "EPICENTRAL_DISTANCE_KM",
        "EARTHQUAKE_BACKAZIMUTH_DEGREE",
        # seismic record
        "DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS",
        "DATE_TIME_FIRST_SAMPLE_PRECISION",
        "SAMPLING_INTERVAL_S",
        "NDATA",
        "DURATION_S",
        "STREAM",
        "UNITS",
        # instrument
        "INSTRUMENT",
        "INSTRUMENT_ANALOG/DIGITAL",
        "INSTRUMENTAL_FREQUENCY_HZ",
        "INSTRUMENTAL_DAMPING",
        "FULL_SCALE_G",
        "N_BIT_DIGITAL_CONVERTER",
        # measures, conditional
        ("peak"),
        ("peak_time"),
        # misc
        "BASELINE_CORRECTION",
        "FILTER_TYPE",
        "FILTER_ORDER",
        "LOW_CUT_FREQUENCY_HZ",
        "HIGH_CUT_FREQUENCY_HZ",
        "LATE/NORMAL_TRIGGERED",
        "DATABASE_VERSION",
        "HEADER_FORMAT",
        "DATA_TYPE",
        "PROCESSING",
        "DATA_TIMESTAMP_YYYYMMDD_HHMMSS",
        "USER1",
        "USER2",
        "USER3",
        "USER4",
        ]

    # data type: acceleration
    ACC = "ACCELERATION"
    SA = "ACCELERATION RESPONSE SPECTRUM"
    U1 = "PGA_CM/S^2"
    U2 = "TIME_PGA_S"
    # data type: velocity
    VEL = "VELOCITY"
    PSV = "PSEUDO-VELOCITY RESPONSE SPECTRUM"
    U1 = "PGV_CM/S"
    U2 = "TIME_PGV_S"
    # data type: displacement
    DIS = "DISPLACEMENT"
    SD = "DISPLACEMENT RESPONSE SPECTRUM"
    U1 = "PGD_CM"
    U2 = "TIME_PGD_S"

    def __init__(self):
        self.is_acc = False
        self.is_vel = False
        self.is_disp = False

    @staticmethod
    def _stats_key(key):
        # TODO: to be decided!
        # return key.translate(str.maketrans('/^', '__')).lower()
        return key.translate(str.maketrans('/^', '__')).upper()

    @staticmethod
    def get_template():
        stats_dyna = AttribDict()
        for key in DynaHdr.KEYS_DYNA_10:
            if key == '*':
                continue
            skey = DynaHdr._stats_key(key)
            stats_dyna[skey] = None

        # format identifier
        stats_dyna.header_format = 'DYNA 1.0'

        return stats_dyna

    @staticmethod
    def get_default(stats):
        """set default values"""

        # from generic headers
        stats.dyna['NETWORK'] = stats['network']
        stats.dyna['STATION_CODE'] = stats['station']
        stats.dyna['LOCATION'] = stats['location']
        stats.dyna['STREAM'] = stats['channel']

        stats.dyna['NDATA'] = stats['npts']
        stats.dyna['SAMPLING_INTERVAL_S'] = stats['delta']

        # conditional: seconds or milliseconds
        # = stats.starttime
        # XX: will need format conversion
        stats.dyna.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS = stats.starttime

        # Warning?
        if stats['calib'] != 1.0:
            pass

        # data type qis acceleration, acc, SA
        # DATA_TYPE == "ACCELERATION"
        # DATA_TYPE == "ACCELERATION RESPONSE SPECTRUM"
        if is_acceleration:
            # key_peak = "PGA_CM/S^2", "PGA_CM_S_2",
            # key_peak_time = "TIME_PGA_S"
            pass

        # data type is velocity, vel, PSV
        # DATA_TYPE == "VELOCITY"
        # DATA_TYPE == "PSEUDO-VELOCITY RESPONSE SPECTRUM"
        if is_velocity:
            # key_peak = "PGV_CM/S", "PGV_CM_S",
            # key_peak_time = "TIME_PGV_S"
            pass

        # data type is displacement, dis, SD
        # DATA_TYPE == "DISPLACEMENT", not written?
        # DATA_TYPE == "DISPLACEMENT RESPONSE SPECTRUM"
        if is_displacement:
            # key_peak = "PGD_CM"
            # key_peak_time = "TIME_PGD_S"
            pass

        return


def _is_dyna(filename):
    """
    Checks whether a file is a DYNA 1.0 ASCII file or not.

    :type filename: str
    :param filename: Name of the DYNA file to be checked.
    :rtype: bool
    :return: ``True`` if a DYNA 1.0 ASCII file.

    .. rubric:: Example

    >>> _is_dyna("/path/to/IT.ARL..HGE.D.20140120.071240.X.ACC.ASC")  #doctest: +SKIP  # NOQA
    True
    """
    # 1: Initial characters contain "EVENT_NAME:"
    # 2: 48th line contains format header field 
    PROBE = ((1, "EVENT_NAME:"),
             (48, "HEADER_FORMAT: DYNA 1.0")
             )

    test = False
    with open(filename, 'rt') as f:
        p = PROBE[0][1]
        test = (f.read(len(p)) == p)

        for i in range(55):
            line = f.readline()
            if i+1 == PROBE[1][0]:
                p = PROBE[1][1]
                test = test and (line[:len(p)] == p)

    return test


def _is_itaca(filename):
    """
    Checks whether a file is a ITACA ASCII file or not.

    :type filename: str
    :param filename: Name of the ITACA file to be checked.
    :rtype: bool
    :return: ``True`` if a ITACA ASCII file.

    .. rubric:: Example

    >>> _is_itaca("/path/to/19971014_152309ITDPC_NCR__WEX.DAT")  #doctest: +SKIP
    True
    """
    # FIXME:
    # If used with file-like object, we cannot check for filename anymore

    # filename must match the following regexp:
    # ^\d{8}_\d{6}\w{5}_\w{5}\w{3}.\w{3}$
    REGEX_FNAME = '.*\d{8}_\d{6}\w{5}_\w{5}\w{3}.\w{3}$'

    if not re.match(REGEX_FNAME, filename):
        return False

    # 1: Initial characters contain "EVENT_NAME:"
    # 2: 8th line begins with "MAGNITUDE_S:"
    PROBE = (( 1, "EVENT_NAME:" ),
             ( 8, "MAGNITUDE_S:" )
             )

    test = False
    with open(filename, 'rt') as f:
        p = PROBE[0][1]
        test = (f.read(len(p)) == p)

        for i in range(8):
            line = f.readline()
            if i+1 == PROBE[1][0]:
                p = PROBE[1][1]
                test = test and (line[:len(p)] == p)

    return test


def _read_dyna(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a DYNA 1.0 ASCII file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: ASCII file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read("/path/to/IT.ARL..HGE.D.20140120.071240.X.ACC.ASC")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    IT.ARL..E | 2014-01-20T07:12:30.000000Z - 2014-01-20T07:13:14.980000Z | 200.0 Hz, 8997 samples  # NOQA
    """
    # read file
    with open(filename, 'rt') as f:
        # read header
        headers = {}
        for i in range(55):
            key, value = f.readline().strip().split(':', 1)
            headers[key.strip()] = value.strip()

        # read data
        if not headonly:
            data = np.loadtxt(f, dtype=np.float32)

    stats = Stats()
    dyna = AttribDict()

    # TEMP!
    header = stats
    header['dyna'] = dyna

    # generic stats
    header['network'] = headers['NETWORK']
    header['station'] = headers['STATION_CODE']
    header['location'] = headers['LOCATION']
    header['channel'] = headers['STREAM']

    try:
        header['starttime'] = UTCDateTime(
            headers['DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'])
    except:
        header['starttime'] = UTCDateTime(0)

    header['delta'] = float(headers['SAMPLING_INTERVAL_S'])
    header['npts'] = int(headers['NDATA'])
    header['calib'] = 1.0  # not in file header

    # DYNA dict float data
    header['dyna']['EVENT_LATITUDE_DEGREE'] = strtofloat(
        headers['EVENT_LATITUDE_DEGREE'])
    header['dyna']['EVENT_LONGITUDE_DEGREE'] = strtofloat(
        headers['EVENT_LONGITUDE_DEGREE'])
    header['dyna']['EVENT_DEPTH_KM'] = strtofloat(headers['EVENT_DEPTH_KM'])
    header['dyna']['HYPOCENTER_REFERENCE'] = headers['HYPOCENTER_REFERENCE']
    header['dyna']['MAGNITUDE_W'] = strtofloat(headers['MAGNITUDE_W'])
    header['dyna']['MAGNITUDE_L'] = strtofloat(headers['MAGNITUDE_L'])
    header['dyna']['STATION_LATITUDE_DEGREE'] = strtofloat(
        headers['STATION_LATITUDE_DEGREE'])
    header['dyna']['STATION_LONGITUDE_DEGREE'] = strtofloat(
        headers['STATION_LONGITUDE_DEGREE'])
    header['dyna']['VS30_M_S'] = strtofloat(headers['VS30_M/S'])
    header['dyna']['EPICENTRAL_DISTANCE_KM'] = strtofloat(
        headers['EPICENTRAL_DISTANCE_KM'])
    header['dyna']['EARTHQUAKE_BACKAZIMUTH_DEGREE'] = strtofloat(
        headers['EARTHQUAKE_BACKAZIMUTH_DEGREE'])
    header['dyna']['DURATION_S'] = strtofloat(headers['DURATION_S'])
    header['dyna']['INSTRUMENTAL_FREQUENCY_HZ'] = strtofloat(
        headers['INSTRUMENTAL_FREQUENCY_HZ'])
    header['dyna']['INSTRUMENTAL_DAMPING'] = strtofloat(
        headers['INSTRUMENTAL_DAMPING'])
    header['dyna']['FULL_SCALE_G'] = strtofloat(headers['FULL_SCALE_G'])

    # data type is acceleration
    if headers['DATA_TYPE'] == "ACCELERATION" \
            or headers['DATA_TYPE'] == "ACCELERATION RESPONSE SPECTRUM":
        header['dyna']['PGA_CM_S_2'] = strtofloat(headers['PGA_CM/S^2'])
        header['dyna']['TIME_PGA_S'] = strtofloat(headers['TIME_PGA_S'])
    # data type is velocity
    if headers['DATA_TYPE'] == "VELOCITY" \
            or headers['DATA_TYPE'] == "PSEUDO-VELOCITY RESPONSE SPECTRUM":
        header['dyna']['PGV_CM_S'] = strtofloat(headers['PGV_CM/S'])
        header['dyna']['TIME_PGV_S'] = strtofloat(headers['TIME_PGV_S'])
    # data type is displacement
    if headers['DATA_TYPE'] == "DISPLACEMENT" \
            or headers['DATA_TYPE'] == "DISPLACEMENT RESPONSE SPECTRUM":
        header['dyna']['PGD_CM'] = strtofloat(headers['PGD_CM'])
        header['dyna']['TIME_PGD_S'] = strtofloat(headers['TIME_PGD_S'])

    header['dyna']['LOW_CUT_FREQUENCY_HZ'] = strtofloat(
        headers['LOW_CUT_FREQUENCY_HZ'])
    header['dyna']['HIGH_CUT_FREQUENCY_HZ'] = strtofloat(
        headers['HIGH_CUT_FREQUENCY_HZ'])

    # DYNA dict int data
    header['dyna']['STATION_ELEVATION_M'] = strtoint(
        headers['STATION_ELEVATION_M'])
    header['dyna']['N_BIT_DIGITAL_CONVERTER'] = strtoint(
        headers['N_BIT_DIGITAL_CONVERTER'])
    header['dyna']['FILTER_ORDER'] = strtoint(headers['FILTER_ORDER'])

    # DYNA dict string data
    header['dyna']['EVENT_NAME'] = headers['EVENT_NAME']
    header['dyna']['EVENT_ID'] = headers['EVENT_ID']
    header['dyna']['EVENT_DATE_YYYYMMDD'] = headers['EVENT_DATE_YYYYMMDD']
    header['dyna']['EVENT_TIME_HHMMSS'] = headers['EVENT_TIME_HHMMSS']
    header['dyna']['MAGNITUDE_W_REFERENCE'] = headers['MAGNITUDE_W_REFERENCE']
    header['dyna']['MAGNITUDE_L_REFERENCE'] = headers['MAGNITUDE_L_REFERENCE']
    header['dyna']['FOCAL_MECHANISM'] = headers['FOCAL_MECHANISM']
    header['dyna']['STATION_NAME'] = headers['STATION_NAME']
    header['dyna']['SITE_CLASSIFICATION_EC8'] = headers[
        'SITE_CLASSIFICATION_EC8']
    header['dyna']['MORPHOLOGIC_CLASSIFICATION'] = headers[
        'MORPHOLOGIC_CLASSIFICATION']
    header['dyna']['DATE_TIME_FIRST_SAMPLE_PRECISION'] = headers[
        'DATE_TIME_FIRST_SAMPLE_PRECISION']
    header['dyna']['UNITS'] = headers['UNITS']
    header['dyna']['INSTRUMENT'] = headers['INSTRUMENT']
    header['dyna']['INSTRUMENT_ANALOG_DIGITAL'] = headers[
        'INSTRUMENT_ANALOG/DIGITAL']
    header['dyna']['BASELINE_CORRECTION'] = headers['BASELINE_CORRECTION']
    header['dyna']['FILTER_TYPE'] = headers['FILTER_TYPE']
    header['dyna']['LATE_NORMAL_TRIGGERED'] = headers['LATE/NORMAL_TRIGGERED']
    header['dyna']['HEADER_FORMAT'] = headers['HEADER_FORMAT']
    header['dyna']['DATABASE_VERSION'] = headers['DATABASE_VERSION']
    header['dyna']['DATA_TYPE'] = headers['DATA_TYPE']
    header['dyna']['PROCESSING'] = headers['PROCESSING']
    header['dyna']['DATA_TIMESTAMP_YYYYMMDD_HHMMSS'] = headers[
        'DATA_TIMESTAMP_YYYYMMDD_HHMMSS']
    header['dyna']['USER1'] = headers['USER1']
    header['dyna']['USER2'] = headers['USER2']
    header['dyna']['USER3'] = headers['USER3']
    header['dyna']['USER4'] = headers['USER4']


    # create ObsPy stream object
    stream = Stream()
    if headonly:
        # skip data
        stream.append(Trace(header=header))
    elif not headers['DATA_TYPE'][-8:] == "SPECTRUM":
        # regular trace
        stream.append(Trace(data=data, header=header))

    else:
        # FIXME: headers should be different !!!
        stream.append(Trace(data=data[:,0], header=header))
        stream.append(Trace(data=data[:,1], header=header))

    return stream


def _read_itaca(filename, headonly=False, **kwargs):  # @UnusedVariable
    """
    Reads a ITACA ASCII file and returns an ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: str
    :param filename: ASCII file to be read.
    :type headonly: bool, optional
    :param headonly: If set to True, read only the head. This is most useful
        for scanning available data in huge (temporary) data sets.
    :rtype: :class:`~obspy.core.stream.Stream`
    :return: A ObsPy Stream object.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read("/path/to/19971014_152309ITDPC_NCR__WEX.DAT")
    >>> st  # doctest: +ELLIPSIS
    <obspy.core.stream.Stream object at 0x...>
    >>> print(st)  # doctest: +ELLIPSIS
    1 Trace(s) in Stream:
    IT.NCR..HNE | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:00:32.795000Z | 200.0 Hz, 6560 samples  # NOQA
    """
    # read file
    with open(filename, 'rt') as f:
        # read header
        headers = {}
        for i in range(43):
            key, value = fh.readline().strip().split(':')
            headers[key.strip()] = value.strip()

        # read data
        if not headonly:
            data = np.loadtxt(fh, dtype=np.float32)

    # construct ObsPy stats
    stats = Stats()
    itaca = AttribDict()

    # TEMP!
    header = stats
    header['itaca'] = itaca

    # FIXME: handle situation when filename is not available!
    header['network'] = filename[-18:-16]
    header['station'] = headers['STATION_CODE']
    header['location'] = ''

    # FIXME: try to determine full and correct channel code
    if headers['COMPONENT'] == 'WE':
        header['channel'] = 'HNE'
    elif headers['COMPONENT'] == 'EW':
        # EW should *NEVER* appear, but we handle it anyway
        header['channel'] = 'HNE'
        #  -just in case ;)

    if headers['COMPONENT'] == 'NS':
        header['channel'] = 'HNN'

    if headers['COMPONENT'] == 'UP':
        header['channel'] = 'HNZ'

    # FIXME: This does not really make sense!
    # Anyway toUTCDateTime can be avoided
    try:
        tfs = headers['EVENT_DATE_YYYYMMDD'] + \
            '_' + headers['TIME_FIRST_SAMPLE_S']

        header['starttime'] = toUTCDateTime(tfs)

        if re.match('^00', headers['TIME_FIRST_SAMPLE_S']) \
               and re.match('^23', headers['EVENT_TIME_HHMMSS']):  # NOQA
            header['starttime'] = header['starttime'] + 86400
        if re.match('^23', headers['TIME_FIRST_SAMPLE_S']) \
               and re.match('^00', headers['EVENT_TIME_HHMMSS']):  # NOQA
            header['starttime'] = header['starttime'] - 86400
    except:
        header['starttime'] = UTCDateTime(0.0)

    header['delta'] = float(headers['SAMPLING_INTERVAL_S'])
    header['npts'] = int(headers['NDATA'])
    header['calib'] = 1.0  # not in file header


    # ITACA dict float data
    header['itaca']['EVENT_LATITUDE_DEGREE'] = strtofloat(
        headers['EVENT_LATITUDE_DEGREE'])
    header['itaca']['EVENT_LONGITUDE_DEGREE'] = strtofloat(
        headers['EVENT_LONGITUDE_DEGREE'])
    header['itaca']['EVENT_DEPTH_KM'] = strtofloat(headers['EVENT_DEPTH_KM'])
    header['itaca']['MAGNITUDE_L'] = strtofloat(headers['MAGNITUDE_L'])
    header['itaca']['MAGNITUDE_S'] = strtofloat(headers['MAGNITUDE_S'])
    header['itaca']['MAGNITUDE_W'] = strtofloat(headers['MAGNITUDE_W'])
    header['itaca']['STATION_LATITUDE_DEGREE'] = strtofloat(
        headers['STATION_LATITUDE_DEGREE'])
    header['itaca']['STATION_LONGITUDE_DEGREE'] = strtofloat(
        headers['STATION_LONGITUDE_DEGREE'])
    header['itaca']['EPICENTRAL_DISTANCE_KM'] = strtofloat(
        headers['EPICENTRAL_DISTANCE_KM'])
    header['itaca']['EARTHQUAKE_BACKAZIMUTH_DEGREE'] = strtofloat(
        headers['EARTHQUAKE_BACKAZIMUTH_DEGREE'])
    header['itaca']['DURATION_S'] = strtofloat(headers['DURATION_S'])
    header['itaca']['INSTRUMENTAL_FREQUENCY_HZ'] = strtofloat(
        headers['INSTRUMENTAL_FREQUENCY_HZ'])
    header['itaca']['INSTRUMENTAL_DAMPING'] = strtofloat(
        headers['INSTRUMENTAL_DAMPING'])
    header['itaca']['FULL_SCALE_G'] = strtofloat(headers['FULL_SCALE_G'])

    # data type is acceleration
    if headers['DATA_TYPE'] == "UNPROCESSED ACCELERATION" \
            or headers['DATA_TYPE'] == "PROCESSED ACCELERATION" \
            or headers['DATA_TYPE'][-8:] == "SPECTRUM":
        header['itaca']['PGA_CM_S_2'] = strtofloat(headers['PGA_CM/S^2'])
        header['itaca']['TIME_PGA_S'] = strtofloat(headers['TIME_PGA_S'])
    # data type is velocity
    if headers['DATA_TYPE'] == "VELOCITY":
        header['itaca']['PGV_CM_S'] = strtofloat(headers['PGV_CM/S'])
        header['itaca']['TIME_PGV_S'] = strtofloat(headers['TIME_PGV_S'])
    # data type is displacement
    if headers['DATA_TYPE'] == "DISPLACEMENT":
        header['itaca']['PGD_CM'] = strtofloat(headers['PGD_CM'])
        header['itaca']['TIME_PGD_S'] = strtofloat(headers['TIME_PGD_S'])

    header['itaca']['LOW_CUT_FREQUENCY_HZ'] = strtofloat(
        headers['LOW_CUT_FREQUENCY_HZ'])
    header['itaca']['HIGH_CUT_FREQUENCY_HZ'] = strtofloat(
        headers['HIGH_CUT_FREQUENCY_HZ'])

    # ITACA dict int data
    header['itaca']['STATION_ELEVATION_M'] = strtoint(
        headers['STATION_ELEVATION_M'])
    header['itaca']['N_BIT_DIGITAL_CONVERTER'] = strtoint(
        headers['N_BIT_DIGITAL_CONVERTER'])
    header['itaca']['FILTER_ORDER'] = strtoint(headers['FILTER_ORDER'])

    # ITACA dict string data
    header['itaca']['EVENT_NAME'] = headers['EVENT_NAME']
    header['itaca']['EVENT_DATE_YYYYMMDD'] = headers['EVENT_DATE_YYYYMMDD']
    header['itaca']['EVENT_TIME_HHMMSS'] = headers['EVENT_TIME_HHMMSS']
    header['itaca']['FOCAL_MECHANISM'] = headers['FOCAL_MECHANISM']
    header['itaca']['STATION_NAME'] = headers['STATION_NAME']
    header['itaca']['SITE_CLASSIFICATION_EC8'] = headers[
        'SITE_CLASSIFICATION_EC8']
    header['itaca']['MORPHOLOGIC_CLASSIFICATION'] = headers[
        'MORPHOLOGIC_CLASSIFICATION']
    header['itaca']['COMPONENT'] = headers['COMPONENT']
    header['itaca']['UNITS'] = headers['UNITS']
    header['itaca']['INSTRUMENT'] = headers['INSTRUMENT']
    header['itaca']['INSTRUMENT_ANALOG_DIGITAL'] = headers[
        'INSTRUMENT_ANALOG/DIGITAL']
    header['itaca']['BASELINE_CORRECTION'] = headers['BASELINE_CORRECTION']
    header['itaca']['FILTER_TYPE'] = headers['FILTER_TYPE']
    header['itaca']['LATE_NORMAL_TRIGGERED'] = headers['LATE/NORMAL_TRIGGERED']
    header['itaca']['DATA_VERSION'] = headers['DATA_VERSION']
    header['itaca']['DATA_TYPE'] = headers['DATA_TYPE']


    # create ObsPy stream object
    stream = Stream()

    if headonly:
        # skip data
        stream.append(Trace(header=header))
    elif not headers['DATA_TYPE'][-8:] == "SPECTRUM":
        # regular trace
        stream.append(Trace(data=data, header=header))
    else:
        # FIXME: does it make sense, use same header, data representation
        stream.append(Trace(data=data[:,0], header=header))
        stream.append(Trace(data=data[:,1], header=header))

    return stream


def _write_dyna(stream, filename, **kwargs):  # @UnusedVariable
    """
    Writes a DYNA 1.0 ASCII file from given ObsPy Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an
        ObsPy :class:`~obspy.core.stream.Stream` object, call this instead.

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: str
    :param filename: Name of the ASCII file to write.
    """

    # TEMP: needs to deal with multiple traces as well, see SAC!
    tr = stream[0]

    # TEMP: add stats.dyna, if missing
    if not hasattr(tr.stats, 'dyna'):
        tr.stats.dyna = DynaHdr.get_template()
        tr.stats.dyna.DATE_TIME_FIRST_SAMPLE_PRECISION = 'milliseconds'
        if True:
            tr.stats.dyna.DATA_TYPE = 'ACCELERATION'
            tr.stats.dyna.PGA_CM_S_2 = None
            tr.stats.dyna.TIME_PGA_S = None

    stats = tr.stats
    dyna  = tr.stats.dyna

    # FIXME:
    # handle file open/close better!
    # Enable file-likes
    fh = open(filename, 'wt')

    fh.write("EVENT_NAME: %s\n" % dyna.EVENT_NAME)
    fh.write("EVENT_ID: %s\n" % dyna.EVENT_ID)

    fh.write("EVENT_DATE_YYYYMMDD: %s\n" %
             dyna.EVENT_DATE_YYYYMMDD)
    fh.write("EVENT_TIME_HHMMSS: %s\n" %
             dyna.EVENT_TIME_HHMMSS)

    fh.write("EVENT_LATITUDE_DEGREE: %s\n" %
             floattostr(dyna.EVENT_LATITUDE_DEGREE, 4))
    fh.write("EVENT_LONGITUDE_DEGREE: %s\n" %
             floattostr(dyna.EVENT_LONGITUDE_DEGREE, 4))
    fh.write("EVENT_DEPTH_KM: %s\n" %
             floattostr(dyna.EVENT_DEPTH_KM, 1))
    fh.write("HYPOCENTER_REFERENCE: %s\n" %
             dyna.HYPOCENTER_REFERENCE)

    fh.write("MAGNITUDE_W: %s\n" %
             floattostr(dyna.MAGNITUDE_W, 1))
    fh.write("MAGNITUDE_W_REFERENCE: %s\n" %
             dyna.MAGNITUDE_W_REFERENCE)
    fh.write("MAGNITUDE_L: %s\n" %
             floattostr(dyna.MAGNITUDE_L, 1))
    fh.write("MAGNITUDE_L_REFERENCE: %s\n" %
             dyna.MAGNITUDE_L_REFERENCE)
    fh.write("FOCAL_MECHANISM: %s\n" %
             dyna.FOCAL_MECHANISM)

    fh.write("NETWORK: %s\n" % stats.network)
    fh.write("STATION_CODE: %s\n" % stats.station)
    fh.write("STATION_NAME: %s\n" % stats.dyna.STATION_NAME)

    fh.write("STATION_LATITUDE_DEGREE: %s\n" %
             floattostr(dyna.STATION_LATITUDE_DEGREE, 6))
    fh.write("STATION_LONGITUDE_DEGREE: %s\n" %
             floattostr(dyna.STATION_LONGITUDE_DEGREE, 6))
    fh.write("STATION_ELEVATION_M: %s\n" %
             floattostr(dyna.STATION_ELEVATION_M, 0))

    fh.write("LOCATION: %s\n" % stats.location)
    fh.write("VS30_M/S: %s\n" % floattostr(stats.dyna.VS30_M_S, 0))
    fh.write("SITE_CLASSIFICATION_EC8: %s\n" %
             dyna.SITE_CLASSIFICATION_EC8)
    fh.write("MORPHOLOGIC_CLASSIFICATION: %s\n" %
             dyna.MORPHOLOGIC_CLASSIFICATION)
    fh.write("EPICENTRAL_DISTANCE_KM: %s\n" %
             floattostr(dyna.EPICENTRAL_DISTANCE_KM, 1))
    fh.write("EARTHQUAKE_BACKAZIMUTH_DEGREE: %s\n" %
             floattostr(dyna.EARTHQUAKE_BACKAZIMUTH_DEGREE, 1))

    # XX: conditional, derived from stats
    # NOTE: this has changed !!!
    if dyna.DATE_TIME_FIRST_SAMPLE_PRECISION == 'milliseconds':
        stime = stats.starttime.strftime("%Y%m%d_%H%M%S.%f")[:-3]
    elif dyna.DATE_TIME_FIRST_SAMPLE_PRECISION == 'seconds':
        stime = stats.starttime.strftime("%Y%m%d_%H%M%S")
    else:
        stime = ""
    fh.write("DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS: %s\n" % stime)

    fh.write("DATE_TIME_FIRST_SAMPLE_PRECISION: %s\n" %
             dyna.DATE_TIME_FIRST_SAMPLE_PRECISION)
    fh.write("SAMPLING_INTERVAL_S: %s\n" %
             floattostr(stats.delta, 6))
    fh.write("NDATA: %s\n" % floattostr(stats.npts, 0))
    fh.write("DURATION_S: %s\n" %
             floattostr(dyna.DURATION_S, 6))
    fh.write("STREAM: %s\n" % stats.channel)
    fh.write("UNITS: %s\n" % stats.dyna.UNITS)
    fh.write("INSTRUMENT: %s\n" % dyna.INSTRUMENT)
    fh.write("INSTRUMENT_ANALOG/DIGITAL: %s\n" %
             dyna.INSTRUMENT_ANALOG_DIGITAL)
    fh.write("INSTRUMENTAL_FREQUENCY_HZ: %s\n" %
             floattostr(dyna.INSTRUMENTAL_FREQUENCY_HZ, 3))
    fh.write("INSTRUMENTAL_DAMPING: %s\n" %
             floattostr(dyna.INSTRUMENTAL_DAMPING, 6))
    fh.write("FULL_SCALE_G: %s\n" %
             floattostr(dyna.FULL_SCALE_G, 1))
    fh.write("N_BIT_DIGITAL_CONVERTER: %s\n" %
             floattostr(dyna.N_BIT_DIGITAL_CONVERTER, 0))

    # data type is acceleration
    if dyna.DATA_TYPE == "ACCELERATION" \
            or dyna.DATA_TYPE \
            == "ACCELERATION RESPONSE SPECTRUM":
        fh.write("PGA_CM/S^2: %s\n" %
                 floattostr(dyna.PGA_CM_S_2, 6))
        fh.write("TIME_PGA_S: %s\n" %
                 floattostr(dyna.TIME_PGA_S, 6))
    # data type is velocity
    elif dyna.DATA_TYPE == "VELOCITY" \
            or dyna.DATA_TYPE \
            == "PSEUDO-VELOCITY RESPONSE SPECTRUM":
        fh.write("PGV_CM/S: %s\n" %
                 floattostr(dyna.PGV_CM_S, 6))
        fh.write("TIME_PGV_S: %s\n" %
                 floattostr(dyna.TIME_PGV_S, 6))
    # data type is displacement
    elif dyna.DATA_TYPE == "DISPLACEMENT" \
            or dyna.DATA_TYPE \
            == "DISPLACEMENT RESPONSE SPECTRUM":
        fh.write("PGD_CM: %s\n" % floattostr(dyna.PGD_CM, 6))
        fh.write("TIME_PGD_S: %s\n" %
                 floattostr(dyna.TIME_PGD_S, 6))

    fh.write("BASELINE_CORRECTION: %s\n" %
             dyna.BASELINE_CORRECTION)
    fh.write("FILTER_TYPE: %s\n" % dyna.FILTER_TYPE)
    fh.write("FILTER_ORDER: %s\n" %
             floattostr(dyna.FILTER_ORDER, 0))
    fh.write("LOW_CUT_FREQUENCY_HZ: %s\n" %
             floattostr(dyna.LOW_CUT_FREQUENCY_HZ, 3))
    fh.write("HIGH_CUT_FREQUENCY_HZ: %s\n" %
             floattostr(dyna.HIGH_CUT_FREQUENCY_HZ, 3))
    fh.write("LATE/NORMAL_TRIGGERED: %s\n" %
             dyna.LATE_NORMAL_TRIGGERED)
    fh.write("DATABASE_VERSION: %s\n" % dyna.DATABASE_VERSION)
    fh.write("HEADER_FORMAT: DYNA 1.0\n")
    fh.write("DATA_TYPE: %s\n" % dyna.DATA_TYPE)
    fh.write("PROCESSING: %s\n" % dyna.PROCESSING)
    fh.write("DATA_TIMESTAMP_YYYYMMDD_HHMMSS: %s\n" %
             dyna.DATA_TIMESTAMP_YYYYMMDD_HHMMSS)
    fh.write("USER1: %s\n" % dyna.USER1)
    fh.write("USER2: %s\n" % dyna.USER2)
    fh.write("USER3: %s\n" % dyna.USER3)
    fh.write("USER4: %s\n" % dyna.USER4)

    if dyna.DATA_TYPE in ("ACCELERATION", "VELOCITY"):
        for d in stream[0].data:
            fh.write("%-.6f\n" % d)
    elif dyna.DATA_TYPE == "DISPLACEMENT":
        for d in stream[0].data:
            fh.write("%e\n" % d)

    # FIXME: does this really make sense??
    elif dyna.DATA_TYPE[-8:] == "SPECTRUM":
        for j in range(len(stream[0].data)):
            fh.write("%12.6f" % stream[0].data[j])
            fh.write("%13.6f\n" % stream[1].data[j])

    fh.close()

# FIXME: could probably be removed (see above though!) 
def toUTCDateTime(value):
    """
    Converts time string used within DYNA/ITACA File into a UTCDateTime.

    :type value: str
    :param value: A Date time string.
    :return: Converted :class:`~obspy.core.UTCDateTime` object.

    .. rubric:: Example

    >>> toUTCDateTime('20090330_215650.967')
    UTCDateTime(2009, 3, 30, 21, 56, 50, 967000)
    """
    try:
        date, time = value.split('_')
    except ValueError:
        date = value

    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])

    hour = int(time[0:2])
    mins = int(time[2:4])
    secs = float(time[4:])

    return UTCDateTime(year, month, day, hour, mins) + secs


# FIXME: probably to tolerant.
def strtofloat(sf):
    try:
        x = float(sf)
    except:
        return None
    return x


# FIXME: probably to tolerant.
def strtoint(sf):
    try:
        x = int(sf)
    except:
        return None
    return x


# FIXME: QuLogic proposes removal, not sure of the exact purpose.
def floattostr(fs, n):
    y = format("%-.0f" % n)
    try:
        x = eval('format("%-.' + y + 'f" % fs)')
    except:
        return ''
    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
