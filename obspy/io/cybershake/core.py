# -*- coding: utf-8 -*-
"""
Support for reading CyberShake seismograms in ObsPy.
"""
import io
import struct
from pathlib import Path

import numpy as np

from obspy import Trace, Stream


NETWORK_CODE = 'CS'  # Assign a network code for CyberShake data
INSTRUMENT_CODE = 'X'  # X indicates synthetic data
LOCATION_CODE = '00'


def _is_cybershake(filename):
    """
    Checks whether a file is a CyberShake version 12.10 seismogram file or not.

    :param filename: file to be checked.
    :type filename: str, open file, or file-like object
    :rtype: bool
    :return: ``True`` if a CyberShake version 12.0 seismogram file.
    """

    if isinstance(filename, io.BufferedIOBase):
        starting_pos = filename.tell()
        is_cybershake_result = _internal_is_cybershake(filename)
        filename.seek(starting_pos, 0)
        return is_cybershake_result
    elif isinstance(filename, (str, bytes, Path)):
        with open(filename, "rb") as fh:
            return _internal_is_cybershake(fh)
    else:
        raise ValueError("Cannot open '%s'." % filename)


def _internal_is_cybershake(buf):
    """
    Checks whether a file-like object contains a CyberShake file or not.

    :param buf: CyberShake file to be checked.
    :type buf: file-like object or open file
    :rtype: bool
    :return: ``True`` if a CyberShake file.
    """
    try:
        header_str = buf.read(56)
        split_point = header_str[0:8].find(b'\x00')
        version = header_str[0:split_point].decode()
        assert version == "12.10"
        return True
    except Exception:
        return False


def _read_cybershake(filename, band_code='M', **kwargs):
    """
    Reads a CyberShake seismogram file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: Filename that contains the CyberShake seismogram data.
    :param band_code: A single character associated with the period band.
        Default is 'M' for mid-period.
    """
    if isinstance(filename, io.BufferedIOBase):
        return _internal_read_cybershake(buf=filename, band_code=band_code,
                                         **kwargs)
    else:
        with open(filename, "rb") as fh:
            return _internal_read_cybershake(buf=fh, band_code=band_code,
                                             **kwargs)


def _internal_read_cybershake(buf, band_code='M', **kwargs):
    """
    Reads a CyberShake seismogram file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param buf: Filename that contains the CyberShake seismogram data.
    :type buf: file or file-like object
    :param band_code: A single character associated with the period band.
        Default is 'M' for mid-period.
    :type band_code: str
    """

    header_str = buf.read(56)
    split_point = header_str[8:16].find(b'\x00')
    site = header_str[8:8+split_point].decode()

    source_id = struct.unpack('i', header_str[24:28])[0]
    rupture_id = struct.unpack('i', header_str[28:32])[0]
    rup_var_id = struct.unpack('i', header_str[32:36])[0]
    dt = struct.unpack('f', header_str[36:40])[0]
    nt = struct.unpack('i', header_str[40:44])[0]
    det_max_freq = struct.unpack('f', header_str[48:52])[0]
    stoch_max_freq = struct.unpack('f', header_str[52:56])[0]

    data_str = buf.read(4 * nt)
    x_data = np.array(struct.unpack("%df" % nt, data_str))
    data_str = buf.read(4 * nt)
    y_data = np.array(struct.unpack("%df" % nt, data_str))

    traces = []
    for data, orientation_code in zip([x_data, y_data], ['E', 'N']):
        header = {
            'network': NETWORK_CODE,
            'station': site,
            'location': LOCATION_CODE,
            'channel': band_code + INSTRUMENT_CODE + orientation_code,
            'delta': dt,
            'cybershake': {
                'source_id': source_id,
                'rupture_id': rupture_id,
                'rup_var_id': rup_var_id,
                'det_max_freq': det_max_freq,
                'stoch_max_freq': stoch_max_freq
            }
        }
        traces.append(Trace(data, header))
    return Stream(traces)
