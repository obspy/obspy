# -*- coding: utf-8 -*-
"""
Support for reading CyberShake seismograms in ObsPy.
"""

import struct
import numpy as np
from obspy import Trace, Stream


NET_CODE = 'CS'  # Assign a network code for CyberShake data
BAND_INST_CODE = 'MX'  # M indicates mid-period and X indicates synthetic data
LOCATION_CODE = '00'


def _is_cybershake(filename):
    """
    Checks whether a file is a CyberShake seismogram or not.

    :type filename: str
    :param filename: File to be checked.
    :rtype: bool
    :return: ``True`` if a CyberShake version 12.10 file.
    """
    try:
        fp_in = open(filename, "rb")
        header_str = fp_in.read(56)
        split_point = header_str[0:8].find(b'\x00')
        version = header_str[0:split_point].decode()
        assert version == "12.10"
        return True
    except Exception:
        return False


def _read_cybershake(filename, **kwargs):
    """
    Reads a CyberShake seismogram file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: Filename that contains the CyberShake seismogram data.
    """

    fp_in = open(filename, "rb")
    header_str = fp_in.read(56)
    split_point = header_str[8:16].find(b'\x00')
    site = header_str[8:8+split_point].decode()
    dt = struct.unpack('f', header_str[36:40])[0]
    nt = struct.unpack('i', header_str[40:44])[0]

    source_id = struct.unpack('i', header_str[24:28])[0]
    rupture_id = struct.unpack('i', header_str[28:32])[0]
    rup_var_id = struct.unpack('i', header_str[32:36])[0]

    data_str = fp_in.read(4 * nt)
    x_data = np.array(struct.unpack("%df" % nt, data_str))
    data_str = fp_in.read(4 * nt)
    y_data = np.array(struct.unpack("%df" % nt, data_str))
    fp_in.close()

    traces = []
    for data, orientation_code in zip([x_data, y_data], ['E', 'N']):
        header = {
            'network': NET_CODE,
            'station': site,
            'location': LOCATION_CODE,
            'channel': BAND_INST_CODE + orientation_code,
            'delta': dt,
            'cybershake': {
                'source_id': source_id,
                'rupture_id': rupture_id,
                'rup_var_id': rup_var_id
            }
        }
        traces.append(Trace(data, header))
    return Stream(traces)
