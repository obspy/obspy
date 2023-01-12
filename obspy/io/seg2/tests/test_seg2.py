# -*- coding: utf-8 -*-
"""
The obspy.io.seg2 test suite.
"""
import gzip
import warnings

import numpy as np

from obspy import read


TRACE2_HEADER = {'ACQUISITION_DATE': '07/JAN/2013',
                 'ACQUISITION_DATE_UTC': '07/JAN/2013',
                 'ACQUISITION_TIME': '10:30:41',
                 'ACQUISITION_TIME_MICROSECONDS': '0',
                 'ACQUISITION_TIME_MICROSECONDS_UTC': '0',
                 'ACQUISITION_TIME_UTC': '09:30:41',
                 'APPLIED_STANDARD': '0',
                 'BATTERY_LEVEL': '99 0 30.35 4.123',
                 'CHANNEL_NUMBER': '1',
                 'CLIENT': 'CLIENT',
                 'CLOCK_SOURCE': '1',
                 'COMPANY': 'COMPANY',
                 'DESCALING_FACTOR': '2.17378e-05',
                 'DEVICE_NAME': 'VIPA 15',
                 'DEVICE_SERIAL_NO': '01-0000143912a3',
                 'DIFFERENCE_TO_REAL_TIME': '0.000000',
                 'FIRMWARE_VERSION': '0.0.125',
                 'FIRST_SAMPLE_NO': '8000',
                 'HIGH_CUT_FILTER': '0 0',
                 'INSTRUMENT': 'DMT_VIPA_01-0000143912a3',
                 'LAST_SAMPLE_NO': '9999',
                 'LOCATION': 'LOCATION',
                 'LOW_CUT_FILTER': '10.000000 12.000000',
                 'NETWORK_NAME': 'BANK',
                 'NOTE': ['Comment'],
                 'OBSERVER': 'OBSERVER',
                 'REAL_TIME_AVAILABLE': 'FALSE',
                 'REGISTRATION_DIRECTION': 'X',
                 'SAMPLE_INTERVAL': '0.00100000',
                 'SCALE_UNIT': 'mm/s',
                 'SECONDS_SINCE_LAST_GPS_SYNC': '-1',
                 'SENSOR_CALIB_DATE': '21/8/12',
                 'SENSOR_FC': '4.500000',
                 'SENSOR_TYPE_ID': '1',
                 'SENSOR_TYPE_NAME': 'DMT-3D/DIN',
                 'STATION_CODE': 'BA1',
                 'STATION_NAME': 'DMT-BANK',
                 'SYSTEM_T0': '0',
                 'TIME_ZONE': 'CET',
                 'TRACE_TYPE': 'SEISMIC_DATA',
                 'TRIGGER_LEVEL': '2.00000000',
                 'TRIGGER_SAMPLE_NO': '8000',
                 'TRIGGER_SOURCE': '0',
                 'UNITS': 'METERS'}
TRACE3_HEADER = {
    'ACQUISITION_DATE': '7/MAR/2018',
    'ACQUISITION_TIME': '3:12:45',
    'INSTRUMENT': 'GEOMETRICS SmartSeis 0000',
    'TRACE_SORT': 'AS_ACQUIRED',
    'UNITS': 'METERS',
    'NOTE': ['DISPLAY_SCALE 48'],
    'CHANNEL_NUMBER': '1',
    'DELAY': '-0.010',
    'DESCALING_FACTOR': '0.001199',
    'LINE_ID': '00-00',
    'LOW_CUT_FILTER': '0 0',
    'NOTCH_FREQUENCY': '0',
    'RAW_RECORD': '1068.DAT',
    'RECEIVER_LOCATION': '1004.00',
    'SAMPLE_INTERVAL': '0.000125',
    'SKEW': '-0.00001796',
    'SOURCE_LOCATION': '1000.00',
    'STACK': '8',
}


class TestSEG2():
    """
    Test cases for SEG2 reading.
    """
    def test_read_data_format_2(self, datapath):
        """
        Test reading a SEG2 data format code 2 file (int32).
        """
        basename = str(datapath / '20130107_103041000.CET.3c.cont.0')
        # read SEG2 data (in counts, int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            st = read(basename + ".seg2.gz")
        assert len(w) == 1
        assert 'custom' in str(w[0])
        # read reference ASCII data (in micrometer/s)
        with gzip.open(basename + ".DAT.gz", 'rb') as f:
            results = np.loadtxt(f).T
        # test all three components
        for tr, result in zip(st, results):
            # convert raw data to micrometer/s (descaling goes to mm/s)
            scaled_data = tr.data * float(tr.stats.seg2.DESCALING_FACTOR) * 1e3
            assert np.allclose(scaled_data, result, rtol=1e-7, atol=1e-7)
        # test seg2 specific header values
        # (trace headers include SEG2 file header)
        assert st[0].stats.seg2 == TRACE2_HEADER

    def test_read_data_format_3(self, datapath):
        """
        Test reading a SEG2 data format code 3 file (20-bit floating point).
        """
        basename = str(datapath / '20180307_031245000.0')
        # read SEG2 data (in counts, int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            st = read(basename + ".seg2")
        assert len(w) == 2
        assert 'custom' in str(w[1])
        # read reference ASCII data
        with gzip.open(basename + '.DAT.gz', 'rb') as f:
            results = np.loadtxt(f, ndmin=2).T
        for tr, result in zip(st, results):
            # convert raw data to unit'd
            scaled_data = tr.data * tr.stats.calib
            np.testing.assert_allclose(scaled_data, result,
                                       rtol=1e-7, atol=1e-7)
        # test seg2 specific header values
        # (trace headers include SEG2 file header)
        assert st[0].stats.seg2 == TRACE3_HEADER
