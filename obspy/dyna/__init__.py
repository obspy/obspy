# -*- coding: utf-8 -*-
"""
obspy_dyna - DYNA and ITACA read and write support for ObsPy
=======================================================================

The obspy_dyna package contains methods in order to read and write seismogram
files in the DYNA and ITACA format as defined by INGV Milano

:copyright:
    The ITACA Development Team (itaca@mi.ingv.it)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
Importing DYNA or ITACA files is done similar to reading any other waveform
data format within ObsPy by using the :func:`~obspy.core.stream.read()` method
of the :mod:`obspy.core` module.

>>> from obspy import read
>>> st = read("/path/to/IT.ARL..HGE.D.20140120.071240.X.ACC.ASC")
>>> st #doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>
>>> print(st)  # doctest: +ELLIPSIS
1 Trace(s) in Stream:
IT.ARL..HGE | 2014-01-20T07:12:30.000000Z - 2014-01-20T07:13:14.980000Z | 200.0 Hz, 8997 samples

The file format will be determined automatically.
The trace will have a stats attribute containing the usual information.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE
         network: IT
         station: ARL
        location:
         channel: HGE
       starttime: 2014-01-20T07:12:30.000000Z
         endtime: 2014-01-20T07:13:14.980000Z
   sampling_rate: 200.0
           delta: 0.005
            npts: 8997
           calib: 1
         _format: DYNA
            dyna: AttribDict({u'LATE_NORMAL_TRIGGERED': u'', u'BASELINE_CORRECTION': u'BASELINE NOT REMOVED', u'DATA_TYPE': u'ACCELERATION', u'STATION_LATITUDE_DEGREE': 41.057068, u'SITE_CLASSIFICATION_EC8': u'B*', u'EARTHQUAKE_BACKAZIMUTH_DEGREE': 346.9, u'FULL_SCALE_G': None, u'USER1': u'MEAN REMOVED (-3.49498661012 cm/s^2)', u'EVENT_LATITUDE_DEGREE': 41.362, u'EVENT_LONGITUDE_DEGREE': 14.449, u'LOW_CUT_FREQUENCY_HZ': None, u'MAGNITUDE_W_REFERENCE': u'', u'DATA_TIMESTAMP_YYYYMMDD_HHMMSS': u'20140127_124152.174', u'FILTER_ORDER': None, u'FOCAL_MECHANISM': u'', u'MAGNITUDE_L_REFERENCE': u'ISIDe', u'STATION_NAME': u'AIROLA', u'PGA_CM_S_2': -1.137432, u'HIGH_CUT_FREQUENCY_HZ': None, u'EVENT_DATE_YYYYMMDD': u'20140120', u'USER4': u'', u'USER2': u'', u'USER3': u'', u'INSTRUMENTAL_DAMPING': None, u'EVENT_ID': u'IT-2014-0003', u'STATION_LONGITUDE_DEGREE': 14.542928, u'PROCESSING': u'none', u'FILTER_TYPE': u'', u'EVENT_DEPTH_KM': 11.1, u'TIME_PGA_S': 23.36, u'DURATION_S': 44.985, u'UNITS': u'cm/s^2', u'INSTRUMENT_ANALOG_DIGITAL': u'D', u'STATION_ELEVATION_M': 504, u'EPICENTRAL_DISTANCE_KM': 34.8, u'MORPHOLOGIC_CLASSIFICATION': u'', u'HYPOCENTER_REFERENCE': u'ISIDe', u'DATE_TIME_FIRST_SAMPLE_PRECISION': u'milliseconds', u'VS30_M_S': None, u'EVENT_NAME': u'NONAME', u'N_BIT_DIGITAL_CONVERTER': None, u'MAGNITUDE_W': None, u'INSTRUMENT': u'sensor = Unknown [Unknown] | digitizer = Unknown [Unknown]', u'EVENT_TIME_HHMMSS': u'071240', u'MAGNITUDE_L': 4.2, u'DATABASE_VERSION': u'ITACA 2.0', u'INSTRUMENTAL_FREQUENCY_HZ': None, u'HEADER_FORMAT': u'DYNA 1.0'})


The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([-0.006763,  0.001193,  0.007042, ..., -0.037417, -0.030865,
       -0.021271], dtype=float32)

Writing
-------
Writing is also done in the usual way:

>>> st.write('filename.ASC', format = 'DYNA') #doctest: +SKIP

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
