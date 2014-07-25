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
IT.ARL..E | 2014-01-20T07:12:30.000000Z - 2014-01-20T07:13:14.980000Z | 200.0 Hz, 8997 samples

The file format will be determined automatically.
The trace will have a stats attribute containing the usual information.

>>> print(st[0].stats) #doctest: +NORMALIZE_WHITESPACE
         network: IT
         station: ARL
        location: 
         channel: E
       starttime: 2014-01-20T07:12:30.000000Z
         endtime: 2014-01-20T07:13:14.980000Z
   sampling_rate: 200.0
           delta: 0.005
            npts: 8997
           calib: 1
         _format: DYNA_10
            dyna: AttribDict({'LATE_NORMAL_TRIGGERED': '', 'BASELINE_CORRECTION': 'BASELINE NOT REMOVED', 'STREAM': 'HGE', 'STATION_LATITUDE_DEGREE': 41.057068, 'SITE_CLASSIFICATION_EC8': 'B*', 'EARTHQUAKE_BACKAZIMUTH_DEGREE': 346.9, 'FULL_SCALE_G': None, 'USER1': 'MEAN REMOVED (-3.49498661012 cm/s^2)', 'EVENT_LATITUDE_DEGREE': 41.362, 'EVENT_LONGITUDE_DEGREE': 14.449, 'DATA_TYPE': 'ACCELERATION', 'LOW_CUT_FREQUENCY_HZ': None, 'MAGNITUDE_W_REFERENCE': '', 'DATA_TIMESTAMP_YYYYMMDD_HHMMSS': '20140127_124152.174', 'FILTER_ORDER': None, 'FOCAL_MECHANISM': '', 'MAGNITUDE_L_REFERENCE': 'ISIDe', 'STATION_NAME': 'AIROLA', 'PGA_CM_S_2': -1.137432, 'HIGH_CUT_FREQUENCY_HZ': None, 'EVENT_DATE_YYYYMMDD': '20140120', 'USER4': '', 'USER2': '', 'USER3': '', 'INSTRUMENTAL_DAMPING': None, 'EVENT_ID': 'IT-2014-0003', 'STATION_LONGITUDE_DEGREE': 14.542928, 'PROCESSING': 'none', 'FILTER_TYPE': '', 'EVENT_DEPTH_KM': 11.1, 'TIME_PGA_S': 23.36, 'DURATION_S': 44.985, 'UNITS': 'cm/s^2', 'INSTRUMENT_ANALOG_DIGITAL': 'D', 'STATION_ELEVATION_M': 504, 'EPICENTRAL_DISTANCE_KM': 34.8, 'MORPHOLOGIC_CLASSIFICATION': '', 'HYPOCENTER_REFERENCE': 'ISIDe', 'DATE_TIME_FIRST_SAMPLE_PRECISION': 'milliseconds', 'VS30_M_S': None, 'EVENT_NAME': 'NONAME', 'N_BIT_DIGITAL_CONVERTER': None, 'MAGNITUDE_W': None, 'INSTRUMENT': 'sensor = Unknown [Unknown] | digitizer = Unknown [Unknown]', 'EVENT_TIME_HHMMSS': '071240', 'MAGNITUDE_L': 4.2, 'DATABASE_VERSION': 'ITACA 2.0', 'INSTRUMENTAL_FREQUENCY_HZ': None, 'HEADER_FORMAT': 'DYNA 1.0'})


The actual data is stored as numpy.ndarray in the data attribute of each trace.

>>> st[0].data #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
array([-0.006763,  0.001193,  0.007042, ..., -0.037417, -0.030865,
       -0.021271], dtype=float32)

Writing
-------
Writing is also done in the usual way:

>>> st.write('filename.ASC', format = 'DYNA') #doctest: +SKIP

"""

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
