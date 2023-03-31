"""
obspy.io.csv - CSV, CSZ and EVENTTXT read/write support for earthquake catalogs
===============================================================================


Usage CSV
---------

CSV format can be used to store a catalog with basic origin properties.
Picks cannot be stored.

>>> from obspy import read_events
>>> events = read_events('/path/to/catalog.csv')
>>> print(events)  # doctest: +NORMALIZE_WHITESPACE
3 Event(s) in Catalog:
2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4  mb
2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3  ML
2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0  ML
>>> events.write('local_catalog.csv', 'CSV')  # declare 'CSV' as format
>>> with open('local_catalog.csv') as f: print(f.read())
id,time,lat,lon,dep,magtype,mag
20120404_0000041,2012-04-04T14:21:42.30000,41.818000,79.689000,1.000,mb,4.40
20120404_0000038,2012-04-04T14:18:37.00000,39.342000,41.044000,14.400,ML,4.30
20120404_0000039,2012-04-04T14:08:46.00000,38.017000,37.736000,7.000,ML,3.00
<BLANKLINE>

It is possible to load arbitrary CSV files. Define the field names in the code
or use the first line in the file to define the field names.
The following field names have to be used to read the origin time:
`time` (UTC time string) or ``'year, mon, day, hour, minu, sec'``.
The following additional field names have to be used:
``lat, lon, dep, mag``. ``magtype``, ``id`` and some other fields are optional.
For external CSV files, the format ``'CSV'`` has to be explicitly specified.

>>> from obspy.core.util import get_example_file
>>> with open(get_example_file('external.csv')) as f: print(f.read())
Year, Month, Day, Hour, Minute, Seconds, code, Lat, Lon, Depth, Magnitude, ID
2023, 05, 06, 19, 55, 01.3, LI, 10.1942, 124.8300, 50.47, 0.2, 2023abcde
<BLANKLINE>
>>> names = 'year mon day hour minu sec _ lat lon dep mag id'
>>> events = read_events('/path/to/external.csv', 'CSV', skipheader=1, names=names)
>>> print(events)  # doctest: +NORMALIZE_WHITESPACE
1 Event(s) in Catalog:
2023-05-06T19:55:01.300000Z | +10.194, +124.830 | 0.2


Usage EVENTTXT
--------------

The EVENTTXT format is a flavour of CSV, reading and writing is directly
supported.

>>> from obspy import read_events
>>> print(read_events('/path/to/events.txt'))  # doctest: +NORMALIZE_WHITESPACE
2 Event(s) in Catalog:
2012-04-11T08:38:37.000000Z |  +2.238,  +93.014 | 8.6  MW
1960-05-22T19:11:14.000000Z | -38.170,  -72.570 | 8.5
>>> print(read_events('https://service.iris.edu/fdsnws/event/1/query?minmagnitude=8.5&format=text&endtime=2020-01-01'))  # doctest: +NORMALIZE_WHITESPACE
7 Event(s) in Catalog:
2012-04-11T08:38:37.000000Z |  +2.238,  +93.014 | 8.6  MW
2011-03-11T05:46:23.000000Z | +38.296, +142.498 | 9.1  MW
2010-02-27T06:34:13.000000Z | -36.148,  -72.933 | 8.8  MW
2007-09-12T11:10:26.000000Z |  -4.464, +101.396 | 8.5  MW
2005-03-28T16:09:35.000000Z |  +2.096,  +97.113 | 8.6  MW
2004-12-26T00:58:52.000000Z |  +3.413,  +95.901 | 9.0  MW
1960-05-22T19:11:14.000000Z | -38.170,  -72.570 | 8.5



Usage CSZ
---------

CSZ format can be used to store a catalog with picks in a set of csv files
zipped into a single file.
It works similar to NumPy's npz format.
Compression may be used with ``compression`` and ``compresslevel`` parameters
(see `zipfile doc <https://docs.python.org/library/zipfile.html#zipfile.ZipFile>`_).

>>> events = read_events('/path/to/example.pha')
>>> print(events)
2 Event(s) in Catalog:
2025-05-14T14:35:35.510000Z | +40.225,  +10.450 | 3.5  None
2025-05-14T15:43:05.280000Z | +40.223,  +10.450 | 1.8  None
>>> print(len(events[0].picks))
2
>>> events.write('catalog.csz', 'CSZ')
>>> events2 = read_events('catalog.csz')
>>> print(events2)  # doctest: +NORMALIZE_WHITESPACE
2 Event(s) in Catalog:
2025-05-14T14:35:35.510000Z | +40.225,  +10.450 | 3.5
2025-05-14T15:43:05.280000Z | +40.223,  +10.450 | 1.8
>>> print(len(events2[0].picks))
2

Load CSV/CSZ/EVENTTXT file into numpy array
-------------------------------------------

For plotting, e.t.c, it is useful to represent the event paramters with a numpy
array.
The :func:`load_csv` function can be used to load a CSV or CSZ file as numpy array.
The :func:`load_eventtxt` function ca be used to load an EVENTTXT file as numpy array.

>>> from obspy.io.csv import load_csv, load_eventtxt
>>> t = load_csv('/path/to/catalog.csv')
>>> print(t)  # doctest: +NORMALIZE_WHITESPACE
[ ('20120404_0000041', '2012-04-04T14:21:42.300',  41.818,  79.689,   1. , 'mb',  4.4)
  ('20120404_0000038', '2012-04-04T14:18:37.000',  39.342,  41.044,  14.4, 'ML',  4.3)
  ('20120404_0000039', '2012-04-04T14:08:46.000',  38.017,  37.736,   7. , 'ML',  3. )]
 >>> print(t['mag'])
 [ 4.4  4.3  3. ]
 >>> t2 = load_eventtxt('/path/to/events.txt')
 >>> print(t2)  # doctest: +NORMALIZE_WHITESPACE
 [ ('3337497', '2012-04-11T08:38:37.000', 2.2376, 93.0144,  26.3, 'MW',  8.6)
   ('2413', '1960-05-22T19:11:14.000', -38.17  , -72.57  ,   0. , '',  8.5)]

Convert ObsPy catalog into numpy array
--------------------------------------

The :func:`_events2array` function  can be used to convert an ObsPy catalog to numpy array.
Code example creating event plots::

    import matplotlib.pyplot as plt
    from obspy import read_events
    from obspy.io.csv import _events2array
    events = read_events()
    t = _events2array(events)
    plt.subplot(121)
    plt.scatter(t['lon'], t['lat'], 4*t['mag']**2)
    plt.subplot(122)
    plt.scatter(t['time'], t['mag'], 4*t['mag']**2)
    plt.show()
"""

from obspy.io.csv.core import _events2array, load_csv, load_eventtxt
