# -*- coding: utf-8 -*-
"""
obspy.mseed - MiniSEED read and write support
=============================================
This module provides read and write support for Mini-SEED waveform data and
some other convinient methods to handle Mini-SEED files. Most methods are based
on libmseed, a C library framework by Chad Trabant and interfaced via python
ctypes.

:copyright: The ObsPy Development Team (devs@obspy.org) & Chad Trabant
:license: GNU General Public License (GPLv2)

Reading via obspy.core
----------------------
Similiar to reading any other waveform data format using obspy.core:

>>> from obspy.core import read
>>> st = read('COP.BHE.DK.2009.050')
>>> st
<obspy.core.stream.Stream object at 0x101700150>
>>> print st
1 Trace(s) in Stream:
DK.COP..BHE | 2009-02-19T00:00:00.035100Z - 2009-02-19T23:59:59.985100Z | 20.0 Hz, 1728000 samples

The format will be determined automatically.

Each trace will have a stats attribute containing the usual information. When
reading a Mini-SEED file it will have one additional attribute: 'mseed'. This
attribute contains all Mini-SEED specific attributes which actually is just the
dataquality.

>>> print st[0].stats
Stats({'network': 'DK',
    'mseed': Stats({'dataquality': 'D'}),
    'station': 'COP',
    'location': '',
    'starttime': UTCDateTime(2009, 2, 19, 0, 0, 0, 35100), 
    'npts': 1728000, 
    'sampling_rate': 20.0,
    'endtime': UTCDateTime(2009, 2, 19, 23, 59, 59, 985100), 
    'channel': 'BHE'
})

The data is stored in the data attribut.

>>> st[0].data
array([1085, 1167, 1131, ...,  -19,  -46,  -55], dtype=int32)

Writing via obspy.core
----------------------
WARNING: No matter what is written in the Stream[i].stats.mseed['dataquality']
field, the dataquality in the resulting Mini-SEED file will always be 'D'.

Writing is also done in the usual way.

>>> st.write('Mini-SEED-filename.mseed', format = 'MSEED')

You can also specify several keyword arguments that change the resulting Mini-SEED file:

* reclen : Record length in bytes of the resulting Mini-SEED file. The record
  length needs to be expressible as 2 to the power of X where X is in between and
  including 8 and 20. If no reclen is given it will default to 4096 bytes.
* encoding: Encoding of the Mini-SEED file. You can either give the a string or the corresponding number. If no encoding is given it will default to STEIM2. Available encodings:
  |   o INT16 or 1
  |   o INT32 or 3
  |   o STEIM1 or 10
  |   o STEIM2 or 11 
* byteorder: Byte order of the Mini-SEED file. 0 will result in a little-endian
  file and 1 in a big-endian file. Defaults to big-endian. Do not change this if
  you don't know what you are doing because most other programs can only read
  big-endian Mini-SEED files.
* flush: If it is not zero all of the data will be packed into records,
  otherwise records will only be packed while there are enough data samples to
  completely fill a record. The default value is -1 and thus every data value
  will be packed by default.
* verbose: Controls verbosity of the underlaying libmseed. A value higher than 0 will give diagnostic output. Defaults to 0. 

So in order to write a STEIM1 encoded Mini-SEED file with a record_length of 512 byte do the following:

>>> st.write('Mini-SEED-filename.mseed', format = 'MSEED', reclen = 512, encoding = 'STEIM1')

Additonal methods of obspy.mseed
--------------------------------
All of the following methods can only be accessed with an instance of the libmseed class.

>>> from obspy.mseed import libmseed
>>> mseed = libmseed()
>>> mseed
<obspy.mseed.libmseed.libmseed object at 0x10178ef50>

printFileInformation
^^^^^^^^^^^^^^^^^^^^
Prints some informations about the file.

Parameters:
    * filename = MiniSEED file. 

>>> mseed.printFileInformation('COP.BHE.DK.2009.050')
   Source                Start sample             End sample        Gap  Hz  Samples
DK_COP__BHE_D     2009-02-19T00:00:00.035100 2009-02-19T23:59:59.985100  ==  20  1728000
Total: 1 trace segment(s)

isMSEED
^^^^^^^
Tests whether a file is a MiniSEED file or not. Returns True on success or False otherwise.

This method only reads the first seven bytes of the file and checks whether it
is a MiniSEED or fullSEED file. It also is true for fullSEED files because
libmseed can read the data part of fullSEED files. If the method finds a
fullSEED file it also checks if it has a data part and returns False otherwise.
Thus it cannot be used to validate a MiniSEED or SEED file.

Parameters:
    * filename = MiniSEED file. 

>>> mseed.isMSEED('COP.BHE.DK.2009.050')
True

getDataQualityFlagsCount
^^^^^^^^^^^^^^^^^^^^^^^^
Counts all data quality flags of the given MiniSEED file. This method will
count all set data quality flag bits in the fixed section of the data header in
a MiniSEED file and returns the total count for each flag type.

Data quality flags:

========  =================================================
Bit       Description
========  =================================================
[Bit 0]   Amplifier saturation detected (station dependent)
[Bit 1]   Digitizer clipping detected
[Bit 2]   Spikes detected
[Bit 3]   Glitches detected
[Bit 4]   Missing/padded data present
[Bit 5]   Telemetry synchronization error
[Bit 6]   A digital filter may be charging
[Bit 7]   Time tag is questionable
========  =================================================

This will only work correctly if each record in the file has the same record length.
Parameters:
    * filename = MiniSEED file. 

>>> mseed.getDataQualityFlagsCount('qualityflags.mseed')
[9, 8, 7, 6, 5, 4, 3, 2]

getTimingQuality
^^^^^^^^^^^^^^^^
Reads timing quality and returns a dictionary containing statistics about it.
This method will read the timing quality in Blockette 1001 for each record in
the file if available and return the following statistics:
Minima, maxima, average, median and upper and lower quantile. It is probably
pretty safe to set the first_record parameter to True because the timing
quality is a vendor specific value and thus it will probably be set for each
record or for none.

Parameters:
* filename = MiniSEED file.
* first_record: Determines whether all records are assumed to either have a
  timing quality in Blockette 1001 or not depending on whether the first records
  has one. If True and the first records does not have a timing quality it will
  not parse the whole file. If False is will parse the whole file anyway and
  search for a timing quality in each record. Defaults to True.
* rl_autodetection: Determines the auto-detection of the record lengths in the
  file. If 0 only the length of the first record is detected automatically. All
  subsequent records are then assumed to have the same record length. If -1 the
  length of each record is automatically detected. Defaults to -1. 

>>> mseed.getTimingQuality('timingquality.mseed')
{'average': 50.0,
 'lower_quantile': 25.0,
 'max': 100.0,
 'median': 50.0,
 'min': 0.0,
 'upper_quantile': 75.0}

getFirstRecordHeaderInfo
^^^^^^^^^^^^^^^^^^^^^^^^
Takes a MiniSEED file and returns some header information from the first record.

Returns a dictionary containing some header information from the first record
of the MiniSEED file only. It returns the location, network, station and
channel information. The advantage is that this method is very fast but keep in
mind that it will only parse the first record.

Parameters:
    * filename = MiniSEED file. 

>>> mseed.getFirstRecordHeaderInfo('COP.BHE.DK.2009.050')
{'channel': 'BHE', 'location': '', 'network': 'DK', 'station': 'COP'}
"""

from obspy.core.util import _getVersionString
from obspy.mseed.libmseed import LibMSEED


__version__ = _getVersionString("obspy.mseed")
