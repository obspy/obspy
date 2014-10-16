#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Acquisition Helpers for FDSN web services
==============================================
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

>>> import obspy
>>> from obspy.fdsn.download_helpers import Restrictions, DownloadHelper
>>> import os

The to be downloaded data set is restricted in two ways. Geographical
constraints are imposed with the use of a subclass of
:class:`~obspy.fdsn.download_helpers.domain.Domain`, see the domain
documentation for more details.

The :class:`~obspy.fdsn.download_helpers.download_helpers.Restrictions` class
is further limit the data set.

>>> restrict = Restrictions(
...     starttime=obspy.UTCDateTime(2012, 1, 1),
...     endtime=obspy.UTCDateTime(2012, 1, 1, 1),
...     network=None, station=None, location=None, channel=None,
...     reject_channels_with_gaps=True, minimum_length=0.9,
...     minimum_interstation_distance_in_m=1000,
...     channel_priorities=["HH[Z,N,E]", "BH[Z,N,E]"],
...     location_priorities=["", "00", "01"]

Storage options
---------------

Filepaths are not fixed to assure the download helpers are usable as a
component of some bigger system. Therefore the paths where data is saved are
flexible.

Storing MiniSEED waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^

MiniSEED filenames are downloaded in larger chunks and then split at the
file level. This assures that no information present in the original MinSEED
files is lost in the process.

In the simplest case it is just a folder name. All MiniSEED files will then
be saved in ``"FOLDER/NETWORK.STATION.LOCATION.CHANNEL.mseed"``.

>>> mseed_path = "waveforms"

The second possibility is to provide a string containing ``"{network}"``,
``"{station}"``, ``"{location}"``, and ``"{channel}"``. The values will be
interpolated to acquire the final filename.

>>> mseed_path = "some_folder/{network}/{station}/{location}.{channel}.mseed"

The most complex but also most powerful possibility is to use a function
which will be evaluated to determine the filename. If the function returns
``True``, the MiniSEED file is assumed to already be available and will not be
downloaded again. If it returns a string, the MiniSEED file will be saved to
that string.

>>> def get_mseed_path(network, station, location, channel):
...     if is_in_db(network, station, location, channel):
...         return True
...     return os.path.join(ROOT, "%s.%s.%s.%s.mseed." % (network, station,
...                                                       location, channel))

>>> mseed_path = get_mseed_path

No matter which approach is chosen, if the file already exists, it will not
be downloaded again.

Storing StationXML files
^^^^^^^^^^^^^^^^^^^^^^^^

The same logic applies to the StationXML files. A simple string will be
interpreted as a folder name. This example will save the files to
``"stations/NETWORK.STATION.xml"``.

>>> stationxml_path = "stations"

Another option is to provide a string formatting templates, e.g.

>>> stationxml_path = "some_folder/{network}/{station}.xml"

will write to ``"some_folder/NETWORK/STATION.xml"``.

The function in this case is a bit more complex then for the waveform case.
It has to return a dictionary with the available channels to not be
downloaded again, the missing channels, that require downloading and
filename for these.

>>> def get_stationxml_path(network, station, channels, time_of_interest):
...     available_channels = []
...     missing_channels = []
...     for chan in channels:
...         if is_in_db(network, station, chan.location, chan.channel,
...                     time_of_interest):
...             available_channels.append((network, station, chan))
...         else:
...             missing_channels.append((network, station, chan))
...     filename = os.path.join(ROOT, "%s.%s.xml" % (network, station))
...     return {
...         "available_channels": available_channels,
...         "missing_channels": missing_channels,
...         "filename": filename}

>>> stationxml_path = get_stationxml_path


Starting the Download
---------------------

>>> dlh = DownloadHelper()

>>> dlh.download(domain, restrictions, chunk_size_in_mb=50,
...              thread_per_client=5, mseed_path=mseed_path,
...              stationxml_path=stationxml_path)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .download_helpers import DownloadHelper, Restrictions
from .domain import Domain, RectangularDomain, CircularDomain, GlobalDomain
from .utils import format_report