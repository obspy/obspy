#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Acquisition Helpers for FDSN web services
==============================================

This package contains functionality to query and integrate data from any
number of FDSN web service providers. It can be used by itself or as a
library integrated into a bigger project.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


How it Works
------------

*to be written*

MiniSEED filenames are downloaded in larger chunks and then split at the
file level. This assures that no information present in the original MinSEED
files is lost in the process.


Usage
-----

Using the download helpers requires three distinct steps:

1. **What to download:** Limiting the data to be downloaded geographically,
   temporal, and with some additional constraints.
2. **Where to store:** Defining where the final MiniSEED and StationXML files
   should be stored.
3. **From where to get:** Choose from which provider(s) to download and then
   launch the downloading process.


Data Selection
~~~~~~~~~~~~~~

Data set selection serves the purpose to limit the data to be downloaded to
data useful for the purpose at hand. It handled by two objects: subclasses of
the  :class:`~obspy.fdsn.download_helpers.domain.Domain` object, and
the :class:`~obspy.fdsn.download_helpers.download_helpers.Restrictions` class.

The :class:`~obspy.fdsn.download_helpers.domain` module currently defines
three different domains used to limit the geographical extent of the queried
data: :class:`~obspy.fdsn.download_helpers.domain.RectangularDomain`,
:class:`~obspy.fdsn.download_helpers.domain.CircularDomain`, and
:class:`~obspy.fdsn.download_helpers.domain.GlobalDomain`. Subclassing
:class:`~obspy.fdsn.download_helpers.domain.Domain` enables the construction
of arbitrary complex domains. Please see the
:class:`~obspy.fdsn.download_helpers.domain` module for more details.
Instances of these classes will later be passed to the function sparking the
downloading process. A rectangular domain for example is defined like this:

>>> from obspy.fdsn.download_helpers.domain import RectangularDomain
>>> domain = RectangularDomain(minlatitude=-10, maxlatitude=10,
...                            minlongitude=-10, maxlongitude=10)

Additional restrictions like temporal bounds, SEED identifier wildcards,
and other things are set with the help of
the :class:`~obspy.fdsn.download_helpers.download_helpers.Restrictions` class.
Please refer to its documentation for a more detailed explanation of the
parameters.

>>> from obspy import UTCDateTime
>>> from obspy.fdsn.download_helpers import Restrictions
>>> restrict = Restrictions(
...     starttime=UTCDateTime(2012, 1, 1),
...     endtime=UTCDateTime(2012, 1, 1, 1),
...     network=None, station=None, location=None, channel=None,
...     reject_channels_with_gaps=True,
...     minimum_length=0.9,
...     minimum_interstation_distance_in_m=1000,
...     channel_priorities=["HH[Z,N,E]", "BH[Z,N,E]"],
...     location_priorities=["", "00", "01"]


Storage Options
~~~~~~~~~~~~~~~

After determining what to download, the helpers must know where to store the to
be downloaded data. This requires some flexibility in case this is integrated
as a component into a bigger system. An example of this is a toolbox that has a
database to manage its data. A major concern is to not download already
existing data. In order to enable such a use case the download helpers can be
given functions that are evaluated when determining the filenames of the to be
downloaded data.  Depending on the return value, the helper class will download
the whole, only parts, or even nothing, of that particular piece of data.

Storing MiniSEED waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^

The MiniSEED storage rules are set by the ``mseed_storage`` argument of the
:func:`~obspy.fdsn.download_helpers.download_helpers.DownloadHelper.download`
method of the DownloadHelper class. In the simplest case it is just a folder
name:

>>> mseed_storage = "waveforms"

will cause all MiniSEED files to be stored as
``"waveforms/NETWORK.STATION.LOCATION.CHANNEL.mseed"``.


The second possibility is to provide a string containing ``"{network}"``,
``"{station}"``, ``"{location}"``, and ``"{channel}"`` format specifiers.  The
values will be interpolated to acquire the final filename.

>>> mseed_storage = "some_folder/{network}/{station}/{location}.{channel}.mseed"

The most complex but also most powerful possibility is to use a function which
will be evaluated to determine the filename. If the function returns ``True``,
the MiniSEED file is assumed to already be available and will not be downloaded
again. If it returns a string, the MiniSEED file will be saved to that path.
Utilize closures to use any other paramters in the function. This hypothetical
function checks if the file is already in a database and otherwise returns a
atring.

>>> def get_mseed_storage(network, station, location, channel):
...     if is_in_db(network, station, location, channel):
...         return True
...     return os.path.join(ROOT, "%s.%s.%s.%s.mseed." % (network, station,
...                                                       location, channel))
>>> mseed_storage = get_mseed_storage

.. note::

    No matter which approach is chosen, if a file already exists, it will not
    be overwritten, regardless of the actual contents of the file.


Storing StationXML files
^^^^^^^^^^^^^^^^^^^^^^^^

The same logic applies to the StationXML files. This time the rules are set by
the ``stationxml_storage`` argument of the
:func:`~obspy.fdsn.download_helpers.download_helpers.DownloadHelper.download`
method of the DownloadHelper class. StationXML files will be downloaded on a
per-station basis thus all channels and locations from one station will end up
in the same StationXML file.

A simple string will be interpreted as a folder name. This example will save
the files to ``"stations/NETWORK.STATION.xml"``.

>>> stationxml_storage = "stations"

Another option is to provide a string formatting template, e.g.

>>> stationxml_storage = "some_folder/{network}/{station}.xml"

will write to ``"some_folder/NETWORK/STATION.xml"``.

.. note::

    If the StationXML file already exists, it will be opened to see what is in
    the file. In case it does not contain all necessary channels, it will be
    deleted and only those channels needed in the current run will be
    downloaded again. Pass a custom function to the ``stationxml_path``
    argument if you require different behavior.


As with the waveform data, the StationXML paths can also be set with the help
of a function. The function in this case is a bit more complex then for the
waveform case. It has to return a dictionary with three keys:
``"available_channels"``, ``"missing_channels"``, and ``"filename"``.
``"available_channel"`` is a list of channels that are already available as
station information and that require no new download. Make sure to include all
already available channels; this information is later used to discard
MiniSEED files that have no corresponding station information.
``"missing_channels"`` is a list of channels for that particular station that
must be downloaded and ``"filename"`` determines where to save these. Please
note that in this particular case the StationXML file will be overwritten if it
already exists.

Alternatively the function can also return a string and the behaviour is the
same as two first options for the ``stationxml_storage`` argument.

The next example illustrates a complex use case where the availability of each
channel's station information is queried in some database and only those
channels that do not exist yet will be downloaded. Use closures to pass more
arguments like the temporal constraints of the station information.

>>> def get_stationxml_storage(network, station, channels):
...     available_channels = []
...     missing_channels = []
...     for chan in channels:
...         if is_in_db(network, station, chan.location, chan.channel):
...             available_channels.append(chan)
...         else:
...             missing_channels.append(chan)
...     filename = os.path.join(ROOT, "%s.%s.xml" % (network, station))
...     return {
...         "available_channels": available_channels,
...         "missing_channels": missing_channels,
...         "filename": filename}
>>> stationxml_storage = get_stationxml_storage

Starting the Download
~~~~~~~~~~~~~~~~~~~~~

>>> dlh = DownloadHelper()

>>> dlh.download(domain, restrictions, chunk_size_in_mb=50,
...              thread_per_client=5, mseed_storage=mseed_storage,
...              stationxml_storage=stationxml_storage)


Logging
~~~~~~~
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .download_helpers import DownloadHelper, Restrictions
from .domain import Domain, RectangularDomain, CircularDomain, GlobalDomain
from .utils import format_report
