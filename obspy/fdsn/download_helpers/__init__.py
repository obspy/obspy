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


Step 1: Data Selection
~~~~~~~~~~~~~~~~~~~~~~

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


Step 2: Storage Options
~~~~~~~~~~~~~~~~~~~~~~~

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
``"waveforms/NETWORK.STATION.LOCATION.CHANNEL.STARTTIME.ENDTIME.mseed"``.


The second possibility is to provide a string containing ``"{network}"``,
``"{station}"``, ``"{location}"``, ``"{channel}"``, ``"{starttime}"``, and
``"{endtime}"`` format specifiers. The values will be interpolated to acquire
the final filename.

>>> mseed_storage = ("some_folder/{network}/{station}/"
...                  "{location}.{channel}.{starttime}.{endtime}.mseed")

The most complex but also most powerful possibility is to use a function which
will be evaluated to determine the filename. If the function returns ``True``,
the MiniSEED file is assumed to already be available and will not be downloaded
again; keep in mind that in that case no station data will be downloaded for
that channel. If it returns a string, the MiniSEED file will be saved to that
path. Utilize closures to use any other paramters in the function. This
hypothetical function checks if the file is already in a database and otherwise
returns a string.

>>> def get_mseed_storage(network, station, location, channel, starttime,
...                       endtime):
...     if is_in_db(network, station, location, channel, starttime, endtime):
...         return True
...     return os.path.join(ROOT, "%s.%s.%s.%s.mseed." % (network, station,
...                                                       location, channel))
>>> mseed_storage = get_mseed_storage

.. note::

    No matter which approach is chosen, if a file already exists, it will not
    be overwritten; it will be parsed and the download helper class will
    attempt to download matching StationXML files.


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

Step 3: Starting the Download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> dlh = DownloadHelper()

>>> dlh.download(domain, restrictions, chunk_size_in_mb=50,
...              thread_per_client=5, mseed_storage=mseed_storage,
...              stationxml_storage=stationxml_storage)


Logging
~~~~~~~

*to be written*


Examples
--------

This section illustrates the usage of the download helpers for a few typical
examples which can serve as templates for your own needs.


Earthquake Data
~~~~~~~~~~~~~~~

One of the most often used type of data set in seismology is a classical
earthquake data set consisting of waveform recordings for a certain earthquake.
The following example downloads all data it can find for the Tohoku-Oki
Earthquake from 5 minutes before the earthquake centroid time to 1 hour after.
It will furthermore only download data with a distance between 70.0 and 90.0
degrees from the data and some additional restrictions.

.. code:: python

    import obspy
    from obspy.fdsn.download_helpers import CircularDomain, Restrictions, \\
        DownloadHelper

    origin_time = obspy.UTCDateTime(2011, 3, 11, 5, 47, 32)

    # Circular domain around the epicenter. This will download all data between
    # 70 and 90 degrees distance from the epicenter.
    domain = CircularDomain(latitude=37.52, longitude=143.04,
                            minradius=70.0, maxradius=90.0)

    restrictions = Restrictions(
        # Get data from 5 minutes before the event to one hours after the
        # event.
        starttime=origin_time - 5 * 60,
        endtime=origin_time + 3600,
        # You might not want to deal with gaps in the data.
        reject_channels_with_gaps=True,
        # And you might only want waveform that have data for at least 95 % of
        # the requested time span.
        minimum_length=0.95,
        # No two stations should be closer than 10 km to each other.
        minimum_interstation_distance_in_m=10E3,
        # Only HH or BH channels. If a station has HH channels, those will be
        # downloaded, otherwise the BH. Nothing will be downloaded if it has
        # neither.
        channel_priorities=("HH[Z,N,E]", "BH[Z,N,E]"),
        # Locations codes are arbitrary and there is no rule which location is
        # best.
        location_priorities=("", "00", "10"))

    # No specified providers will result in all known ones being queried.
    dlh = DownloadHelper()
    dlh.download(domain, restrictions, mseed_storage="waveforms",
                 stationxml_storage="stations")


Continuous Request
~~~~~~~~~~~~~~~~~~

Ambient seismic noise correlations require continuous recordings from stations
over a large time span. This example downloads data, from within a certain
geographical domain, for a whole year. Individual MiniSEED files will be split
per day.

.. code:: python

    import obspy
    from obspy.fdsn.download_helpers import RectangularDomain, Restrictions, \\
        DownloadHelper

    # Rectangular domain containing parts of southern Germany.
    domain = RectangularDomain(minlatitude=30, maxlatitude=50,
                               minlongitude=5, maxlongitude=25)

    restrictions = Restrictions(
        # Get data for a whole year.
        starttime=obspy.UTCDateTime(2012, 1, 1),
        endtime=obspy.UTCDateTime(2013, 1, 1),
        # Chunk it to have one file per day.
        chunklength=86400,
        # Considering the enormous amount of data associated with continuous
        # requests, you might want to limit the data based on SEED identifiers.
        # If the location code is specified, the location priority list is not
        # used; the same is true for the channel argument and priority list.
        network="BW", station="A*", location="", channel="BH*",
        # The typical use case for such a data set are noise correlations where
        # gaps are dealt with at a later stage.
        reject_channels_with_gaps=False,
        # Same is true with the minimum length. Any data during a day might be
        # useful.
        minimum_length=0.0,
        # Guard against the same station having different names.
        minimum_interstation_distance_in_m=100.0)

    # Restrict the number of providers if you know which serve the desired
    # data. If in doubt just don't specify - then all providers will be
    # queried.
    dlh = DownloadHelper(providers=["ORFEUS", "GFZ"])
    dlh.download(domain, restrictions, mseed_storage="waveforms",
                 stationxml_storage="stations")

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .download_helpers import DownloadHelper, Restrictions
from .domain import Domain, RectangularDomain, CircularDomain, GlobalDomain
from .utils import format_report
