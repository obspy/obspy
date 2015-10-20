#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mass Downloader for FDSN Compliant Web Services
===============================================

This package contains functionality to query and integrate data from any number
of `FDSN web service <http://www.fdsn.org/webservices/>`_ providers
simultaneously. The package aims to formulate download requests in a way that
is convenient for seismologists without having to worry about political and
technical data center issues. It can be used by itself or as a library
component integrated into a bigger project.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


Why Would You Want to Use This?
-------------------------------

Directly using the FDSN web services for example via the
:mod:`obspy.clients.fdsn` client is fine for small amounts of data but quickly
becomes cumbersome for larger data sets. Many data centers do provide tools to
easily download larger amounts of data but that is usually only from one data
center. Now most seismologists don't really care a lot where the data they
download originates - they just want the data for their use case and
oftentimes they want as much data as they can get. As the number of FDSN
compliant web services increases this becomes more and more cumbersome.  That
is where this module comes in. You

1. specify the **geographical region** from which to download data,
2. define a number of **other restrictions** (temporal, data quality, ...),
3. and launch the download.


The mass downloader module will acquire all waveforms and associated station
information across all known FDSN web service implementations producing a
**clean data set** ready for further use.  It works by

1. figuring out what stations each provider offers,
2. downloading MiniSEED and associated StationXML meta information in an
   efficient and data center friendly manner, and
3. dealing with all the nasty real-world data issues like missing or incomplete
   data, duplicate data across data centers, e.g.

   * Basic optional automatic quality control by assuring that the data has
     no-gaps/overlaps or is available for a certain percentage of the requested
     time span.
   * It can relaunch download to acquire missing pieces which might happen for
     example if a data center has been offline.
   * It can assure that there always is a corresponding StationXML file for the
     waveforms.

Usage Examples
--------------

Before delving into the nitty-gritty details of how it works and why it does
things in a certain way we'll demonstrate the usage of this module on two
annotated examples. They can serve as templates for your own needs.

Earthquake Data
~~~~~~~~~~~~~~~

The classic seismological data set consists of waveform recordings for a
certain earthquake. This example downloads all data it can find for the
Tohoku-Oki Earthquake from 5 minutes before the earthquake centroid time to 1
hour after.  It will furthermore only download data with an epicentral distance
between 70.0 and 90.0 degrees and some additional restrictions. Be aware that
this example will attempt to download data from all FDSN data centers that
ObsPy knows of and combine it into one data set.

.. code-block:: python

    import obspy
    from obspy.clients.fdsn.mass_downloader import CircularDomain, \\
        Restrictions, MassDownloader

    origin_time = obspy.UTCDateTime(2011, 3, 11, 5, 47, 32)

    # Circular domain around the epicenter. This will download all data between
    # 70 and 90 degrees distance from the epicenter. This module also offers
    # rectangular and global domains. More complex domains can be defined by
    # inheriting from the Domain class.
    domain = CircularDomain(latitude=37.52, longitude=143.04,
                            minradius=70.0, maxradius=90.0)

    restrictions = Restrictions(
        # Get data from 5 minutes before the event to one hour after the
        # event. This defines the temporal bounds of the waveform data.
        starttime=origin_time - 5 * 60,
        endtime=origin_time + 3600,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=True,
        # And you might only want waveforms that have data for at least 95 % of
        # the requested time span. Any trace that is shorter than 95 % of the
        # desired total duration will be discarded.
        minimum_length=0.95,
        # No two stations should be closer than 10 km to each other. This is
        # useful to for example filter out stations that are part of different
        # networks but at the same physical station. Settings this option to
        # zero or None will disable that filtering.
        minimum_interstation_distance_in_m=10E3,
        # Only HH or BH channels. If a station has HH channels, those will be
        # downloaded, otherwise the BH. Nothing will be downloaded if it has
        # neither. You can add more/less patterns if you like.
        channel_priorities=("HH[ZNE]", "BH[ZNE]"),
        # Location codes are arbitrary and there is no rule as to which
        # location is best. Same logic as for the previous setting.
        location_priorities=("", "00", "10"))

    # No specified providers will result in all known ones being queried.
    mdl = MassDownloader()
    # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
    # folders with automatically chosen file names.
    mdl.download(domain, restrictions, mseed_storage="waveforms",
                 stationxml_storage="stations")


Continuous Request
~~~~~~~~~~~~~~~~~~

Another use case requiring massive amounts of data are noise studies. Ambient
seismic noise correlations require continuous recordings from stations over a
large time span. This example downloads data, from within a certain
geographical domain, for a whole year. Individual MiniSEED files will be split
per day. The download helpers will attempt to optimize the queries to the data
centers and split up the files again if required.

.. code-block:: python

    import obspy
    from obspy.clients.fdsn.mass_downloader import RectangularDomain, \\
        Restrictions, MassDownloader

    # Rectangular domain containing parts of southern Germany.
    domain = RectangularDomain(minlatitude=30, maxlatitude=50,
                               minlongitude=5, maxlongitude=35)

    restrictions = Restrictions(
        # Get data for a whole year.
        starttime=obspy.UTCDateTime(2012, 1, 1),
        endtime=obspy.UTCDateTime(2013, 1, 1),
        # Chunk it to have one file per day.
        chunklength_in_sec=86400,
        # Considering the enormous amount of data associated with continuous
        # requests, you might want to limit the data based on SEED identifiers.
        # If the location code is specified, the location priority list is not
        # used; the same is true for the channel argument and priority list.
        network="BW", station="A*", location="", channel="EH*",
        # The typical use case for such a data set are noise correlations where
        # gaps are dealt with at a later stage.
        reject_channels_with_gaps=False,
        # Same is true with the minimum length. All data might be useful.
        minimum_length=0.0,
        # Guard against the same station having different names.
        minimum_interstation_distance_in_m=100.0)

    # Restrict the number of providers if you know which serve the desired
    # data. If in doubt just don't specify - then all providers will be
    # queried.
    mdl = MassDownloader(providers=["LMU", "GFZ"])
    mdl.download(domain, restrictions, mseed_storage="waveforms",
                 stationxml_storage="stations")


Usage
-----

Using the download helpers requires the definition of three separate things,
all of which are detailed in the following paragraphs.

1. **Data Selection:** The data to be downloaded can be defined by enforcing
   geographical or temporal constraints and a couple of other options.
2. **Storage Options:** Choosing where the final MiniSEED and StationXML files
   should be stored.
3. **Start the Download:** Choose from which provider(s) to download and then
   launch the downloading process.


Step 1: Data Selection
~~~~~~~~~~~~~~~~~~~~~~

Data set selection serves the purpose to limit the data to be downloaded to
data useful for the purpose at hand. It is handled by two objects:
subclasses of the  :class:`~obspy.clients.fdsn.mass_downloader.domain.Domain`
object and the
:class:`~obspy.clients.fdsn.mass_downloader.restrictions.Restrictions` class.

The :class:`~obspy.clients.fdsn.mass_downloader.domain` module currently
defines three different domain types used to limit the geographical extent of
the queried data:
:class:`~obspy.clients.fdsn.mass_downloader.domain.RectangularDomain`,
:class:`~obspy.clients.fdsn.mass_downloader.domain.CircularDomain`, and
:class:`~obspy.clients.fdsn.mass_downloader.domain.GlobalDomain`. Subclassing
:class:`~obspy.clients.fdsn.mass_downloader.domain.Domain` enables the
construction of arbitrarily complex domains. Please see the
:class:`~obspy.clients.fdsn.mass_downloader.domain` module for more details.
Instances of these classes will later be passed to the function sparking the
downloading process. A rectangular domain for example is defined like this:

>>> from obspy.clients.fdsn.mass_downloader.domain import RectangularDomain
>>> domain = RectangularDomain(minlatitude=-10, maxlatitude=10,
...                            minlongitude=-10, maxlongitude=10)

Additional restrictions like temporal bounds, SEED identifier wildcards,
and other things are set with the help of
the :class:`~obspy.clients.fdsn.mass_downloader.restrictions.Restrictions`
class. Please refer to its documentation for a more detailed explanation of
the parameters.

>>> from obspy import UTCDateTime
>>> from obspy.clients.fdsn.mass_downloader import Restrictions
>>> restrict = Restrictions(
...     starttime=UTCDateTime(2012, 1, 1),
...     endtime=UTCDateTime(2012, 1, 1, 1),
...     network=None, station=None, location=None, channel=None,
...     reject_channels_with_gaps=True,
...     minimum_length=0.9,
...     minimum_interstation_distance_in_m=1000,
...     channel_priorities=["HH[ZNE]", "BH[ZNE]"],
...     location_priorities=["", "00", "01"])


Step 2: Storage Options
~~~~~~~~~~~~~~~~~~~~~~~

After determining what to download, the helpers must know where to store the
requested data. That requires some flexibility in case the mass downloader
is to be integrated as a component into a bigger system. An example is
a toolbox that has a database to manage its data.

A major concern is to not download pre-existing data. In order to enable such
a use case the download helpers can be given functions that are evaluated when
determining the file names of the requested data. Depending on the return
value, the helper class will download the whole, part, or even none, of that
particular piece of data.

Storing MiniSEED waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^

The MiniSEED storage rules are set by the ``mseed_storage`` argument of the
:meth:`~obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download`
method of the
:class:`~obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader`
class


**Option 1: Folder Name**

In the simplest case it is just a folder name:

>>> mseed_storage = "waveforms"

This will cause all MiniSEED files to be stored as

``waveforms/NETWORK.STATION.LOCATION.CHANNEL__STARTTIME__ENDTIME.mseed``.

An example of this is

``waveforms/BW.FURT..BHZ__20141027T163723Z__20141027T163733Z.mseed``

which is rather general but also quite long.

**Option 2: String Template**

For more control use the second possibility and provide a string containing
``{network}``, ``{station}``, ``{location}``, ``{channel}``, ``{starttime}``,
and ``{endtime}`` format specifiers. These values will be interpolated to
acquire the final filename. The start and end times will be formatted with
``strftime()`` with the specifier ``"%Y%m%dT%H%M%SZ"`` in an effort to
avoid colons which are troublesome in file names on many systems.

>>> mseed_storage = ("some_folder/{network}/{station}/"
...                  "{location}.{channel}.{starttime}.{endtime}.mseed")

results in

``some_folder/BW/FURT/.BHZ.20141027T163723Z.20141027T163733Z.mseed``.

The download helpers will create any non-existing folders along the path.

**Option 3: Custom Function**

The most complex but also most powerful possibility is to use a function which
will be evaluated to determine the filename. **If the function returns**
``True`` **, the MiniSEED file is assumed to already be available and will not
be downloaded again; keep in mind that in that case no station data will be
downloaded for that channel.** If it returns a string, the MiniSEED file will
be saved to that path. Utilize closures to use any other parameters in the
function. This hypothetical function checks if the file is already in a
database and otherwise returns a string which will be interpreted as a
filename.

>>> def get_mseed_storage(network, station, location, channel, starttime,
...                       endtime):
...     # Returning True means that neither the data nor the StationXML file
...     # will be downloaded.
...     if is_in_db(network, station, location, channel, starttime, endtime):
...         return True
...     # If a string is returned the file will be saved in that location.
...     return os.path.join(ROOT, "%s.%s.%s.%s.mseed" % (network, station,
...                                                      location, channel))
>>> mseed_storage = get_mseed_storage

.. note::

    No matter which approach is chosen, if a file already exists, it will not
    be overwritten; it will be parsed and the download helper class will
    attempt to download matching StationXML files.


Storing StationXML files
^^^^^^^^^^^^^^^^^^^^^^^^

The same logic applies to the StationXML files. This time the rules are set by
the ``stationxml_storage`` argument of the
:func:`~obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download`
method of the
:class:`~obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader`
class. StationXML files will be downloaded on a per-station basis thus all
channels and locations from one station will end up in the same StationXML
file.

**Option 1: Folder Name**

A simple string will be interpreted as a folder name. This example will save
the files to ``"stations/NETWORK.STATION.xml"``, e.g. to
``"stations/BW.FURT.xml"``.

>>> stationxml_storage = "stations"

**Option 2: String Template**

Another option is to provide a string formatting template, e.g.

>>> stationxml_storage = "some_folder/{network}/{station}.xml"

will write to ``"some_folder/NETWORK/STATION.xml"``, in this case for example
to ``"some_folder/BW/FURT.xml"``.

.. note::

    If the StationXML file already exists, it will be opened to see what is in
    the file. In case it does not contain all necessary channels, it will be
    deleted and **only those channels needed in the current run will be
    downloaded again**. Pass a custom function to the ``stationxml_path``
    argument if you require different behavior as documented in the
    following section.

**Option 3: Custom Function**

As with the waveform data, the StationXML paths can also be set with the help
of a function. The function in this case is a bit more complex than for the
waveform case. It has to return a dictionary with three keys:
``"available_channels"``, ``"missing_channels"``, and ``"filename"``.
``"available_channels"`` is a list of channels that are already available as
station information and that require no new download. Make sure to include all
already available channels; this information is later used to discard
MiniSEED files that have no corresponding station information.
``"missing_channels"`` is a list of channels for that particular station that
must be downloaded and ``"filename"`` determines where to save these. Please
note that in this particular case the StationXML file will be overwritten if it
already exists and only the ``"missing_channels"`` will be downloaded to it,
independent of what already exists in the file.

Alternatively the function can also return a string and the behaviour is the
same as two first options for the ``stationxml_storage`` argument.

The next example illustrates a complex use case where the availability of each
channel's station information is queried in some database and only those
channels that do not exist yet will be downloaded. Use closures to pass more
arguments to the function.

>>> def get_stationxml_storage(network, station, channels, starttime, endtime):
...     available_channels = []
...     missing_channels = []
...     for location, channel in channels:
...         if is_in_db(network, station, location, channel, starttime,
...                     endtime):
...             available_channels.append((location, channel))
...         else:
...             missing_channels.append((location, channel))
...     filename = os.path.join(ROOT, "%s.%s.xml" % (network, station))
...     return {
...         "available_channels": available_channels,
...         "missing_channels": missing_channels,
...         "filename": filename}
>>> stationxml_storage = get_stationxml_storage

Step 3: Start the Download
~~~~~~~~~~~~~~~~~~~~~~~~~~

The final step is to actually start the download. Pass the previously created
domain, restrictions, and path settings and off you go. Two more parameters of
interest are the ``chunk_size_in_mb`` setting which controls how much data is
requested per thread, client and request. ``threads_per_clients`` control how
many threads are used to download data in parallel per data center - 3 is a
value in agreement with some data centers.


>>> mdl = MassDownloader()  # doctest: +SKIP
>>> mdl.download(domain, restrictions, chunk_size_in_mb=50,
...              threads_per_client=3, mseed_storage=mseed_storage,
...              stationxml_storage=stationxml_storage)  # doctest: +SKIP


How it Works
------------

At a high level the mass downloader works by looping over each FDSN web service
and downloading whatever it offers. A bit more detail:

1. Loop over all passed or known FDSN web service implementations and
   auto-discover if they are available and what they can do. If an
   implementation has a ``dataselect`` and a ``station`` service it will be
   part of the following steps. Otherwise it will be discarded.

2. For each web service client:

   a) Request the availability for the given time and domain settings. It will
      request a text file from the ``station`` service at the channel level. If
      the service supports the ``matchtimeseries`` parameter it will be used
      and the availability is considered to be *"reliable"* for the further
      stages.

   b) Channel and location priorities are applied resulting in a single
      instrument per station.

   c) Any already existing network + station combinations are discarded.

   d) If the availability for the particular client is considered reliable it
      will perform the minimum distance filtering now. If no stations have
      already been downloaded it will select the largest subset of stations
      satisfying the minimum interstation distance constraint. Otherwise it
      will successively add new stations with the largest distance to the
      closest already existing station until no more stations satisfying the
      minimum distance remain. This results in the maximum possible amount of
      chosen stations satisfying the constraints.

   e) Download the MiniSEED data - this is threaded and it will use a bulk
      request honoring the desired ``chunk_size_in_mb`` setting. Afterwards it
      splits the MiniSEED files again to match the desired restrictions. The
      split happens at the record level thus no information available in the
      original MiniSEED records is lost.

   f) Any MiniSEED files not fulfilling the minimum length or no/gap overlap
      restrictions will be deleted. Faulty MiniSEED files as well.

   g) For each downloaded MiniSEED file: Download the corresponding StationXML
      file at the response level.

   h) If the ``sanitize`` argument of the Restrictions object is ``True``,
      delete all MiniSEED files for which no station information could be
      downloaded. This is a useful setting if you want a clean data set.

   g) If the availability information is not reliable, perform the minimum
      interstation distance filtering now. This is a bit unfortunate but many
      client do return pretty terrible availability information (or interpret
      the ``station`` service differently) so there is no way around that for
      now.

   h) Rinse and repeat for all remaining FDSN web service implementations.


Logging
~~~~~~~

The download helpers utilizes Python's `logging facilities
<https://docs.python.org/2/library/logging.html>`__. By default it will log to
stdout at the ``logging.INFO`` level which provides a fair amount of detail. If
you want to change the log level or setup a different stream handler, just get
the corresponding logger after you import the download helpers module:


>>> import logging
>>> logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
>>> logger.setLevel(logging.DEBUG)  # doctest: +SKIP


Further Documentation
~~~~~~~~~~~~~~~~~~~~~

Further functionality of this module is documented at a couple of other places:

* :mod:`~.domain` module
* :class:`~.restrictions.Restrictions` class
* :class:`~.mass_downloader.MassDownloader` class
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

# Convenience imports.
from .mass_downloader import MassDownloader  # NOQA
from .restrictions import Restrictions  # NOQA
from .domain import (Domain, RectangularDomain,  # NOQA
                     CircularDomain, GlobalDomain)  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
