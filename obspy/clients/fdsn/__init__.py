# -*- coding: utf-8 -*-
"""
obspy.clients.fdsn - FDSN web service client for ObsPy
======================================================
The obspy.clients.fdsn package contains a client to access web servers that
implement the `FDSN web service definitions`_.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


.. contents:: Contents
    :local:
    :depth: 2

Basic FDSN Client Usage
-----------------------

The first step is always to initialize a client object.

>>> from obspy.clients.fdsn import Client
>>> client = Client("IRIS")

A client object can be initialized either with the base URL of any FDSN web
service or with a shortcut name which will be mapped to a FDSN URL. All the
example make use of the FDSN web service at IRIS. For a list of other
available web service providers, see the
:meth:`~obspy.clients.fdsn.client.Client.__init__()` method. The currently
available providers are:

>>> from obspy.clients.fdsn.header import URL_MAPPINGS
>>> for key in sorted(URL_MAPPINGS.keys()):
...     print("{0:<11} {1}".format(key,  URL_MAPPINGS[key]))  # doctest: +SKIP
AUSPASS     http://auspass.edu.au
BGR         http://eida.bgr.de
EIDA        http://eida-federator.ethz.ch
EMSC        http://www.seismicportal.eu
ETH         http://eida.ethz.ch
GEOFON      http://geofon.gfz-potsdam.de
GEONET      http://service.geonet.org.nz
GFZ         http://geofon.gfz-potsdam.de
ICGC        http://ws.icgc.cat
IESDMC      http://batsws.earth.sinica.edu.tw
INGV        http://webservices.ingv.it
IPGP        http://ws.ipgp.fr
IRIS        http://service.iris.edu
IRISPH5     http://service.iris.edu
ISC         http://isc-mirror.iris.washington.edu
KNMI        http://rdsa.knmi.nl
KOERI       http://eida.koeri.boun.edu.tr
LMU         http://erde.geophysik.uni-muenchen.de
NCEDC       http://service.ncedc.org
NIEP        http://eida-sc3.infp.ro
NOA         http://eida.gein.noa.gr
ODC         http://www.orfeus-eu.org
ORFEUS      http://www.orfeus-eu.org
RASPISHAKE  https://fdsnws.raspberryshakedata.com
RESIF       http://ws.resif.fr
RESIFPH5    http://ph5ws.resif.fr
SCEDC       http://service.scedc.caltech.edu
TEXNET      http://rtserve.beg.utexas.edu
UIB-NORSAR  http://eida.geo.uib.no
USGS        http://earthquake.usgs.gov
USP         http://sismo.iag.usp.br

(1) :meth:`~obspy.clients.fdsn.client.Client.get_waveforms()`: The following
    example illustrates how to request and plot 60 minutes of the ``"LHZ"``
    channel of station Albuquerque, New Mexico (``"ANMO"``) of the Global
    Seismograph Network (``"IU"``) for an seismic event around 2010-02-27 06:45
    (UTC). Results are returned as a :class:`~obspy.core.stream.Stream` object.
    See the :meth:`~obspy.clients.fdsn.client.Client.get_waveforms_bulk()`
    method for information on how to send multiple requests simultaneously to
    avoid unnecessary network overhead.

    >>> from obspy import UTCDateTime
    >>> t = UTCDateTime("2010-02-27T06:45:00.000")
    >>> st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
        client = Client()
        t = UTCDateTime("2010-02-27T06:45:00.000")
        st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
        st.plot()

(2) :meth:`~obspy.clients.fdsn.client.Client.get_events()`: Retrieves event
    data from the server. Results are returned as a
    :class:`~obspy.core.event.Catalog` object.

    >>> starttime = UTCDateTime("2002-01-01")
    >>> endtime = UTCDateTime("2002-01-02")
    >>> cat = client.get_events(starttime=starttime, endtime=endtime,
    ...                         minmagnitude=6, catalog="ISC")
    >>> print(cat)  # doctest: +NORMALIZE_WHITESPACE
    2 Event(s) in Catalog:
    2002-01-01T11:29:22.720000Z |  +6.282, +125.749 | 6.3 MW
    2002-01-01T07:28:57.480000Z | +36.991,  +72.336 | 6.3 Mb
    >>> cat.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
        client = Client()
        starttime = UTCDateTime("2002-01-01")
        endtime = UTCDateTime("2002-01-02")
        cat = client.get_events(starttime=starttime, endtime=endtime,
                                minmagnitude=6, catalog="ISC")
        cat.plot()

(3) :meth:`~obspy.clients.fdsn.client.Client.get_stations()`: Retrieves station
    data from the server. Results are returned as an
    :class:`~obspy.core.inventory.inventory.Inventory` object.

    >>> inventory = client.get_stations(network="IU", station="A*",
    ...                                 starttime=starttime,
    ...                                 endtime=endtime)
    >>> print(inventory)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Inventory created at ...
        Created by: IRIS WEB SERVICE: fdsnws-station | version: ...
                    ...
        Sending institution: IRIS-DMC (IRIS-DMC)
        Contains:
                Networks (1):
                        IU
                Stations (3):
                        IU.ADK (Adak, Aleutian Islands, Alaska)
                        IU.AFI (Afiamalu, Samoa)
                        IU.ANMO (Albuquerque, New Mexico, USA)
                Channels (0):
    >>> inventory.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
        client = Client()
        starttime = UTCDateTime("2002-01-01")
        endtime = UTCDateTime("2002-01-02")
        inventory = client.get_stations(network="IU", station="A*",
                                        starttime=starttime,
                                        endtime=endtime)
        inventory.plot()


Basic Routing Clients Usage
---------------------------

Routers are web services that can be queried for which data centers offer
certain pieces of data. That information can then be used to get the actual
data from various data center across the globe. All current routing services
only support the ``dataselect`` and the ``station`` FDSNWS services.

ObsPy has support for two routing services:

(i) The `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_.
(ii) The `EIDAWS Routing Service
     <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_.

To use them, call the
:func:`~obspy.clients.fdsn.routing.routing_client.RoutingClient` function:


>>> from obspy.clients.fdsn import RoutingClient

Get an instance of a routing client using the IRIS Federator:

>>> client = RoutingClient("iris-federator")
>>> print(type(client))  # doctest: +ELLIPSIS
<class '...fdsn.routing.federator_routing_client.FederatorRoutingClient'>

Or get an instance of a routing client using the EIDAWS routing web service:

>>> client = RoutingClient("eida-routing")
>>> print(type(client))  # doctest: +ELLIPSIS
<class '...fdsn.routing.eidaws_routing_client.EIDAWSRoutingClient'>

They can be used like the normal FDSNWS clients, meaning the
``get_(waveforms|stations)(_bulk)()`` functions should work as expected.

To be able to do geographic waveform queries with the EIDA service,
ObsPy will internally perform a station query before downloading the
waveforms. This results in a similar usage between the EIDA and IRIS routing
services from a user's perspective.

The following snippet will call the IRIS federator to figure out who has
waveform data for that particular query and subsequently call the individual
data centers to actually get the data. This happens fully automatically -
please note that the clients also supports non-standard waveform
query parameters like geographical constraints.

>>> from obspy import UTCDateTime
>>> client = RoutingClient("iris-federator")
>>> st = client.get_waveforms(
...     channel="LHZ", starttime=UTCDateTime(2017, 1, 1),
...     endtime=UTCDateTime(2017, 1, 1, 0, 5), latitude=10,
...     longitude=10, maxradius=25)  # doctest: +SKIP
>>> print(st)  # doctest: +SKIP
2 Trace(s) in Stream:
II.MBAR.00.LHZ | 2017-01-01T00:00:00Z - ... | 1.0 Hz, 300 samples
II.MBAR.10.LHZ | 2017-01-01T00:00:00Z - ... | 1.0 Hz, 300 samples

The same works for stations:

>>> client = RoutingClient("iris-federator")
>>> inv = client.get_stations(
...     channel="LHZ", starttime=UTCDateTime(2017, 1, 1),
...     endtime=UTCDateTime(2017, 1, 1, 0, 5), latitude=10,
...     level="channel", longitude=10, maxradius=25)  # doctest: +SKIP
>>> print(inv)  # doctest: +SKIP
Inventory created at 2017-10-09T14:26:28.466161Z
    Created by: ObsPy 1.0.3.post0+1706.gef068de324.dirty
                https://www.obspy.org
    Sending institution: IRIS-DMC,ObsPy ..., SeisComP3 (GFZ,IPGP, IRIS-DMC)
    Contains:
        Networks (6):
            AF, G, II, IU, TT, YY
        Stations (10):
            AF.EKNA (Ekona, Cameroon)
            AF.KIG (Kigali, Rwanda)
            G.TAM (Tamanrasset, Algeria)
            II.MBAR (Mbarara, Uganda)
            IU.KOWA (Kowa, Mali)
            TT.TATN (Station Tataouine, Tunisia)
            YY.GIDA (Health Center, Gidami, Ethiopia)
            YY.GUBA (Police Station, Guba, Ethiopia)
            YY.MEND (High School, Mendi, Ethiopia)
            YY.SHER (Police Station, Sherkole, Ethiopia)
        Channels (11):
            AF.EKNA..LHZ, AF.KIG..LHZ, G.TAM.00.LHZ, II.MBAR.00.LHZ,
            II.MBAR.10.LHZ, IU.KOWA.00.LHZ, TT.TATN.00.LHZ, YY.GIDA..LHZ,
            YY.GUBA..LHZ, YY.MEND..LHZ, YY.SHER..LHZ

Please see the documentation for each method for further information and
examples.

.. _FDSN web service definitions: https://www.fdsn.org/webservices/
"""
from .client import Client  # NOQA
from .routing.routing_client import RoutingClient  # NOQA
from .header import URL_MAPPINGS  # NOQA


# insert supported URL mapping list dynamically in docstring
# we need an if clause because add_doctests() executes the file once again
if Client.__init__.__doc__ is not None and r"%s" in Client.__init__.__doc__:
    Client.__init__.__doc__ = \
        Client.__init__.__doc__ % \
        str(sorted(URL_MAPPINGS.keys())).strip("[]")

__all__ = ["Client", "RoutingClient"]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
