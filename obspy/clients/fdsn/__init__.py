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
...     print("{0:<7} {1}".format(key,  URL_MAPPINGS[key]))
BGR     http://eida.bgr.de
EMSC    http://www.seismicportal.eu
ETH     http://eida.ethz.ch
GEONET  http://service.geonet.org.nz
GFZ     http://geofon.gfz-potsdam.de
ICGC    http://ws.icgc.cat
INGV    http://webservices.rm.ingv.it
IPGP    http://eida.ipgp.fr
IRIS    http://service.iris.edu
ISC     http://isc-mirror.iris.washington.edu
KOERI   http://eida.koeri.boun.edu.tr
LMU     http://erde.geophysik.uni-muenchen.de
NCEDC   http://service.ncedc.org
NIEP    http://eida-sc3.infp.ro
NOA     http://eida.gein.noa.gr
ODC     http://www.orfeus-eu.org
ORFEUS  http://www.orfeus-eu.org
RESIF   http://ws.resif.fr
SCEDC   http://service.scedc.caltech.edu
TEXNET  http://rtserve.beg.utexas.edu
USGS    http://earthquake.usgs.gov
USP     http://sismo.iag.usp.br

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


Basic FDSN FedCatalog Client Usage
----------------------------------

The
:mod:`FDSN fedcatalog_client <obspy.clients.fdsn.routers.fedcatalog_client>`
module provides federated
access to multiple web servers that implement the
`FDSN Station and Dataselect web service definitions
<https://www.fdsn.org/webservices/>`_.

The first step is always to initialize a
:class:`~obspy.clients.fdsn.FederatedClient` object.

>>> from obspy.clients.fdsn import FederatedClient
>>> client = FederatedClient()

(1) :meth:`~obspy.clients.fdsn.routers.fedcatalog_client.
FederatedClient.get_waveforms()`: The following
example illustrates how to request 60 minutes of the ``"LHZ"`` channel of
station Apirathos, Naxos, Greece (``"APE"``) of the GEOFON (``"GE"``) for a
seismic event around 2006-01-08T11:34:54.000 (UTC). Results are returned as a
:class:`~obspy.core.stream.Stream` object.

    >>> from obspy import UTCDateTime
    >>> t = UTCDateTime("2006-01-08T11:34:54.000")
    >>> st = client.get_waveforms("GE", "APE", "", "LHZ", t, t + 60 * 60)
    >>> st.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
        client = Client('GFZ')
        t = UTCDateTime("2006-01-08T11:34:54.000")
        st = client.get_waveforms("GE", "APE", "", "LHZ", t, t + 60 * 60)
        st.plot()

(2) :meth:`~obspy.clients.fdsn.routers.fedcatalog_client.
FederatedClient.get_stations()`: Uses the IRIS Fed Catalog web service to
    return station metadata as an
    :class:`~obspy.core.inventory.inventory.Inventory` object.

    >>> inventory = client.get_stations(network="GE", station="A*",
    ...                                 channel="?HZ", level="station",
    ...                                 endtime="2016-12-31")
    >>> print(inventory)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Inventory created at 2...Z
        Sending institution: SeisComP3 (GFZ)
        Contains:
            Networks (1):
                GE
            Stations (4):
                GE.APE (GEOFON Station Apirathos, Naxos)
                GE.APE (NOA/GEOFON Station Apeiranthos,Naxos, Greece)
                GE.APEZ (GEOFON Station Moni Apezanon, Greece)
                GE.ARPR (GEOFON/MedNet/KOERI Station Arapgir, Turkey)
            Channels (0):
    <BLANKLINE>

Please see the documentation for each method for further information and
examples.

.. _FDSN web service definitions: https://www.fdsn.org/webservices/
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str

from .client import Client  # NOQA
from .header import URL_MAPPINGS  # NOQA
from .routers import FederatedClient  # NOQA


# insert supported URL mapping list dynamically in docstring
# we need an if clause because add_doctests() executes the file once again
if r"%s" in Client.__init__.__doc__:
    if PY2:
        Client.__init__.__func__.__doc__ = \
            Client.__init__.__doc__ % \
            str(sorted(URL_MAPPINGS.keys())).strip("[]")
    else:
        Client.__init__.__doc__ = \
            Client.__init__.__doc__ % \
            str(sorted(URL_MAPPINGS.keys())).strip("[]")

__all__ = [native_str("Client")]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
