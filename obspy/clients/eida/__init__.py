# -*- coding: utf-8 -*-
"""
obspy.clients.eida - FDSN/EIDA Web service client for ObsPy
===========================================================
The obspy.clients.eida package contains a client to access web servers that
implement the FDSN web services using EIDA routing and authentication
extensions (http://www.orfeus-eu.org/eida/eida-webservices.html).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Helmholtz-Zentrum Potsdam - Deutsches GeoForschungsZentrum GFZ
    (geofon@gfz-potsdam.de)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

The first step is always to initialize a client object.

>>> from obspy.clients.eida import Client
>>> client = Client("GFZ")

A client object can be initialized either with the base URL of any EIDA routing
service or with a :mod:`shortcut name <obspy.clients.fdsn>` that will be mapped
to such a URL (only works if the node is running the EIDA routing service). The
examples use ``"GFZ"``, which is the default.

Regardless of which EIDA routing service is used, requests will be routed to
all EIDA nodes. However, there may be special routing services that route
requests to a different set of nodes, even globally.

To access restricted data, add a token obtained from an EIDA authentication
service:

>>> from obspy.clients.eida import Client
>>> authdata = open("token.asc").read()  # doctest: +SKIP
>>> client = Client(authdata=authdata)  # doctest: +SKIP

For backwards compatibility, username/password authentication is supported as
well. Different credentials can be specified for each node and they take
precedence over token authentication at that node.

Add debug=True to see what is going on.

>>> from obspy.clients.eida import Client
>>> credentials = {"http://service.iris.edu/fdsnws/dataselect/1/queryauth":
...                ("nobody@iris.edu", "anonymous")}
>>> authdata = open("token.asc").read()  # doctest: +SKIP
>>> client = Client(credentials=credentials,
...                 authdata=authdata,
...                 debug=True)  # doctest: +SKIP

(1) :meth:`~obspy.clients.eida.client.Client.get_waveforms()`: The following
    example illustrates how to request and plot 60 minutes of the ``"LHZ"``
    channel of EIDA stations starting with ``"A"`` for a seismic event around
    2010-02-27 07:00 (UTC). Results are returned as a
    :class:`~obspy.core.stream.Stream` object.
    See the :meth:`~obspy.clients.eida.client.Client.get_waveforms_bulk()`
    method for information on how to send multiple requests simultaneously to
    avoid unnecessary network overhead.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.eida import Client
    >>> client = Client()
    >>> t = UTCDateTime("2010-02-27T07:00:00.000")
    >>> st = client.get_waveforms("*", "A*", "", "LHZ", t, t+60*60)
    >>> st.plot(starttime=t, endtime=t+60*60)  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.eida import Client
        client = Client()
        t = UTCDateTime("2010-02-27T07:00:00.000")
        st = client.get_waveforms("*", "A*", "", "LHZ", t, t+60*60)
        st.plot(starttime=t, endtime=t+60*60)

(2) :meth:`~obspy.clients.eida.client.Client.get_events()`: Retrieves event
    data from the server. Results are returned as a
    :class:`~obspy.core.event.Catalog` object.

    The event service is not routed, so a node that is running an event service
    locally must be used.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.eida import Client
    >>> client = Client("INGV")
    >>> starttime = UTCDateTime("2002-01-01")
    >>> endtime = UTCDateTime("2002-01-02")
    >>> cat = client.get_events(starttime=starttime, endtime=endtime)
    >>> print(cat)  # doctest: +NORMALIZE_WHITESPACE
    4 Event(s) in Catalog:
    2002-01-01T19:42:34.670000Z | +43.434,  +12.525 | 1.9 Md | manual
    2002-01-01T10:54:09.040000Z | +43.439,  +12.476 | 2.2 Md | manual
    2002-01-01T08:21:30.790000Z | +46.071,  +10.630 | 2.3 Md | manual
    2002-01-01T05:07:31.250000Z | +44.781,   +8.361 | 2.7 Md | manual
    >>> cat.plot(projection="local")  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.eida import Client
        client = Client("INGV")
        starttime = UTCDateTime("2002-01-01")
        endtime = UTCDateTime("2002-01-02")
        cat = client.get_events(starttime=starttime, endtime=endtime)
        cat.plot(projection="local")

(3) :meth:`~obspy.clients.eida.client.Client.get_stations()`: Retrieves station
    data from a set of nodes known to the routing service. Results are returned
    as an :class:`~obspy.core.inventory.inventory.Inventory` object.

    Since the information comes from multiple nodes, "Created by" and
    "Sending institution" cannot be relied on.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.eida import Client
    >>> client = Client()
    >>> starttime = UTCDateTime("2002-01-01")
    >>> endtime = UTCDateTime("2002-01-02")
    >>> inventory = client.get_stations(network="*", station="A*",
    ...                                 starttime=starttime,
    ...                                 endtime=endtime)
    >>> inventory.networks = sorted(inventory.networks, key=lambda x: x.code)
    >>> print(inventory)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Inventory created at ...
        Created by: ...
        Sending institution: ...
        Contains:
                Networks (16):
                        BW
                        CH
                        CL
                        DK
                        FR
                        G
                        GE
                        II
                        IU
                        JS
                        MN
                        NO
                        OE
                        YF
                        YR
                        ZC
                Stations (25):
                        BW.ALTM (Beilngries, Bavaria, BW-Net)
                        CH.ACB (Klingnau, Acheberg, AG)
                        CH.AIGLE (Aigle, VD)
                        CL.AGEO (Agios Giorgios, Aigialia, West Greece, Greece)
                        CL.AIOA (Agios Ioannis, Aigialia, West Greece, Greece)
                        DK.ANGG (Station Ammassalik, Greenland)
                        FR.ARBF (technopole de l'Arbois - 13001, Aix-en-Prov...
                        G.AIS (Nouvelle-Amsterdam - TAAF, France)
                        G.ATD (Arta Cave - Arta, Republic of Djibouti)
                        GE.APE (GEOFON Station Apirathos, Naxos)
                        GE.APEZ (GEOFON Station Moni Apezanon, Greece)
                        II.AAK (Ala Archa, Kyrgyzstan)
                        II.ARU (Arti, Russia)
                        IU.ANTO (Ankara, Turkey)
                        JS.AQBJ (Station JS station, Jordan)
                        JS.ASF (Station Asfer, Jordan)
                        MN.AIO (Antillo, Italy)
                        MN.AQU (L'Aquila, Italy)
                        NO.ARE0 (ARE0)
                        OE.ARSA (ARZBERG, AUSTRIA)
                        YF.ABSA (DJEBEL ABABSIA, ALGERIA)
                        YR.ALE (Alemeya, Ethiopie)
                        ZC.ACBG (Station ACBG, Spain)
                        ZC.ACLR (Station ACLR, Spain)
                        ZC.ALB (Station ALB, Spain)
                Channels (0):
    >>> inventory.plot()  # doctest: +SKIP

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.eida import Client
        client = Client()
        starttime = UTCDateTime("2002-01-01")
        endtime = UTCDateTime("2002-01-02")
        inventory = client.get_stations(network="*", station="A*",
                                        starttime=starttime,
                                        endtime=endtime)
        inventory.plot()

Please see the documentation for each method for further information and
examples.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from .client import Client  # NOQA
from obspy.clients.fdsn.header import URL_MAPPINGS  # NOQA


__all__ = [native_str("Client")]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
