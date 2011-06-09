# -*- coding: utf-8 -*-
"""
obspy.arclink - ArcLink/WebDC request client for of ObsPy
=========================================================

ArcLink is a distributed data request protocol usable to access archived
waveform data in the MiniSEED or SEED format and associated meta information as
Dataless SEED files. It has been originally founded within the German WebDC
initiative of GEOFON_ (Geoforschungsnetz) and BGR_ (Bundesanstalt für
Geowissenschaften und Rohstoffe). ArcLink has been designed as a "straight
consequent continuation" of the NetDC concept originally developed by the IRIS_
DMC. Instead of requiring waveform data via E-mail or FTP requests, ArcLink
offers a direct TCP/IP communication approach. A prototypic web-based request
tool is available via the WebDC homepage at http://www.webdc.eu.

Recent development efforts within the NERIES_ (Network of Excellence of
Research and Infrastructures for European Seismology) project focuses on
extending the ArcLink network to all major seismic data centers within Europe
in order to create an European Integrated Data Center (EIDAC). Currently
(September 2009) there are four European data centers contributing to this
network: ORFEUS_, GFZ_ (GeoForschungsZentrum), INGV_ (Istituto Nazionale di
Geofisica e Vulcanologia), and IPGP_ (Institut de Physique du Globe de Paris).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------
The example illustrates how to request and plot 30 minutes of all three
broadband channels ("BH*") of station Fürstenfeldbruck ("FUR") of the German
Regional network ("GR") for an seismic event around 2009-08-20 06:35:00 (UTC).

.. note:: 
    The client needs to open port 18001 to the host webdc.eu via TCP/IP in
    order to download the requested data. Please make sure that no firewall is
    blocking access to this server/port combination.

>>> from obspy.core import UTCDateTime
>>> from obspy.arclink.client import Client
>>> client = Client("webdc.eu", 18001)
>>> t = UTCDateTime("2009-08-20 04:03:12")
>>> st = client.getWaveform("BW", "RJOB", "", "EH*", t - 3, t + 15)
>>> st.plot() #doctest: +SKIP 

.. plot::

    from obspy.core import UTCDateTime
    from obspy.arclink.client import Client
    client = Client("webdc.eu", 18001)
    t = UTCDateTime("2009-08-20 04:03:12")
    st = client.getWaveform("BW", "RJOB", "", "EH*", t - 3, t + 15)
    st.plot() #doctest: +SKIP

Waveform data fetched from an ArcLink node is converted into an ObsPy stream
object. The seismogram is truncated by the ObsPy client to the actual requested
time span, as ArcLink internally cuts SEED files for performance reasons on
record base in order to avoid uncompressing the waveform data. The output of
the script above is shown in the next picture.

Video Tutorial
--------------
http://www.obspy.org/www/obspy.arclink/arclink.html

Further Examples
----------------
The following methods are demonstrated using the initialized client from the
example above.

(1) :meth:`~obspy.arclink.client.Client.getPAZ()`: Requests poles, zeros, gain
    and sensitivity of a single channel for a certain time span.

    >>> t = UTCDateTime(2009, 1, 1)
    >>> paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', t, t + 1)
    >>> paz  #doctest: +NORMALIZE_WHITESPACE +SKIP
    {'STS-2/N/g=1500': {
        'poles': [(-0.037004000000000002+0.037016j), 
                  (-0.037004000000000002-0.037016j), 
                  (-251.33000000000001+0j), 
                  (-131.03999999999999-467.29000000000002j), 
                  (-131.03999999999999+467.29000000000002j)], 
        'sensitivity': 2516778600.0, 
        'zeros': [0j, 0j], 
        'gain': 60077000.0}}

(2) :meth:`~obspy.arclink.client.Client.saveResponse()`: Writes a response
    information into a file.

    >>> t = UTCDateTime(2009, 1, 1)
    >>> client.saveResponse('BW.MANZ..EHZ.dataless', 'BW', 'MANZ', '', '*',
    ...                     t, t + 1, format="SEED") #doctest: +SKIP

(3) :meth:`~obspy.arclink.client.Client.saveWaveform()`: Writes a seismogram
    into a Full SEED volume.

    >>> t = UTCDateTime(2009, 1, 1, 12, 0)
    >>> client.saveWaveform('BW.MANZ..EHZ.seed', 'BW', 'MANZ', '', '*',
    ...                     t, t + 20, format='FSEED') #doctest: +SKIP

(4) :meth:`~obspy.arclink.client.Client.getInventory()`: Request inventory data.

    >>> inv = client.getInventory('BW', 'M*', '*', 'EHZ', restricted=False,
    ...                           permanent=True, min_longitude=12, 
    ...                           max_longitude=12.2) #doctest: +SKIP
    >>> inv.keys() #doctest: +SKIP
    ['BW.MROB', 'BW.MANZ..EHZ', 'BW', 'BW.MANZ', 'BW.MROB..EHZ']
    >>> inv['BW'] #doctest: +SKIP
    AttribDict({'description': 'BayernNetz', 'region': 'Germany', 'archive': ...
    >>> inv['BW.MROB'] #doctest: +SKIP
    AttribDict({'code': 'MROB', 'description': 'Rosenbuehl, Bavaria', 'start'...

Further Resources
-----------------
* ArcLink protocol specifications:
    - http://www.seiscomp3.org/wiki/doc/applications/arclink
    - http://geofon.gfz-potsdam.de/_sc3_neries_/arclink.pdf
    - https://svn.obspy.org/trunk/obspy.arclink/docs/other/protocol.txt
* Short introduction_ to the ArcLink protocol
* Latest `ArcLink server`_ package
* SeismoLink_: a SOAP Web service on top of the ArcLink network

.. _GEOFON:
        http://geofon.gfz-potsdam.de
.. _BGR:
        http://www.bgr.de
.. _IRIS:
        http://www.iris.edu
.. _NERIES:
        http://www.neries-eu.org
.. _INGV:
        http://www.ingv.it
.. _ORFEUS:
        http://www.orfeus-eu.org
.. _GFZ:
        http://www.gfz-potsdam.de
.. _IPGP:
        http://www.ipgp.fr
.. _introduction:
        http://www.webdc.eu/webdc_sum.html
.. _`ArcLink server`:
        ftp://ftp.gfz-potsdam.de/pub/home/st/GEOFON/software/SeisComP/ArcLink/
.. _SeismoLink:
        http://neriesdataportalblog.freeflux.net/webservices/
.. _`GNU Lesser General Public License, Version 3`:
        http://www.gnu.org/copyleft/lesser.html
"""

from obspy.core.util import _getVersionString
from client import Client


__version__ = _getVersionString("obspy.arclink")


if __name__ == '__main__': # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
