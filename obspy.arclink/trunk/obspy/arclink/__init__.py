# -*- coding: utf-8 -*-
"""
obspy.arclink: ArcLink/WebDC request client for of ObsPy
========================================================
ArcLink is a distributed data request protocol usable to access archived
waveform data in the MiniSEED or SEED format and associated meta information as
Dataless SEED files. It has been originally founded within the German WebDC
initiative of GEOFON (Geoforschungsnetz) and BGR (Bundesanstalt für
Geowissenschaften und Rohstoffe). ArcLink has been designed as a "straight
consequent continuation" of the NetDC concept originally developed by the IRIS
DMC. Instead of requiring waveform data via E-mail or FTP requests, ArcLink
offers a direct TCP/IP communication approach. A prototypic web-based request
tool is available via the WebDC homepage at http://www.webdc.eu.

Recent development efforts within the NERIES (Network of Excellence of Research
and Infrastructures for European Seismology) project focuses on extending the
ArcLink network to all major seismic data centers within Europe in order to
create an European Integrated Data Center (EIDAC). Currently (September 2009)
there are four European data centers contributing to this network: ORFEUS, GFZ
(GeoForschungsZentrum), INGV (Istituto Nazionale di Geofisica e Vulcanologia),
and IPGP (Institut de Physique du Globe de Paris).

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)

Basic Usage
-----------
The example illustrates how to request and plot 30 minutes of all three
broadband channels ("BH*") of station Fürstenfeldbruck ("FUR") of the German
Regional network ("GR") for an seismic event around 2009-08-20 06:35:00 (UTC).

:note: The client needs to open port 18001 to the server webdc.eu via TCP/IP in
       order to download the requested data. Please make sure that no firewall
       is blocking access to this server/port combination.

>>> from obspy.core import UTCDateTime
>>> from obspy.arclink.client import Client
>>> 
>>> client = Client("webdc.eu", 18001)
>>> start = UTCDateTime(2009, 8, 20, 6, 35, 0, 0)
>>> st = client.getWaveform("GR", "FUR", "", "BH*", start, start + 60*30)
>>> st.plot()

Waveform data fetched from an ArcLink node is converted into an ObsPy stream
object. The seismogram is truncated by the ObsPy client to the actual requested
time span, as ArcLink internally cuts SEED files for performance reasons on
record base in order to avoid uncompressing the waveform data. The output of
the script above is shown in the next picture.

Video Tutorial
--------------
http://svn.geophysik.uni-muenchen.de/obspy/obspy.arclink/arclink.swf

Further Examples
----------------
The following methods are demonstrated using the initialized client from the
example above.

>>> Client.getPAZ()

Requests poles, zeros, gain and sensitivity of a single channel for a certain
time span.

>>> start = UTCDateTime(2009, 1, 1)
>>> paz = client.getPAZ('BW', 'MANZ', '', 'EHZ', start, start + 1)
>>> paz
{'gain': 60077000.0,
 'poles': [(-0.037004000000000002+0.037016j),
           (-0.037004000000000002-0.037016j),
           (-251.33000000000001+0j),
           (-131.03999999999999-467.29000000000002j),
           (-131.03999999999999+467.29000000000002j)],
 'sensitivity': 2516778400.0,
 'zeros': [0j, 0j]}

Client.saveResponse()
^^^^^^^^^^^^^^^^^^^^^
Writes a response information into a file.

>>> start = UTCDateTime(2009, 1, 1)
>>> client.saveResponse('BW.MANZ..EHZ.dataless', 'BW', 'MANZ', '', 'EHZ', start, start + 1)

Client.saveWaveform()
^^^^^^^^^^^^^^^^^^^^^
Writes a seismogramm into a Full SEED volume.

>>> start = UTCDateTime(2009, 1, 1)
>>> client.saveWaveform('BW.MANZ..EHZ.seed', 'BW', 'MANZ', '', '*', start, start + 60, format='FSEED')

Source Code & Installation
--------------------------
The source code of obspy.arclink is accessable via a web-based interface or by
using a Subversion client::

    svn co https://svn.geophysik.uni-muenchen.de/svn/obspy/obspy.arclink/trunk

On a machine with a proper Python setup the package can be installed via a
simple::

    easy_install obspy.arclink

Please take a look into installation instructions if you run into issues during
the installation process.

Further resources
-----------------
* ArcLink protocol specifications
* Short introduction to the ArcLink protocol
* Latest ArcLink server package
* Seismolink: a SOAP Web Service on top of the ArcLink network 
"""

from client import Client
