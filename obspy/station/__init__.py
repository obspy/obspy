# -*- coding: utf-8 -*-
"""
obspy.station - Classes for handling station metadata
=====================================================
This module provides a class hierarchy to consistently handle station metadata.
This class hierarchy is closely modelled after the upcoming de-facto standard
format `FDSN StationXML <http://www.fdsn.org/xml/station/>`_ which was
developed as a human readable XML replacement for Dataless SEED.

.. note:: IRIS is maintaining a Java tool for converting dataless SEED into
          StationXML and vice versa at
          https://seiscode.iris.washington.edu/projects/stationxml-converter

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Chad Trabant
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Reading
-------
StationXML files can be read using the
:func:`~obspy.station.inventory.read_inventory()` function that returns an
:class:`~obspy.station.inventory.Inventory` object.

>>> from obspy import read_inventory
>>> inv = read_inventory("/path/to/BW_RJOB.xml")
>>> inv  # doctest: +ELLIPSIS
<obspy.station.inventory.Inventory object at 0x...>
>>> print(inv)  # doctest: +NORMALIZE_WHITESPACE
Inventory created at 2013-12-07T18:00:42.878000Z
    Created by: fdsn-stationxml-converter/1.0.0
            http://www.iris.edu/fdsnstationconverter
    Sending institution: Erdbebendienst Bayern
    Contains:
        Networks (1):
            BW
        Stations (1):
            BW.RJOB (Jochberg, Bavaria, BW-Net)
        Channels (3):
            BW.RJOB..EHE, BW.RJOB..EHN, BW.RJOB..EHZ

The file format in principle is autodetected. However, the autodetection uses
the official StationXML XSD schema and unfortunately many real world files
currently show minor deviations from the official StationXML definition causing
the autodetection to fail. Thus, manually specifying the format is a good idea:

>>> inv = read_inventory("/path/to/BW_RJOB.xml", format="STATIONXML")

Class hierarchy
---------------
The :class:`~obspy.station.inventory.Inventory` class has a hierarchical
structure, starting with a list of :class:`Networks
<obspy.station.network.Network>`, each containing a list of :class:`Stations
<obspy.station.station.Station>` which again each contain a list of
:class:`Channels <obspy.station.channel.Channel>`. The :class:`Responses
<obspy.station.response.Response>` are attached to the channels as an
attribute.

.. figure:: /_images/Inventory.png

>>> net = inv[0]
>>> net  # doctest: +ELLIPSIS
<obspy.station.network.Network object at 0x...>
>>> print(net)  # doctest: +NORMALIZE_WHITESPACE
Network BW (BayernNetz)
    Station Count: None/None (Selected/Total)
    None -
    Access: None
    Contains:
        Stations (1):
            BW.RJOB (Jochberg, Bavaria, BW-Net)
        Channels (3):
            BW.RJOB..EHZ, BW.RJOB..EHN, BW.RJOB..EHE


>>> sta = net[0]
>>> print(sta)  # doctest: +NORMALIZE_WHITESPACE
Station RJOB (Jochberg, Bavaria, BW-Net)
    Station Code: RJOB
    Channel Count: None/None (Selected/Total)
    2007-12-17T00:00:00.000000Z -
    Access: None
    Latitude: 47.74, Longitude: 12.80, Elevation: 860.0 m
    Available Channels:
        RJOB..EHZ, RJOB..EHN, RJOB..EHE

>>> cha = sta[0]
>>> print(cha)  # doctest: +NORMALIZE_WHITESPACE
Channel 'EHZ', Location ''
   Timerange: 2007-12-17T00:00:00.000000Z - --
   Latitude: 47.74, Longitude: 12.80, Elevation: 860.0 m, Local Depth: 0.0 m
   Azimuth: 0.00 degrees from north, clockwise
   Dip: -90.00 degrees down from horizontal
   Channel types: TRIGGERED, GEOPHYSICAL
   Sampling Rate: 200.00 Hz
   Sensor: Streckeisen STS-2/N seismometer
   Response information available

>>> print(cha.response)  # doctest: +NORMALIZE_WHITESPACE + ELLIPSIS
Channel Response
   From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
   Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
   4 stages:
      Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
      Stage 2: CoefficientsTypeResponseStage from V to COUNTS, gain: 1.67...
      Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
      Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1

Preview plots of station map and instrument response
----------------------------------------------------
For station metadata, preview plot routines for geographic location of stations
as well as bode plots for channel instrument response information are
available. The routines for station map plots are:

 * :meth:`Inventory.plot() <obspy.station.inventory.Inventory.plot>`
 * :meth:`Network.plot() <obspy.station.network.Network.plot>`

For example:

.. code-block:: python

    >>> from obspy import read_inventory
    >>> inv = read_inventory()
    >>> inv.plot()  # doctest: +SKIP

.. plot::

    from obspy import read_inventory
    inv = read_inventory()
    inv.plot()

The routines for bode plots of channel instrument response are:

 * :meth:`Inventory.plot_response()
   <obspy.station.inventory.Inventory.plot_response>`
 * :meth:`Network.plot_response()
   <obspy.station.network.Network.plot_response>`
 * :meth:`Station.plot() <obspy.station.station.Station.plot>`
 * :meth:`Channel.plot() <obspy.station.channel.Channel.plot>`
 * :meth:`Response.plot() <obspy.station.response.Response.plot>`

For example:

.. code-block:: python

    >>> from obspy import read_inventory
    >>> inv = read_inventory()
    >>> resp = inv[0][0][0].response
    >>> resp.plot(0.001, output="VEL")  # doctest: +SKIP

.. plot::

    from obspy import read_inventory
    inv = read_inventory()
    resp = inv[0][0][0].response
    resp.plot(0.001, output="VEL")

For more examples see the :ref:`Obspy Gallery <gallery>`.

Dealing with the Response information
-------------------------------------
The :meth:`~obspy.station.response.Response.get_evalresp_response` method will
call some functions within evalresp to generate the response.

>>> response = cha.response
>>> response, freqs = response.get_evalresp_response(0.1, 16384, output="VEL")
>>> print(response)  # doctest: +NORMALIZE_WHITESPACE
 [  0.00000000e+00 +0.00000000e+00j  -1.36383361e+07 +1.42086194e+06j
   -5.36470300e+07 +1.13620679e+07j ...,   2.48907496e+09 -3.94151237e+08j
    2.48906963e+09 -3.94200472e+08j   2.48906430e+09 -3.94249707e+08j]


Some convenience methods to perform an instrument correction on
:class:`~obspy.core.stream.Stream` (and :class:`~obspy.core.trace.Trace`)
objects are available and most users will want to use those. The
:meth:`~obspy.core.stream.Stream.attach_response()` method will attach matching
responses to each trace if they are available within the inventory object. The
:meth:`~obspy.core.stream.Stream.remove_response()` method deconvolves the
instrument response in-place. As always see the corresponding docs pages for a
full list of options and a more detailed explanation.

>>> from obspy import read
>>> st = read()
>>> inv = read_inventory("/path/to/BW_RJOB.xml")
>>> st.attach_response(inv)  # doctest: +NORMALIZE_WHITESPACE
 []
>>> st.remove_response(output="VEL", water_level=20)  # doctest: +ELLIPSIS
<obspy.core.stream.Stream object at 0x...>

Writing
-------
:class:`~obspy.station.inventory.Inventory` objects can be exported to
StationXML files, e.g. after making modifications.

>>> inv.write('my_inventory.xml', format='STATIONXML')  # doctest: +SKIP
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from obspy.station.util import BaseNode, Equipment, Operator, Person, \
    PhoneNumber, ExternalReference, Comment, Site
from obspy.station.channel import Channel
from obspy.station.station import Station
from obspy.station.network import Network
from obspy.station.inventory import Inventory, read_inventory
from obspy.station.response import ResponseStage, PolesZerosResponseStage, \
    CoefficientsTypeResponseStage, ResponseListResponseStage, \
    FIRResponseStage, PolynomialResponseStage, Response, \
    InstrumentSensitivity, InstrumentPolynomial


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
