# -*- coding: utf-8 -*-
"""
obspy.clients.syngine - Client for the IRIS Syngine service
===========================================================

Contains request abilities that naturally integrate with the rest of ObsPy
for the IRIS syngine service (http://ds.iris.edu/ds/products/syngine/). The
service is able to generate fully three dimensional synthetics through
various 1D Earth models with arbitrary source-receiver geometries and source
mechanisms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)


Basic Usage
-----------

First initialize a client object.

>>> from obspy.clients.syngine import Client
>>> client = Client()

Then requests some data.

>>> st = client.get_waveforms(model="ak135f_5s", network="IU", station="ANMO",
...                           eventid="GCMT:C201002270634A")
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
IU.ANMO.SE.MXZ | 2010-02-27T06:35:14... - ... | 4.0 Hz, 15520 samples
IU.ANMO.SE.MXN | 2010-02-27T06:35:14... - ... | 4.0 Hz, 15520 samples
IU.ANMO.SE.MXE | 2010-02-27T06:35:14... - ... | 4.0 Hz, 15520 samples
>>> st.plot()  # doctest: +SKIP

.. plot::

    from obspy.clients.syngine import Client
    client = Client()
    st = client.get_waveforms(model="ak135f_5s", network="IU", station="ANMO",
                              eventid="GCMT:C201002270634A")
    st.plot()

The available parameters are explained in detail in the
:meth:`~obspy.clients.Syngine.client.Client.get_waveforms()` method and on
the `Syngine <http://ds.iris.edu/ds/products/syngine/>`_ website.


The queries are quite flexible. The following uses a station name wildcard
and only requests data around the P arrival. Please be a bit careful as one
can potentially download a lot of data with a single request.

>>> st = client.get_waveforms(model="ak135f_5s", network="IU", station="AN*",
...                           eventid="GCMT:C201002270634A",
...                           starttime="P-10", endtime="P+20")
>>> st.plot()  # doctest: +SKIP

.. plot::

    from obspy.clients.syngine import Client
    client = Client()
    st = client.get_waveforms(model="ak135f_5s", network="IU", station="AN*",
                              eventid="GCMT:C201002270634A",
                              starttime="P-10", endtime="P+20")
    st.plot()


Bulk Requests
-------------

It is also possible to send requests for multiple user-defined stations. See
the :meth:`~obspy.clients.Syngine.client.Client.get_waveforms_bulk()` method
for details. This example specifies a bunch of receiver coordinates and
requests seismograms for all of these.

>>> bulk = [{"latitude": 10.1, "longitude": 12.2, "stacode": "AA"},
...         {"latitude": 14.5, "longitude": 154.0, "stacode": "BB"}]
>>> st = client.get_waveforms_bulk(
...     model="ak135f_5s", eventid="GCMT:C201002270634A",
...     bulk=bulk, starttime="P-10", endtime="P+20")
>>> print(st)  # doctest: +ELLIPSIS
3 Trace(s) in Stream:
XX.AA.SE.MXZ | 2010-02-27T06:48:11.500000Z - ... | 4.0 Hz, 120 samples
XX.AA.SE.MXN | 2010-02-27T06:48:11.500000Z - ... | 4.0 Hz, 120 samples
XX.AA.SE.MXE | 2010-02-27T06:48:11.500000Z - ... | 4.0 Hz, 120 samples
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from .client import Client  # NOQA

__all__ = [native_str("Client")]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
