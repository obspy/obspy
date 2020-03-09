# -*- coding: utf-8 -*-
"""
obspy.clients.syngine - IRIS Syngine client for ObsPy
=====================================================

This module offers methods to download from the IRIS syngine service
(https://ds.iris.edu/ds/products/syngine/). The service is able to generate
fully three dimensional synthetics through various 1D Earth models with
arbitrary source-receiver geometries and source mechanisms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


Basic Usage
-----------

First initialize a client object.

>>> from obspy.clients.syngine import Client
>>> client = Client()

Then request some data.

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
:meth:`~obspy.clients.syngine.client.Client.get_waveforms()` method and on
the `Syngine <https://ds.iris.edu/ds/products/syngine/>`_ website.


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
    st = client.get_waveforms(model="ak135f_5s", network="IU", station="A*",
                              eventid="GCMT:C201002270634A",
                              starttime="P-10", endtime="P+20")
    st.plot()


Bulk Requests
-------------

It is also possible to send requests for multiple user-defined stations. See
the :meth:`~obspy.clients.syngine.client.Client.get_waveforms_bulk()` method
for details. This example specifies a bunch of receiver coordinates and
requests seismograms for all of these.

>>> bulk = [{"latitude": 10.1, "longitude": 12.2, "stationcode": "AA"},
...         {"latitude": 14.5, "longitude": 10.0, "stationcode": "BB"}]
>>> st = client.get_waveforms_bulk(
...     model="ak135f_5s", eventid="GCMT:C201002270634A",
...     bulk=bulk, starttime="P-10", endtime="P+20")
>>> print(st)  # doctest: +ELLIPSIS
6 Trace(s) in Stream:
XX.AA.SE.MXZ | 2010-02-27T06:48:11... - ... | 4.0 Hz, 120 samples
XX.AA.SE.MXN | 2010-02-27T06:48:11... - ... | 4.0 Hz, 120 samples
XX.AA.SE.MXE | 2010-02-27T06:48:11... - ... | 4.0 Hz, 120 samples
XX.BB.SE.MXZ | 2010-02-27T06:48:15... - ... | 4.0 Hz, 120 samples
XX.BB.SE.MXN | 2010-02-27T06:48:15... - ... | 4.0 Hz, 120 samples
XX.BB.SE.MXE | 2010-02-27T06:48:15... - ... | 4.0 Hz, 120 samples


Other Useful Methods
--------------------

Use the :meth:`~obspy.clients.syngine.client.Client.get_available_models()`
method for a list of all available methods including some meta-information.

>>> client.get_available_models()  # doctest: +SKIP
{'ak135f_1s': {'components': 'vertical only',
  'default_components': 'Z',
  'default_dt': '0.05',
  'description': 'ak135 with density & Q of Montagner & Kennet(1996)',
  'length': 1815.00690721715,
  'max_event_depth': 750000,
  'max_period': '~100',
  'max_sampling_period': '0.255383',
  'min_period': 1.04999995231628},
 'ak135f_2s': ...
 ...
 }

The :func:`~obspy.clients.syngine.client.Client.get_model_info()`
method should be used to retrieve information about a specific model.

>>> from obspy.clients.syngine import Client
>>> c = Client()
>>> db_info = c.get_model_info(model_name="ak135f_5s")
>>> print(db_info.period)
5.125
"""
from .client import Client  # NOQA

__all__ = ["Client"]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
