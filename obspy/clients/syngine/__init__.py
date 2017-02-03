# -*- coding: utf-8 -*-
"""
obspy.clients.syngine - Client for the IRIS Syngine service
===========================================================

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
...                           eventid="GCMT:C201002270634A")  # doctest: +VCR
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
                              eventid="GCMT:C201002270634A")  # doctest: +VCR
    st.plot()

The available parameters are explained in detail in the
:meth:`~obspy.clients.syngine.client.Client.get_waveforms()` method and on
the `Syngine <https://ds.iris.edu/ds/products/syngine/>`_ website.


The queries are quite flexible. The following uses a station name wildcard
and only requests data around the P arrival. Please be a bit careful as one
can potentially download a lot of data with a single request.

>>> st = client.get_waveforms(model="ak135f_5s", network="IU", station="AN*",
...                           eventid="GCMT:C201002270634A", starttime="P-10",
...                           endtime="P+20")  # doctest: +VCR
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
...     bulk=bulk, starttime="P-10", endtime="P+20")  # doctest: +VCR
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

>>> from obspy.clients.syngine import Client
>>> c = Client()
>>> c.get_available_models()  # doctest: +VCR +ELLIPSIS +IGNORE_WHITESPACES
{'ak135f_5s': {'components': 'vertical and horizontal',
  'description': 'ak135 with density & Q of Montagner & Kennet(1996)',
  'max_sampling_period': '1.278000',
  'default_components': 'ZNE',
  'max_event_depth': 750000,
  'length': 3904.29,
  'min_period': 5.125,
  'max_period': '~100',
  'default_dt': '0.25'},
 'prem_a_2s': ...
 ...
 }

The :func:`~obspy.clients.syngine.client.Client.get_model_info()`
method should be used to retrieve information about a specific model.

>>> from obspy.clients.syngine import Client
>>> c = Client()
>>> db_info = c.get_model_info(model_name="ak135f_5s")  # doctest: +VCR
>>> print(db_info.period)
5.125
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
