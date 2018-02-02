# -*- coding: utf-8 -*-
"""
obspy.clients.iris - IRIS web service client for ObsPy
======================================================
The obspy.clients.iris package contains a client for the DMC Web services
provided by IRIS (https://service.iris.edu/irisws/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

Local Alternatives
------------------

ObsPy contains local alternatives to some of the calculation tools offered by
the IRIS web services. Consider using them when working within ObsPy:

+---------------------------------------------------------+--------------------------------------------------------------+
| IRIS Web Service                                        | Equivalent ObsPy Function/Module                             |
+=========================================================+==============================================================+
| :meth:`obspy.clients.iris.client.Client.traveltime()`   | :mod:`obspy.taup`                                            |
+---------------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.clients.iris.client.Client.distaz()`       | :mod:`obspy.geodetics`                                       |
+---------------------------------------------------------+--------------------------------------------------------------+
| :meth:`obspy.clients.iris.client.Client.flinnengdahl()` | :class:`obspy.geodetics.flinnengdahl.FlinnEngdahl`           |
+---------------------------------------------------------+--------------------------------------------------------------+

Web service Interfaces
----------------------

Each of the following methods directly wrap a single Web service provided by
IRIS (https://service.iris.edu/irisws/):

**Request Tools**

* :meth:`~obspy.clients.iris.client.Client.evalresp()` - evaluates instrument
  response information stored at the IRIS DMC and outputs ASCII data or
  `Bode Plots <https://en.wikipedia.org/wiki/Bode_plots>`_.
* :meth:`~obspy.clients.iris.client.Client.resp()` - provides access to channel
  response information in the SEED RESP format (as used by evalresp)
* :meth:`~obspy.clients.iris.client.Client.sacpz()` - provides access to
  instrument response information (per-channel) as poles and zeros in the ASCII
  format used by SAC and other programs
* :meth:`~obspy.clients.iris.client.Client.timeseries()` - fetches segments of
  seismic data and returns data formatted in either MiniSEED, ASCII or SAC. It
  can optionally filter the data.

**Calculation Tools**

* :meth:`~obspy.clients.iris.client.Client.traveltime()` - calculates
  travel-times for seismic phases using a 1-D spherical Earth model.
* :meth:`~obspy.clients.iris.client.Client.distaz()` - calculate the distance
  and azimuth between two points on a sphere.
* :meth:`~obspy.clients.iris.client.Client.flinnengdahl()` - converts a
  latitude, longitude pair into either a Flinn-Engdahl seismic region code or
  region name.


Please see the documentation for each method for further information and
examples to retrieve various data from the IRIS DMC.
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
