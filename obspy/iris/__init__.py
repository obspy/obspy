# -*- coding: utf-8 -*-
"""
obspy.iris - IRIS Web service client for ObsPy
==============================================
The obspy.iris package contains a client for the DMC Web services provided by
IRIS (http://service.iris.edu/irisws/).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Web service Interfaces
----------------------

Each of the following methods directly wrap a single Web service provided by
IRIS (http://service.iris.edu/irisws/):

**Request Tools**

* :meth:`~obspy.iris.client.Client.evalresp()` - evaluates instrument response
  information stored at the IRIS DMC and outputs ASCII data or
  `Bode Plots <http://en.wikipedia.org/wiki/Bode_plots>`_.
* :meth:`~obspy.iris.client.Client.resp()` - provides access to channel
  response information in the SEED RESP format (as used by evalresp)
* :meth:`~obspy.iris.client.Client.sacpz()` - provides access to instrument
  response information (per-channel) as poles and zeros in the ASCII format
  used by SAC and other programs
* :meth:`~obspy.iris.client.Client.timeseries()` - fetches segments of seismic
  data and returns data formatted in either MiniSEED, ASCII or SAC. It can
  optionally filter the data.

**Calculation Tools**

* :meth:`~obspy.iris.client.Client.traveltime()` - calculates travel-times for
  seismic phases using a 1-D spherical earth model.
* :meth:`~obspy.iris.client.Client.distaz()` - calculate the distance and
  azimuth between two points on a sphere.
* :meth:`~obspy.iris.client.Client.flinnengdahl()` - converts a latitude,
  longitude pair into either a Flinn-Engdahl seismic region code or region
  name.


Please see the documentation for each method for further information and
examples to retrieve various data from the IRIS DMC.
"""
from __future__ import absolute_import
from __future__ import unicode_literals

from .client import Client  # NOQA
import warnings

msg = ("Development and maintenance efforts will focus on the new obspy.fdsn "
       "client. Please consider moving all code from using obspy.iris to "
       "using obspy.fdsn.")
warnings.warn(msg, DeprecationWarning)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
