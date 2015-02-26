# -*- coding: utf-8 -*-
"""
obspy.neries - NERIES Web service client for ObsPy
==================================================
The obspy.neries package contains a client for the Seismic Data Portal
(http://www.seismicportal.eu) which was developed under the European
Commission-funded NERIES project. The Portal provides a single point of access
to diverse, distributed European earthquake data provided in a unique joint
initiative by observatories and research institutes in and around Europe.

.. warning::
    The `obspy.neries` module is deprecated and will be removed with the
    next major release. To access EMSC event data please use the
    :class:`obspy.fdsn client <obspy.fdsn.client.Client>`
    (use `Client(base_url='NERIES', ...)`), for access to
    ORFEUS waveform data please use the
    :class:`obspy.fdsn client <obspy.fdsn.client.Client>`
    (use `Client(base_url='ORFEUS', ...)`) and for travel times please use
    :mod:`obspy.taup`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------
(1) :meth:`~obspy.neries.client.Client.getEvents()`:
    This service was shut down on the server side, please use the
    obspy.fdsn Client instead (with `base_url='NERIES'`).

(2) :meth:`~obspy.neries.client.Client.getLatestEvents()`:
    This service was shut down on the server side, please use the
    obspy.fdsn Client instead (with `base_url='NERIES'`).

(3) :meth:`~obspy.neries.client.Client.getEventDetail()`:
    This service was shut down on the server side, please use the
    obspy.fdsn Client instead (with `base_url='NERIES'`).

(4) :meth:`~obspy.neries.client.Client.getWaveform()`: Wraps a NERIES Web
    service build on top of the ArcLink protocol. Here we give a small example
    how to fetch and display waveforms.

    >>> from obspy.neries import Client
    >>> from obspy import UTCDateTime
    >>> client = Client(user='test@obspy.org')
    >>> dt = UTCDateTime("2009-08-20 04:03:12")
    >>> st = client.getWaveform("BW", "RJOB", "", "EH*", dt - 3, dt + 15)
    >>> st.plot()  #doctest: +SKIP

    .. plot::

        from obspy.neries import Client
        from obspy import UTCDateTime
        client = Client(user='test@obspy.org')
        dt = UTCDateTime("2009-08-20 04:03:12")
        st = client.getWaveform("BW", "RJOB", "", "EH*", dt - 3, dt + 15)
        st.plot()

(5) :meth:`~obspy.neries.client.Client.getTravelTimes()`: Wraps a Taup Web
    service, an utility to compute arrival times using a few default velocity
    models such as ``'iasp91'``, ``'ak135'`` or ``'qdt'``.

    >>> from obspy.neries import Client
    >>> client = Client(user='test@obspy.org')
    >>> locations = [(48.0, 12.0), (48.1, 12.0)]
    >>> result = client.getTravelTimes(latitude=20.0, longitude=20.0,
    ...                                depth=10.0, locations=locations,
    ...                                model='iasp91')
    >>> len(result)
    2
    >>> result[0] # doctest: +SKIP
    {'P': 356981.13561726053, 'S': 646841.5619481194}
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import warnings

from .client import Client  # NOQA

msg = ("The obspy.neries module is deprecated and will be removed with the "
       "next major release. To access EMSC event data please use the "
       "obspy.fdsn client (use `Client(base_url='NERIES')`), for access to "
       "ORFEUS waveform data please use the obspy.fdsn client "
       "(use `Client(base_url='ORFEUS')`) and for travel times please use "
       "obspy.taup.")
warnings.warn(msg, DeprecationWarning)

try:
    import suds  # NOQA
except ImportError:
    msg = ("To use obspy.neries you need to install either 'suds' (works on "
           "Python 2.x) or 'suds-jurko' (works on Python 2.x and Python 3.x)")
    raise ImportError(msg)

__all__ = [native_str("Client")]

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
