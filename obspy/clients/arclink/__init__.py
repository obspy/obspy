# -*- coding: utf-8 -*-
"""
obspy.clients.arclink - ArcLink/WebDC request client for ObsPy
==============================================================

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
    (https://www.gnu.org/copyleft/lesser.html)

Basic Usage
-----------

.. note::
    The default client needs to open port 18002 to the host webdc.eu via TCP/IP
    in order to download the requested data. Please make sure that no firewall
    is blocking access to this server/port combination.

.. note::
    The ``user`` keyword in the following examples is used for identification
    with the ArcLink server as well as for usage statistics within the data
    center, so please provide a meaningful user id such as an email address.

(1) :meth:`~obspy.clients.arclink.client.Client.get_waveforms()`: The following
    example illustrates how to request and plot 18 seconds of all three single
    band channels (``"EH*"``) of station Jochberg/Hochstaufen (``"RJOB"``)
    of the Bavarian network (``"BW"``) for an seismic event around
    2009-08-20 04:03:12 (UTC).

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.arclink.client import Client
    >>> client = Client(user='test@obspy.org')
    >>> t = UTCDateTime("2009-08-20 04:03:12")
    >>> st = client.get_waveforms("BW", "RJOB", "", "EH*", t - 3, t + 15)
    >>> st.plot()  # doctest: +SKIP

    Waveform data fetched from an ArcLink node is converted into an ObsPy
    :class:`~obspy.core.stream.Stream` object. The seismogram is truncated by
    the ObsPy client to the actual requested time span, as ArcLink internally
    cuts SEED files for performance reasons on record base in order to avoid
    uncompressing the waveform data. The output of the script above is shown in
    the next picture.

    .. plot::

        from obspy import UTCDateTime
        from obspy.clients.arclink.client import Client
        client = Client(user='test@obspy.org')
        t = UTCDateTime("2009-08-20 04:03:12")
        st = client.get_waveforms("BW", "RJOB", "", "EH*", t - 3, t + 15)
        st.plot()  # doctest: +SKIP

(2) :meth:`~obspy.clients.arclink.client.Client.get_paz()`: Requests poles,
    zeros, gain and sensitivity of a single channel at a given time.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.arclink.client import Client
    >>> client = Client(user='test@obspy.org')
    >>> dt = UTCDateTime(2009, 1, 1)
    >>> paz = client.get_paz('BW', 'MANZ', '', 'EHZ', dt)
    >>> paz  # doctest: +NORMALIZE_WHITESPACE +SKIP
    AttribDict({'poles': [(-0.037004+0.037016j), (-0.037004-0.037016j),
                          (-251.33+0j), (-131.04-467.29j), (-131.04+467.29j)],
                'sensitivity': 2516778600.0,
                'zeros': [0j, 0j],
                'name': 'LMU:STS-2/N/g=1500',
                'gain': 60077000.0})

(3) :meth:`~obspy.clients.arclink.client.Client.save_response()`: Writes
    response information into a file.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.arclink.client import Client
    >>> client = Client(user='test@obspy.org')
    >>> t = UTCDateTime(2009, 1, 1)
    >>> client.save_response('BW.MANZ..EHZ.dataless', 'BW', 'MANZ', '', '*',
    ...                     t, t + 1, format="SEED")  # doctest: +SKIP

(4) :meth:`~obspy.clients.arclink.client.Client.save_waveforms()`: Writes the
    requested waveform unmodified into your local file system. Here we request
    a Full SEED volume.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.arclink.client import Client
    >>> client = Client(user='test@obspy.org')
    >>> t = UTCDateTime(2009, 1, 1, 12, 0)
    >>> client.save_waveforms('BW.MANZ..EHZ.seed', 'BW', 'MANZ', '', '*',
    ...                     t, t + 20, format='FSEED')  # doctest: +SKIP

(5) :meth:`~obspy.clients.arclink.client.Client.get_inventory()`: Request
    inventory data.

    >>> from obspy import UTCDateTime
    >>> from obspy.clients.arclink.client import Client
    >>> client = Client(user='test@obspy.org')
    >>> inv = client.get_inventory('BW', 'M*', '*', 'EHZ', restricted=False,
    ...                           permanent=True, min_longitude=12,
    ...                           max_longitude=12.2) #doctest: +SKIP
    >>> inv.keys()  # doctest: +SKIP
    ['BW.MROB', 'BW.MANZ..EHZ', 'BW', 'BW.MANZ', 'BW.MROB..EHZ']
    >>> inv['BW']  # doctest: +SKIP
    AttribDict({'description': 'BayernNetz', 'region': 'Germany', ...
    >>> inv['BW.MROB']  # doctest: +SKIP
    AttribDict({'code': 'MROB', 'description': 'Rosenbuehl, Bavaria', ...

Further Resources
-----------------
* ArcLink protocol specifications:

  * https://www.seiscomp3.org/wiki/doc/applications/arclink
  * http://geofon.gfz-potsdam.de/_sc3_neries_/arclink.pdf
  * https://raw.github.com/obspy/obspy/master/obspy/clients/arclink/docs/\
protocol.txt

* `Short introduction <http://www.webdc.eu/arclink/help.html>`_ to the ArcLink
  protocol
* Latest `ArcLink server
  <ftp://ftp.gfz-potsdam.de/pub/home/st/GEOFON/software/SeisComP/ArcLink/>`_
* `SeismoLink <http://neriesdataportalblog.freeflux.net/webservices/>`_: a SOAP
  Web service on top of the ArcLink network

.. _GEOFON:
        http://geofon.gfz-potsdam.de
.. _BGR:
        https://www.bgr.de
.. _IRIS:
        https://www.iris.edu/hq/
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
.. _`GNU Lesser General Public License, Version 3`:
        https://www.gnu.org/copyleft/lesser.html
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import warnings

from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

# Raise a deprecation warning. This has been requested by the EIDA management
# board.
msg = (
    "The ArcLink protocol will be deprecated in the near future. Please, use "
    "the client contacting the routing service provided by EIDA: "
    "https://docs.obspy.org/packages/obspy.clients.fdsn.html#"
    "basic-routing-clients-usage")
# suppress warning on docs build
if os.environ.get('SPHINX') != 'true':
    warnings.warn(msg, category=ObsPyDeprecationWarning)

from .client import Client  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
