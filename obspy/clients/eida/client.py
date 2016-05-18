#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web Service client using EIDA routing and authentication.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Helmholtz-Zentrum Potsdam - Deutsches GeoForschungsZentrum GFZ
    (geofon@gfz-potsdam.de)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2

import io
import obspy.clients.fdsn
from . import fetch

if PY2:
    import urlparse

else:
    import urllib.parse as urlparse


class Client(obspy.clients.fdsn.Client):
    """
    FDSN Web Service client using EIDA routing and authentication.

    For details see the :meth:`~obspy.clients.eida.client.Client.__init__()`
    method.
    """
    def __init__(self, base_url='GFZ', retry_count=10, retry_wait=60,
                 maxthreads=5, credentials=None, authdata=None, **kwargs):
        """
        Initializes an FDSN/EIDA Web Service client.

        :type base_url: str
        :param base_url: Base URL of FDSN/EIDA web service compatible server
            (e.g. "http://geofon.gfz-potsdam.de") or key string for recognized
            server. See :mod:`FDSN client documentation <obspy.clients.fdsn>`
        :type debug: bool
        :param debug: Debug flag.
        :type timeout: float
        :param timeout: Maximum time (in seconds) to wait for a single request
            to receive the first byte of the response (after which an exception
            is raised).
        :type retry_count: int
        :param retry_count: Number of retries.
        :type retry_wait: int
        :param retry_wait: Seconds to wait before each retry.
        :type maxthreads: int
        :param maxthreads: Maximum number of download threads.
        :type credentials: dict
        :param credentials: url => (username, password).
        :type authdata: string
        :param authdata: Authentication token (PGP base64 format).

        """
        super(Client, self).__init__(base_url, **kwargs)
        self.__retry_count = retry_count
        self.__retry_wait = retry_wait
        self.__maxthreads = maxthreads
        self.__credentials = credentials
        self.__authdata = authdata

    def _download(self, url, return_string=False, data=None, use_gzip=True):
        u = urlparse.urlparse(url)
        q = dict((p, v[0]) for (p, v) in urlparse.parse_qs(u.query).items())

        if '/dataselect/' in u.path:
            q['service'] = 'dataselect'

        elif '/station/' in u.path:
            q['service'] = 'station'

        else:  # 'event' is not routed
            return super(Client, self)._download(
                    url, return_string, data, use_gzip)

        u = urlparse.ParseResult(
                u.scheme, u.netloc, '/eidaws/routing/1/query', '', '', '')

        if data is not None:
            if isinstance(data, bytes):
                data = data.decode('utf-8')

            # Remove key=value parameters (not applicable to the routing
            # service) from POST data and put them into a separate dict.
            postlines = data.splitlines()

            for i in range(len(postlines)):
                kv = postlines[i].split('=')

                if len(kv) != 2:
                    break

                q[kv[0]] = kv[1]

            data = '\n'.join(postlines[i:])

        elif 'end' not in q:
            # IRIS FDSNWS does not accept POST data with missing endtime.
            q['end'] = '2500-01-01T00:00:00Z'

        dest = io.BytesIO()

        try:
            fetch.route(fetch.RoutingURL(u, q),
                        self.__credentials,
                        self.__authdata,
                        data,
                        dest,
                        self.timeout,
                        self.__retry_count,
                        self.__retry_wait,
                        self.__maxthreads,
                        self.debug)

        except fetch.Error as e:
            raise obspy.clients.fdsn.header.FDSNException(str(e))

        if dest.tell() == 0:
            raise obspy.clients.fdsn.header.FDSNException(
                    "No data available for request.")

        if return_string:
            return dest.getvalue()

        else:
            return dest


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
