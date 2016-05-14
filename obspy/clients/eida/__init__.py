#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web Service client with EIDA routing and authentication support.

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
from future import standard_library
from future.utils import PY2, native_str

import io
import obspy.clients.fdsn
from . import fetch

if PY2:
    import urlparse

else:
    import urllib.parse as urlparse

class Client(obspy.clients.fdsn.Client):
    def __init__(self, base_url='GFZ', retry_count=10, retry_wait=60, maxthreads=5, credentials=None, authdata=None, **kwargs):
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
        :param credentials: url -> (username, password).
        :type authdata: string
        :param authdata: Authentication token (PGP base64 format).

        """
        super(Client, self).__init__(base_url, **kwargs)
        self.__retry_count = retry_count
        self.__retry_wait = retry_wait
        self.__maxthreads = maxthreads
        self.__credentials = credentials
        self.__authdata = authdata

    def _create_url_from_parameters(self, service, *args):
        url = super(Client, self)._create_url_from_parameters(service, *args)

        if service in ('dataselect', 'station'):
            # construct a URL for the routing service
            u = urlparse.urlparse(url)
            return urlparse.urlunparse((u.scheme, u.netloc, u'/eidaws/routing/1/query', u'', u.query + u'&service=' + service, u''))

        else: # 'event' is not routed
            return url

    def _download(self, url, return_string=False, data=None, use_gzip=True):
        u = urlparse.urlparse(url)
        q = dict((p, v[0]) for (p, v) in urlparse.parse_qs(u[4]).items())

        if 'service' in q:
            dest = io.BytesIO()

            try:
                fetch.route(fetch.RoutingURL(u, q), self.__credentials, self.__authdata, data, dest,
                    self.timeout, self.__retry_count, self.__retry_wait, self.__maxthreads, self.debug)

            except fetch.Error as e:
                raise obspy.clients.fdsn.header.FDSNException(str(e))

            if dest.tell() == 0:
                raise obspy.clients.fdsn.header.FDSNException("No data available for request.")

            if return_string:
                return dest.getvalue()

            else:
                return dest

        else:
            return super(Client, self)._download(self, url, return_string, data, use_gzip)

