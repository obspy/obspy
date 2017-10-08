#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routing client for the EIDAWS routing service.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import io

import decorator

from ..client import get_bulk_string, raise_on_error
from .routing_client import _RoutingClient


@decorator.decorator
def _assert_filename_not_in_kwargs(f, *args, **kwargs):
    if "filename" in kwargs:
        raise ValueError("The `filename` argument is not supported")
    return f(*args, **kwargs)

@decorator.decorator
def _assert_attach_response_not_in_kwargs(f, *args, **kwargs):
    if "attach_response" in kwargs:
        raise ValueError("The `attach_response` argument is not supported")
    return f(*args, **kwargs)


class EIDAWSRoutingClient(_RoutingClient):
    """
    Routing client for the EIDAWS routing service.

    http://www.orfeus-eu.org/data/eida/webservices/routing/
    """
    def __init__(self, url="http://www.orfeus-eu.org/eidaws/routing/1",
                 include_providers=None, exclude_providers=None,
                 debug=False, timeout=120):
        """
        Initialize an EIDAWS router client.

        All parameters except ``url`` are passed on to the
        :class:`~obspy.clients.fdsn.routing.routing_clieng._RoutingClient`
        parent class

        :param url: The URL of the routing service.
        :type url: str
        """
        _RoutingClient.__init__(self, debug=debug, timeout=timeout,
                                include_providers=include_providers,
                                exclude_providers=exclude_providers)
        self._url = url

    @_assert_filename_not_in_kwargs
    @_assert_attach_response_not_in_kwargs
    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, **kwargs):
        """
        Get waveforms from multiple data centers.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_waveforms()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        dataselect service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` and ``attach_response`` parameters of the single
        provider FDSN client are not supported.
        """
        # This just calls the bulk downloader to only implement the logic once.
        return self.get_waveforms_bulk(
            [(network, station, location, channel, starttime, endtime)],
            **kwargs)

    @_assert_filename_not_in_kwargs
    @_assert_attach_response_not_in_kwargs
    def get_waveforms_bulk(self, bulk, **kwargs):
        """
        Get waveforms from multiple data centers.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_waveforms_bulk()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        dataselect service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` and ``attach_response`` parameters of the single
        provider FDSN client are not supported.
        """
        # XXX: Really confusing but the waveform version of the service does
        # not really ever return something when called with POST. Not sure
        # if I'm doing it wrong or the service is wrong.
        arguments = collections.OrderedDict(
            service="station", format="post")

        bulk_str = get_bulk_string(bulk, arguments)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self.split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content)
        return self._download_waveforms(split, **kwargs)

    @_assert_filename_not_in_kwargs
    def get_stations(self, **kwargs):
        """
        Get stations from multiple data centers.

        Only the ``network``, ``station``, ``location``, ``channel``,
        ``starttime``, and ``endtime`` parameters are used for the actual
        routing. All other arguments are just passed on each single fdsnws
        station implementation.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_stations()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        station service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` parameter of the single provider FDSN client is not
        supported for practical reasons.
        """
        # This unfortunately cannot just be passed to the bulk service
        # as NSLC and the times might be empty and the bulk service does not
        # support that.
        #
        # Parameters the routing service can work with.
        kwargs_of_interest = ["network", "station", "location", "channel",
                              "starttime", "endtime"]
        params = {k: str(kwargs[k])
                  for k in kwargs_of_interest if k in kwargs}
        params["format"] = "post"
        params["service"] = "station"
        params["alternative"] = "false"
        r = self._download(self._url + "/query", params=params)
        split = self.split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content)
        return self._download_stations(split, **kwargs)

    @_assert_filename_not_in_kwargs
    def get_stations_bulk(self, bulk, **kwargs):
        """
        Bulk station download from multiple stations.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_stations_bulk()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        station service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` parameter of the single provider FDSN client is not
        supported for practical reasons.
        """
        arguments = collections.OrderedDict(
            service="station", format="post")
        bulk_str = get_bulk_string(bulk, arguments)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self.split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content)
        return self._download_stations(split, **kwargs)

    @staticmethod
    def split_routing_response(data):
        """
        Splits the routing responses per data center for the EIDAWS output.

        Returns a dictionary with the keys being the root URLs of the fdsnws
        endpoints and the values the data payloads for that endpoint.

        :param data: The return value from the EIDAWS routing service.
        """
        split = collections.defaultdict(list)
        current_key = None
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if "http" in line and "fdsnws" in line:
                current_key = line[:line.find("/fdsnws")]
                continue
            split[current_key].append(line)

        return {k: "\n".join(v) for k, v in split.items()}

    def get_service_version(self):
        """
        Return a semantic version number of the remote service as a string.
        """
        r = self._download(self._url + "/version")
        return r.content.decode() if \
            hasattr(r.content, "decode") else r.content

    def _handle_requests_http_error(self, r):
        # In lieu of any more knowledge let's just assume the same semantics
        # as for the fdsn web services.
        if r.content:
            c = r.content
        else:
            c = r.reason

        if hasattr(c, "encode"):
            c = c.encode()

        with io.BytesIO(c) as f:
            f.seek(0, 0)
            raise_on_error(r.status_code, c)


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)