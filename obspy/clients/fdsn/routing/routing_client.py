#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for all FDSN routers.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
from multiprocessing.dummy import Pool as ThreadPool

import decorator

from obspy.core.compatibility import urlparse, string_types
import obspy

from ...base import HTTPClient
from .. import client
from ..client import raise_on_error
from ..header import FDSNException, URL_MAPPINGS, FDSNNoDataException


def RoutingClient(routing_type, *args, **kwargs):  # NOQA
    """
    Helper function to get the correct routing instance.

    :type routing_type: str
    :param routing_type: The type of router to initialize.
        ``"iris-federator"`` or ``"eida-routing"``. Will consequently return
        either a :class:`~.federator_routing_client.FederatorRoutingClient` or
        a :class:`~.eidaws_routing_client.EIDAWSRoutingClient` object,
        respectively.

    Remaining ``args`` and ``kwargs`` will be passed to the underlying classes.

    >>> from obspy.clients.fdsn import RoutingClient

    Get an instance of a routing client using the IRIS Federator:

    >>> c = RoutingClient("iris-federator")
    >>> print(type(c))  # doctest: +ELLIPSIS
    <class '...routing.federator_routing_client.FederatorRoutingClient'>

    Or get an instance of a routing client using the EIDAWS routing web
    service:

    >>> c = RoutingClient("eida-routing")
    >>> print(type(c))  # doctest: +ELLIPSIS
    <class '...routing.eidaws_routing_client.EIDAWSRoutingClient'>
    """
    if routing_type.lower() == "eida-routing":
        from .eidaws_routing_client import EIDAWSRoutingClient
        return EIDAWSRoutingClient(*args, **kwargs)
    if routing_type.lower() == "iris-federator":
        from .federator_routing_client import FederatorRoutingClient
        return FederatorRoutingClient(*args, **kwargs)
    else:
        raise NotImplementedError(
            "Routing type '%s' is not implemented. Available types: "
            "`iris-federator`, `eida-routing`" % routing_type)


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


def _download_bulk(r):
    c = client.Client(r["endpoint"], debug=r["debug"], timeout=r["timeout"])
    if r["data_type"] == "waveform":
        fct = c.get_waveforms_bulk
        service = c.services["dataselect"]
    elif r["data_type"] == "station":
        fct = c.get_stations_bulk
        service = c.services["station"]
    # Keep only kwargs that are supported by this particular service.
    kwargs = {k: v for k, v in r["kwargs"].items() if k in service}
    bulk_str = ""
    for key, value in kwargs.items():
        bulk_str += "%s=%s\n" % (key, str(value))
    try:
        return fct(bulk_str + r["bulk_str"])
    except FDSNException:
        return None


def _strip_protocol(url):
    url = urlparse(url)
    return url.netloc + url.path


# Does not inherit from the FDSN client as that would be fairly hacky as
# some methods just make no sense for the routing client to have (e.g.
# get_events() but also others).
class BaseRoutingClient(HTTPClient):
    def __init__(self, debug=False, timeout=120, include_providers=None,
                 exclude_providers=None):
        """
        :type routing_type: str
        :param routing_type: str
        :type exclude_providers: str or list of str
        :param exclude_providers: Get no data from these providers. Can be
            the full HTTP address or one of the shortcuts ObsPy knows about.
        :type include_providers: str or list of str
        :param include_providers: Get data only from these providers. Can be
            the full HTTP address of one of the shortcuts ObsPy knows about.
        """
        HTTPClient.__init__(self, debug=debug, timeout=timeout)
        self.include_providers = include_providers
        self.exclude_providers = exclude_providers

    @property
    def include_providers(self):
        return self.__include_providers

    @include_providers.setter
    def include_providers(self, value):
        self.__include_providers = self._expand_providers(value)

    @property
    def exclude_providers(self):
        return self.__exclude_providers

    @exclude_providers.setter
    def exclude_providers(self, value):
        self.__exclude_providers = self._expand_providers(value)

    def _expand_providers(self, providers):
        if providers is None:
            providers = []
        elif isinstance(providers, string_types):
            providers = [providers]
        return [_strip_protocol(URL_MAPPINGS[_i])
                if _i in URL_MAPPINGS
                else _strip_protocol(_i) for _i in providers]

    def _filter_requests(self, split):
        """
        Filter requests based on including and excluding providers.

        :type split: dict
        :param split: A dictionary containing the desired routing.
        """
        key_map = {_strip_protocol(url): url for url in split.keys()}

        # Apply both filters.
        f_keys = set(key_map.keys())
        if self.include_providers:
            f_keys = f_keys.intersection(set(self.include_providers))
        f_keys = f_keys.difference(set(self.exclude_providers))

        return {key_map[k]: split[key_map[k]] for k in f_keys}

    def _download_waveforms(self, split, **kwargs):
        return self._download_parallel(split, data_type="waveform", **kwargs)

    def _download_stations(self, split, **kwargs):
        return self._download_parallel(split, data_type="station", **kwargs)

    def _download_parallel(self, split, data_type, **kwargs):
        # Apply the provider filter.
        split = self._filter_requests(split)

        if not split:
            raise FDSNNoDataException(
                "Nothing remains to download after the provider "
                "inclusion/exclusion filters have been applied.")

        if data_type not in ["waveform", "station"]:  # pragma: no cover
            raise ValueError("Invalid data type.")

        # One thread per data center.
        dl_requests = []
        for k, v in split.items():
            dl_requests.append({
                "debug": self._debug,
                "timeout": self._timeout,
                "endpoint": k,
                "bulk_str": v,
                "data_type": data_type,
                "kwargs": kwargs})
        pool = ThreadPool(processes=len(dl_requests))
        results = pool.map(_download_bulk, dl_requests)

        # Merge all results into a single object.
        if data_type == "waveform":
            collection = obspy.Stream()
        elif data_type == "station":
            collection = obspy.Inventory(
                networks=[],
                source="ObsPy FDSN Routing %s" % obspy.__version__)
        else:  # pragma: no cover
            raise ValueError

        for _i in results:
            if not _i:
                continue
            collection += _i

        return collection

    def _handle_requests_http_error(self, r):
        """
        This assumes the same error code semantics as the base fdsnws web
        services.

        Please overwrite this method in a child class if necessary.
        """
        if r.content:  # pragma: no cover
            c = r.content
        else:
            c = r.reason

        if hasattr(c, "encode"):
            c = c.encode()

        with io.BytesIO(c) as f:
            f.seek(0, 0)
            raise_on_error(r.status_code, c)

    @_assert_filename_not_in_kwargs
    @_assert_attach_response_not_in_kwargs
    def get_waveforms(self, starttime, endtime, **kwargs):
        """
        Get waveforms from multiple data centers.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_waveforms()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        dataselect service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` and ``attach_response`` parameters of the single
        provider FDSN client are not supported.

        This can route on a number of different parameters, depending on the
        service, please see the web site of each individual routing service
        for details.
        """
        # This just calls the bulk downloader to only implement the logic once.
        # Just pass these to the bulk request.
        bulk = []
        for _i in ["network", "station", "location", "channel"]:
            if _i in kwargs:
                bulk.append(kwargs[_i])
                del kwargs[_i]
            else:
                bulk.append("*")
        bulk.extend([starttime, endtime])
        return self.get_waveforms_bulk([bulk], **kwargs)


if __name__ == '__main__':  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
