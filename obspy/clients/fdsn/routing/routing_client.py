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
from multiprocessing.dummy import Pool as ThreadPool

import decorator
import io
import sys
import traceback
import warnings
from urllib.parse import urlparse

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
    For example, credentials can be supported for all underlying data centers.
    See :meth:`BaseRoutingClient <BaseRoutingClient.__init__>` for details.

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
def _assert_format_not_in_kwargs(f, *args, **kwargs):
    if "format" in kwargs:
        raise ValueError("The `format` argument is not supported")
    return f(*args, **kwargs)


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


def _try_download_bulk(r):
    try:
        return _download_bulk(r)
    except Exception:
        reason = "".join(traceback.format_exception(*sys.exc_info()))
        warnings.warn(
            "Failed to download data of type '%s' from '%s' due to: \n%s" % (
                r["data_type"], r["endpoint"], reason))
        return None


def _download_bulk(r):
    # Figure out the passed credentials, if any. Two possibilities:
    # (1) User and password, given explicitly for the base URLs (or an
    #     explicity given `eida_token` key per URL).
    # (2) A global EIDA_TOKEN key. It will be used for all services that
    #     don't have explicit credentials and also support the `/auth` route.
    credentials = r["credentials"].get(urlparse(r["endpoint"]).netloc, {})
    try:
        c = client.Client(r["endpoint"], debug=r["debug"],
                          timeout=r["timeout"], **credentials)
    # This should rarely happen but better safe than sorry.
    except FDSNException as e:  # pragma: no cover
        msg = e.args[0]
        msg += "It will not be used for routing. Try again later?"
        warnings.warn(msg)
        return None

    if not credentials and "EIDA_TOKEN" in r["credentials"] and \
            c._has_eida_auth:
        c.set_eida_token(r["credentials"]["EIDA_TOKEN"])

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
                 exclude_providers=None, credentials=None):
        """
        :type routing_type: str
        :param routing_type: The type of
            router to initialize. For details see :func:`RoutingClient`.
        :type exclude_providers: str or list[str]
        :param exclude_providers: Get no data from these providers. Can be
            the full HTTP address or one of the shortcuts ObsPy knows about.
        :type include_providers: str or list[str]
        :param include_providers: Get data only from these providers. Can be
            the full HTTP address of one of the shortcuts ObsPy knows about.
        :type credentials: dict
        :param credentials: Credentials for the individual data centers as a
            dictionary that maps base url of FDSN web service to either
            username/password or EIDA token, e.g.
            ``credentials={
            'geofon.gfz-potsdam.de': {'eida_token': 'my_token_file.txt'},
            'service.iris.edu': {'user': 'me', 'password': 'my_pass'}
            'EIDA_TOKEN': '/path/to/token.txt'
            }``
            The root level ``'EIDA_TOKEN'`` will be applied to all data centers
            that claim to support the ``/auth`` route and don't have data
            center specific credentials.
            You can also use a URL mapping as for the normal FDSN client
            instead of the URL.
        """
        HTTPClient.__init__(self, debug=debug, timeout=timeout)
        self.include_providers = include_providers
        self.exclude_providers = exclude_providers

        # Parse credentials.
        self.credentials = {}
        for key, value in (credentials or {}).items():
            if key == "EIDA_TOKEN":
                self.credentials[key] = value
            # Map, if necessary.
            if key in URL_MAPPINGS:
                key = URL_MAPPINGS[key]
            # Make sure urlparse works correctly.
            if not key.startswith("http"):
                key = "http://" + key
            # Only use the location.
            self.credentials[urlparse(key).netloc] = value

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
        elif isinstance(providers, str):
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
                "kwargs": kwargs,
                "credentials": self.credentials})
        pool = ThreadPool(processes=len(dl_requests))
        results = pool.map(_try_download_bulk, dl_requests)

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

        # Explitly close the thread pool as somehow this does not work
        # automatically under linux. See #2342.
        pool.close()

        return collection

    def _handle_requests_http_error(self, r):
        """
        This assumes the same error code semantics as the base fdsnws web
        services.

        Please overwrite this method in a child class if necessary.
        """
        reason = r.reason.encode()
        if hasattr(r, "content"):
            reason += b" -- " + r.content
        with io.BytesIO(reason) as buf:
            raise_on_error(r.status_code, buf)

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

    def get_service_version(self):
        """
        Return a semantic version number of the remote service as a string.
        """
        r = self._download(self._url + "/version")
        return r.content.decode() if \
            hasattr(r.content, "decode") else r.content

    @_assert_filename_not_in_kwargs
    def get_stations(self, **kwargs):
        """
        Get stations from multiple data centers.

        It will pass on most parameters to the underlying routed service.
        They will also be passed on to the individual FDSNWS implementations
        if a service supports them.

        The ``filename`` parameter of the single provider FDSN client is not
        supported.

        This can route on a number of different parameters, please see the
        web sites of the
        `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_
        and of the `EIDAWS Routing Service
        <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_ for
        details.
        """
        # Just pass these to the bulk request.
        bulk = [kwargs.pop(key, '*') for key in (
                "network", "station", "location", "channel", "starttime",
                "endtime")]
        return self.get_stations_bulk([bulk], **kwargs)


if __name__ == '__main__':  # pragma: no cover
    import doctest
    doctest.testmod(exclude_empty=True)
