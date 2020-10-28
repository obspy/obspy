# -*- coding: utf-8 -*-
"""
Routing client for the EIDAWS routing service.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections

from ..client import get_bulk_string
from ..header import FDSNNoDataException
from .routing_client import (
    BaseRoutingClient, _assert_attach_response_not_in_kwargs,
    _assert_filename_not_in_kwargs)


class EIDAWSRoutingClient(BaseRoutingClient):
    """
    Routing client for the EIDAWS routing service.

    http://www.orfeus-eu.org/data/eida/webservices/routing/

    For waveform queries it will first launch a station query, get the
    station information at each data center with additional constraints
    (e.g. latitude/longitude/...) and use that information for the final
    waveform query. This means that with ObsPy the EIDA routing client
    behaves very similar to the IRIS federator routing client.
    """
    def __init__(self, url="http://www.orfeus-eu.org/eidaws/routing/1",
                 include_providers=None, exclude_providers=None,
                 debug=False, timeout=120, **kwargs):
        """
        Initialize an EIDAWS router client.

        All parameters except ``url`` are passed on to the
        :class:`~obspy.clients.fdsn.routing.routing_clieng.BaseRoutingClient`
        parent class

        :param url: The URL of the routing service.
        :type url: str
        """
        BaseRoutingClient.__init__(self, debug=debug, timeout=timeout,
                                   include_providers=include_providers,
                                   exclude_providers=exclude_providers,
                                   **kwargs)
        self._url = url

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

        This can route on a number of different parameters, please see the
        web site of the `EIDAWS Routing Service
        <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_
        for details.
        """
        # Multi-step procedure - first get the stations to be able to use
        # more query parameters - and then construct the waveform string
        # from it.
        #
        # This has to be done for each time interval - otherwise it will get
        # a lot more complicated. I guess in most cases people will use bulk
        # requests for the same time span so it should be fine.

        # Group by time interval - utilize the existing get_bulk_string()
        # method to not have to deal with various different inputs.
        _tmp_bulk_str = get_bulk_string(bulk, {})
        if hasattr(_tmp_bulk_str, "decode"):
            _tmp_bulk_str = _tmp_bulk_str.decode()

        # Parse and split.
        bulk_per_time_interval = collections.defaultdict(list)
        for line in _tmp_bulk_str.splitlines():
            # Cannot really happen - just a safety measure.
            if not line:  # pragma: no cover
                continue
            item = line.split()
            bulk_per_time_interval[(item[-2], item[-1])].append(item)

        # Build up the new bulk string for each found time interval by
        # querying the station services.
        new_bulk = []
        for t, _b in bulk_per_time_interval.items():
            # channel level and text to keep it fast.
            inv = self.get_stations_bulk(_b, format="text",
                                         level="channel", **kwargs)
            for c in sorted(set(inv.get_contents()["channels"])):
                new_bulk.append(c.split("."))
                new_bulk[-1].extend(t)

        # no available data, show appropriate error message and raise
        if not new_bulk:
            msg = ('No data available for request (requested time window '
                   'might be out of bounds of valid station epochs).')
            raise FDSNNoDataException(msg)

        # Finally get the waveforms by getting the routes and downloading
        # everytyhing. Don't directly pass in the initializer as the order
        # would not be guaranteed across all Python version.
        arguments = collections.OrderedDict()
        arguments["service"] = "dataselect"
        arguments["format"] = "post"

        bulk_str = get_bulk_string(new_bulk, arguments)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self._split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content)
        return self._download_waveforms(split, **kwargs)

    @_assert_filename_not_in_kwargs
    def get_stations(self, **kwargs):
        """
        Get stations from multiple data centers.

        Only the ``network``, ``station``, ``location``, ``channel``,
        ``starttime``, and ``endtime`` parameters are used for the actual
        routing. These and all other arguments are then just passed on to each
        single fdsnws station implementation.

        Arguments are the same as in
        :meth:`obspy.clients.fdsn.client.Client.get_stations()`.
        Any additional ``**kwargs`` are passed on to each individual service's
        station service if the service supports them (otherwise they are
        silently ignored for that particular fdsnws endpoint).

        The ``filename`` parameter of the single provider FDSN client is not
        supported for practical reasons.

        This can route on a number of different parameters, please see the
        web site of the `EIDAWS Routing Service
        <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_
        for details.
        """
        return super(EIDAWSRoutingClient, self).get_stations(**kwargs)

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

        This can route on a number of different parameters, please see the
        web site of the `EIDAWS Routing Service
        <http://www.orfeus-eu.org/data/eida/webservices/routing/>`_
        for details.
        """
        arguments = collections.OrderedDict()
        arguments["service"] = "station"
        arguments["format"] = "post"
        arguments["alternative"] = "false"
        bulk_str = get_bulk_string(bulk, arguments)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self._split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content)
        return self._download_stations(split, **kwargs)

    @staticmethod
    def _split_routing_response(data):
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
                current_key = line[:line.rfind("/fdsnws")]
                continue
            split[current_key].append(line)

        return {k: "\n".join(v) for k, v in split.items()}
