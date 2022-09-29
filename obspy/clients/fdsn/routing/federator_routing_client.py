# -*- coding: utf-8 -*-
"""
Routing client for the IRIS federator routing service.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
    Celso G Reyes, 2017
    IRIS-DMC
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections

from ..client import get_bulk_string
from .routing_client import (
    BaseRoutingClient, _assert_filename_not_in_kwargs,
    _assert_attach_response_not_in_kwargs)


class FederatorRoutingClient(BaseRoutingClient):
    def __init__(self, url="http://service.iris.edu/irisws/fedcatalog/1",
                 include_providers=None, exclude_providers=None,
                 debug=False, timeout=120, **kwargs):
        """
        Initialize a federated routing client.

        All parameters except ``url`` are passed on to the
        :class:`~obspy.clients.fdsn.routing.routing_client.BaseRoutingClient`
        parent class

        :param url: The URL of the routing service.
        :type url: str
        """
        BaseRoutingClient.__init__(self, debug=debug, timeout=timeout,
                                   include_providers=include_providers,
                                   exclude_providers=exclude_providers,
                                   **kwargs)
        self._url = url

        # Parameters the routing service can work with. If this becomes a
        # standardized service and more implementations show up we should also
        # parse the WADL files - but right now this would just complicate
        # things.
        self.kwargs_of_interest = [
            "includeoverlaps", "level", "network", "station", "channel",
            "location", "starttime", "endtime", "startbefore", "endbefore",
            "startafter", "endafter", "minlatitude", "maxlatitude",
            "minlongitude", "maxlongitude", "latitude", "longitude",
            "minradius", "maxradius"]

    @_assert_attach_response_not_in_kwargs
    @_assert_filename_not_in_kwargs
    def get_waveforms_bulk(self, bulk, **kwargs):
        """
        Get waveforms from multiple data centers.

        It will pass on most parameters to the federated routing service.
        They will also be passed on to the individual FDSNWS implementations
        if a service supports them.

        The ``filename`` and ``attach_response`` parameters of the single
        provider FDSN client are not supported.

        This can route on a number of different parameters, please see the
        web site of the
        `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_
        for details.
        """
        bulk_params = ["network", "station", "location", "channel",
                       "starttime", "endtime"]
        for _i in bulk_params:
            if _i in kwargs:
                raise ValueError("`%s` must not be part of the optional "
                                 "parameters in a bulk request." % _i)

        params = {k: str(kwargs[k])
                  for k in self.kwargs_of_interest if k in kwargs}
        params["format"] = "request"

        bulk_str = get_bulk_string(bulk, params)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self._split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content,
            service="dataselect")
        return self._download_waveforms(split, **kwargs)

    @_assert_filename_not_in_kwargs
    def get_stations(self, **kwargs):
        """
        Get stations from multiple data centers.

        It will pass on most parameters to the federated routing service.
        They will also be passed on to the individual FDSNWS implementations
        if a service supports them.

        The ``filename`` parameter of the single provider FDSN client is not
        supported.

        This can route on a number of different parameters, please see the
        web site of the
        `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_
        for details.
        """
        return super(FederatorRoutingClient, self).get_stations(**kwargs)

    @_assert_filename_not_in_kwargs
    def get_stations_bulk(self, bulk, **kwargs):
        """
        Get stations from multiple data centers.

        It will pass on most parameters to the federated routing service.
        They will also be passed on to the individual FDSNWS implementations
        if a service supports them.

        The ``filename`` parameter of the single provider FDSN client is not
        supported.

        This can route on a number of different parameters, please see the
        web site of the
        `IRIS Federator  <https://service.iris.edu/irisws/fedcatalog/1/>`_
        for details.
        """
        bulk_params = ["network", "station", "location", "channel",
                       "starttime", "endtime"]
        for _i in bulk_params:
            if _i in kwargs:
                raise ValueError("`%s` must not be part of the optional "
                                 "parameters in a bulk request." % _i)

        params = collections.OrderedDict()
        for k in self.kwargs_of_interest:
            if k in kwargs:
                params[k] = str(kwargs[k])
        params["format"] = "request"

        bulk_str = get_bulk_string(bulk, params)
        r = self._download(self._url + "/query", data=bulk_str)
        split = self._split_routing_response(
            r.content.decode() if hasattr(r.content, "decode") else r.content,
            service="station")
        return self._download_stations(split, **kwargs)

    @staticmethod
    def _split_routing_response(data, service):
        """
        Splits the routing responses per data center for the federator output.

        Returns a dictionary with the keys being the root URLs of the fdsnws
        endpoints and the values the data payloads for that endpoint.

        :param data: The return value from the EIDAWS routing service.
        """
        if service.lower() == "dataselect":
            key = "DATASELECTSERVICE"
        elif service.lower() == "station":
            key = "STATIONSERVICE"
        else:
            raise ValueError("Service must be 'dataselect' or 'station'.")

        split = collections.defaultdict(list)
        current_key = None
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if "http://" in line or "https://" in line:
                if key not in line:
                    continue
                current_key = line[len(key) + 1:line.rfind("/fdsnws")]
                continue
            # Anything before the first data center can be ignored.
            if current_key is None:
                continue
            split[current_key].append(line)

        return {k: "\n".join(v) for k, v in split.items()}
