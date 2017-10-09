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

from multiprocessing.dummy import Pool as ThreadPool

import obspy

from ...base import HTTPClient
from ..client import Client
from ..header import FDSNException, URL_MAPPINGS
from future.utils import string_types, with_metaclass


def RoutingClient(routing_type, *args, **kwargs):
    from .eidaws_routing_client import EIDAWSRoutingClient
    if routing_type.lower() == "eidaws":
        return EIDAWSRoutingClient(*args, **kwargs)
    else:
        raise NotImplementedError(
            "Routing type '%s' is not implemented. Available types: "
            "EIDAWS")


def _download_bulk(r):
    c = Client(r["endpoint"], debug=r["debug"], timeout=r["timeout"])
    if r["data_type"] == "waveform":
        fct = c.get_waveforms_bulk
    elif r["data_type"] == "station":
        fct = c.get_stations_bulk
    try:
        return fct(r["bulk_str"])
    except FDSNException:
        return None


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
        self.include_providers = self._expand_providers(include_providers)
        self.exclude_providers = self._expand_providers(exclude_providers)

    def _expand_providers(self, providers):
        if providers is None:
            providers = []
        elif isinstance(providers, string_types):
            providers = [providers]
        return [URL_MAPPINGS[_i] if _i in URL_MAPPINGS else _i
                for _i in providers]

    def _filter_requests(self, split):
        """
        Filter requests based on including and excluding providers.

        :type split: dict
        :param split: A dictionary containing the desired routing.
        """
        filtered_split = {}
        for key, value in split.items():
            pass

    def _download_waveforms(self, split, **kwargs):
        return self._download_parallel(split, data_type="waveform", **kwargs)

    def _download_stations(self, split, **kwargs):
        return self._download_parallel(split, data_type="station", **kwargs)

    def _download_parallel(self, split, data_type, **kwargs):
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
