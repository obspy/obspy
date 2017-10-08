#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:class:`~obspy.clients.fdsn.routers.FedcatalogProviders` contains data center
(provider) details retrieved from the fedcatalog service

:class:`~obspy.clients.fdsn.routers.FederatedClient` is the FDSN Web service
request client. The end user will work almost exclusively with this class,
which has methods similar to :class:`~obspy.clients.fdsn.Client`

:class:`~obspy.clients.fdsn.routers.FederatedRoutingManager` provides parsing
capabilities, and helps the FederatedClient make requests to each individual
provider's service

:func:`distribute_args()` helps determine what parameters belong to the routing
service and which belong to the data provider's client service

:func:`get_bulk_string()` helps turn text and parameters into a valid bulk
request text block.

:func:`data_to_request()` helper function to convert
:class:`~obspy.core.inventory.inventory.Inventory` or
:class:`~obpsy.core.Stream` into FDSNBulkRequests. Useful for comparing what
    has been retrieved with what was requested.

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
from future.utils import string_types, PY2

from .routing_client import RoutingClient


class FederatorRoutingClient(RoutingClient):
    def __init__(self, *args, **kwargs):
        pass

    def get_waveforms(self):


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)
