#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' init for obspy.clients.fdsn.routers.routers '''

# convenience imports
from .routing_client import (RoutingClient, ResponseManager)  # NOQA
from .routing_response import RoutingResponse    # NOQA
from .fedcatalog_response_parser import (FederatedResponse)    # NOQA
from .fedcatalog_routing_client import (PROVIDER_METADATA, FederatedClient,
                                        FederatedResponseManager)    # NOQA
