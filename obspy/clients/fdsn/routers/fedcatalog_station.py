#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    from obspy import UTCDateTime
    starttime = UTCDateTime("2001-01-01")

    # to access metadata via Client from a single data center's station service
    from obspy.clients.fdsn import Client
    client = Client("IRIS")

    # or, access data via FederatedClient from multiple data centers
    from obspy.clients.fdsn import FederatedClient
    client = FederatedClient("IRIS")


    # the requests are submited in exactly the same manner
    inv = client.get_stations(network="IU", station="req_str*", starttime=starttime)
'''

from __future__ import print_function
import sys
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.routers.fedcatalog_response_parser import (parse_federated_response,
                                                                   FederatedResponse,
                                                                   filter_requests)

def request_exists_in_inventory(inv, bulktext, level):
    'compare inventory to bulktext'
    # does not account for timespans, only for existence net-sta-loc-cha matches
    inv_set = INV_CONVERTER[level](inv)
    req_set = REQ_CONVERTER[level](bulktext)
    members = req_set.intersection(inv_set)
    non_members = req_set.difference(inv_set)
    return members, non_members



if __name__ == '__main__':
    #import doctest
    #doctest.testmod(exclude_empty=True)
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    ALL_OK = station_missing_from_federated_response()
    INV = get_stations(network="*", station="AN*", level="station", channel="*Z",
                       exclude_datacenter="IRIS")
    print(INV)
    INV = get_stations(network="*", station="AN*", level="station", channel="*Z",
                       include_datacenter=["GFZ", "RESIF"])
    print(INV)
