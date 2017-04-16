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
import multiprocessing as mp
import requests
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.clients.fdsn.routers.fedcatalog_response_parser import (parse_federated_response,
                                                                   FederatedResponse,
                                                                   filter_requests)
#from obspy.clients.fdsn.routers import parse_federated_response, FederatedResponse

# inv2x_set(inv) will be used to quickly decide what exists and what doesn't
def inv2channel_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code + "." + s.code + "." +
            c.location_code + "." + c.code
            for n in inv for s in n for c in s}

def inv2station_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code + "."+ s.code for n in inv for s in n}

def inv2network_set(inv):
    'return a set containing string representations of an inv object'
    return {n.code for n in inv}

def req2network_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net)
    return req_str

def req2station_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net + "." + sta)
    return req_str

def req2channel_set(req):
    'return a set containing string representations of an request line'
    req_str = set()
    for line in req:
        (net, sta, loc, cha, startt, endt) = line.split()
        req_str.add(net + "." + sta + "." + loc + "." + cha)
    return req_str

#converters used to make comparisons between inventory items and requests
INV_CONVERTER = {"channel":inv2channel_set, "station":inv2station_set, "network":inv2network_set}
REQ_CONVERTER = {"channel":req2channel_set, "station":req2station_set, "network":req2network_set}

def request_exists_in_inventory(inv, requests, level):
    'compare inventory to requests'
    # does not account for timespans, only for existence net-sta-loc-cha matches
    inv_set = INV_CONVERTER[level](inv)
    req_set = REQ_CONVERTER[level](requests)
    members = req_set.intersection(inv_set)
    non_members = req_set.difference(inv_set)
    return members, non_members



def query_fedcatalog(targetservice, params=None, bulk=None):
    ''' send request to fedcatalog service, return ResponseManager object'''
    # send request to the FedCatalog
    url = 'https://service.iris.edu/irisws/fedcatalog/1/'
    params = kwargs.copy()
    params["targetservice"] = targetservice
    if params:
        resp = requests.get(url + "query", params=params, verify=False)
    elif bulk:
        resp = requests.get(url + "query", data=bulk, verify=False)
    else:
        #TODO throw an error
        pass
    resp.raise_for_status()
    return resp

def submit_waveform_request(request, output, failed, **kwargs):
    try:
        client = request.client()
        print("requesting data from:" + request.code + " : " + request.services["DATACENTER"])
        data = client.get_waveforms_bulk(bulk=request.text("DATASELECTSERVICE"), kwargs=kwargs)
    except FDSNNoDataException as ex:
        failed.put(request.request_lines)
    except Exception as ex: #except expression as identifier:
        failed.put(request.request_lines)
        print(code + " error!", file=sys.stderr)
    else:
        print(data)
        output.put(data)

def submit_station_request(request, output, failed, **kwargs):
    try:
        client = request.client()
        print("requesting data from:" + request.code + " : " + request.services["DATACENTER"])
        data = client.get_stations_bulk(bulk=request.text("STATIONSERVICE"), kwargs=kwargs)
    except FDSNNoDataException as ex:
        failed.put(request.request_lines)
    except Exception as ex: #except expression as identifier:
        failed.put(request.request_lines)
        print(code + " error!", file=sys.stderr)
    else:
        print(data)
        output.put(data)

def parallel_service_query(target_process, fed_resp_mgr, **kwargs):
    '''
    '''

    output = mp.Queue()
    failed = mp.Queue()
    # Setup process for each datacenter
    processes = [mp.Process(target=target_process, args=(req, output, failed, kwargs))
                 for req in fed_resp_mgr]

    # run
    for p in processes:
        p.start()

    # exit completed processes
    for p in processes:
        p.join()

    data = output.get() if not output.empty() else None
    while not output.empty():
        data += output.get()

    retry = failed.get() if not failed.empty() else None
    while not failed.empty():
        retry.extend(failed.get())
    retry = '\n'.join(retry)
    return data, retry



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
