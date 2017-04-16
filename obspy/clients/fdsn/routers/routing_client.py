#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.
"""

from __future__ import print_function
import multiprocessing as mp
import requests
from obspy.clients.fdsn import Client

class RoutingClient(Client):
    """
    This class serves as the user-facing layer for federated requests, and uses
    the Client's methods to communicate with each data center.  Where possible,
    it will also leverage the Client's methods to interact with the federated catalog service.
    The federated catalog's response is passed to the ResponseManager.
    The ResponseManager is then repeatedly queried for provider/url/bulk-request parcels,
    which are each routed to the appropriate provider using either Client.get_stations_bulk
    or Client.get_waveforms_bulk.  As each parcel of data is requested, the provider
    is displayed to console. As each request is fulfilled, then a summary of retrieved data
     is displayed.
    Upon completion, all the waveforms or inventory (station) data are returned in a
    single list /structure as though they were all requested from the same source. It
    appears that the existing obspy.core.inventory module will appropriately attribute
    each station to a provider as it is downloaded, but once they are merged into the
    same inventory object, individual station:provider identity is lost.
    """
    pass

class ResponseManager(object):
    """
    This class will wrap the response given by routers.  Its primary purpose is to
    divide the response into parcels, each being an XYZResponse containing the information
    required for a single request.
    Input would be the response from the routing service, or a similar text file Output is a list
    of FederatedResponse objects

    """
    def __init__(self, textblock, include_provider=None, exclude_provider=None):
        '''
        :param textblock: input is returned text from routing service
        '''

        self.responses = self.parse_response(textblock)
        if include_provider or exclude_provider:
            self.responses = self.subset_requests(self.responses)

    def __iter__(self):
        return self.responses.__iter__()

    def __len__(self):
        return len(self.responses)

    def __str__(self):
        responsestr = "\n  ".join([str(x) for x in self.responses])
        towrite = "ResponseManager with " + str(len(self)) + " items:\n" +responsestr
        return towrite

    def parse_response(self, parameter_list):
        '''create a list of FederatedResponse objects, one for each provider in response'''
        raise NotImplementedError()

    def get_request(self, code, get_multiple=False):
        '''retrieve the response for a particular provider, by code
        if get_multiple is true, then a list will be returned with 0 or more
        FederatedResponse objects that meet the criteria.  otherwise, the first
        matching FederatedResponse will be returned

        Set up sample data:
        >>> fedresps = [FederatedResponse('IRIS'), FederatedResponse('SED'),
        ...             FederatedResponse('RESIF'), FederatedResponse('SED')]

        Test methods that return multiple FederatedResponse objects
        >>> get_datacenter_request(fedresps, 'SED')
        SED
        <BLANKLINE>
        >>> get_request(fedresps, 'SED', get_multiple=True)
        [SED
        , SED
        ]
        '''
        if get_multiple:
            return [resp for resp in self.responses if resp.code == code]
        for resp in self.responses:
            if resp.code == code:
                return resp
        return None

    def subset_requests(self, include_provider=None, exclude_provider=None):
        '''provide more flexibility by specifying which datacenters to include or exclude

        Set up sample data:
        >>> fedresps = [FederatedResponse('IRIS'), FederatedResponse('SED'),
        ...             FederatedResponse('RESIF')]

        >>> unch = subset_requests(fedresps)
        >>> print(".".join([dc.code for dc in unch]))
        IRIS.SED.RESIF

        Test methods that return multiple FederatedResponse objects
        >>> no_sed_v1 = subset_requests(fedresps, exclude_provider='SED')
        >>> no_sed_v2 = subset_requests(fedresps, include_provider=['IRIS', 'RESIF'])
        >>> print(".".join([dc.code for dc in no_sed_v1]))
        IRIS.RESIF
        >>> ".".join([x.code for x in no_sed_v1]) == ".".join([x.code for x in no_sed_v2])
        True

        Test methods that return single FederatedResponse (still in a container, though)
        >>> only_sed_v1 = subset_requests(fedresps, exclude_provider=['IRIS', 'RESIF'])
        >>> only_sed_v2 = subset_requests(fedresps, include_provider='SED')
        >>> print(".".join([dc.code for dc in only_sed_v1]))
        SED
        >>> ".".join([x.code for x in only_sed_v1]) == ".".join([x.code for x in only_sed_v2])
        True
        '''
        if include_provider:
            return [resp for resp in self.responses if resp.code in include_provider]
        elif exclude_provider:
            return [resp for resp in self.responses if resp.code not in exclude_provider]
        else:
            return self.responses

    def parallel_service_query(self, target_process, **kwargs):
        '''
        :param target_process: submit_waveform_request or submit_station_request
        '''

        output = mp.Queue()
        failed = mp.Queue()
        # Setup process for each provider
        processes = [mp.Process(target=req.target_process, args=(req, output, failed, kwargs))
                     for req in self]

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
    import doctest
    doctest.testmod(exclude_empty=True)