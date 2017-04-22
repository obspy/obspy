#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.
"""

from __future__ import print_function
import queue
import multiprocessing as mp
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.routers.fedcatalog_parser import RoutingResponse

class RoutingClient(Client):
    """
    This class serves as the user-facing layer for routing requests, and uses
    the Client's methods to communicate with each data center.  Where possible,
    it will also leverage the Client's methods to interact with the federated
    catalog service. The federated catalog's response is passed to the
    ResponseManager.

    The ResponseManager is then repeatedly queried for provider/url/bulk-request
    parcels, which are each routed to the appropriate provider using either
    Client.get_stations_bulk or Client.get_waveforms_bulk.  As each parcel of
    data is requested, the provider is displayed to console. As each request is
    fulfilled, then a summary of retrieved data is displayed.

    Upon completion, all the waveforms or inventory (station) data are returned
    in a single list /structure as though they were all requested from the same
    source. It appears that the existing obspy.core.inventory module will
    appropriately attribute each station to a provider as it is downloaded, but
    once they are merged into the same inventory object, individual
    station:provider identity is lost.
    """
    def __init__(self, use_parallel=False, include_provider=None,
                 exclude_provider=None, **kwargs):
        """
        :type use_parallel: boolean
        :param use_parallel: determines whether clients will be polled in in
        parallel or in series.  If the Client appears to hang during a request,
        set this to False.
        :type exclude_provider: str or list of str
        :param exclude_provider: Get no data from these providers
        :type include_provider: str or list of str
        :param include_provider: Get data only from these providers
        :param **kwargs: additional arguments are passed along to each instance
        of the new client
        """
        # do not pass initialization on to the Client. its series of checks
        # do this service no good, since this is a non-standard service, and
        # furthermore it is technically an IRIS service, not an FDSN service

        self.use_parallel = use_parallel
        self.args_to_clients = kwargs  # passed to clients as they are initialized
        self.include_provider = include_provider
        self.exclude_provider = exclude_provider

    @property
    def query(self):
        """
        return query based on use_parallel preference
        """
        if self.use_parallel:
            return self.parallel_query_machine
        return self.serial_query_machine

    def serial_query_machine(self, request_mgr, service, **kwargs):
        """
        query clients in series

        :type request_mgr: :class:`~obspy.clients.fdsn.routers.ResponseManager`
        :param request_mgr:
        :type service: str
        :param service:
        :rtype: tuple (data, list of failed queries)
        :return:
        """
        output = queue.Queue()
        failed = queue.Queue()

        for req in request_mgr:  #each FederatedResponse() / RoutingResponse()
            if self.exclude_provider and req.code in self.exclude_provider:
                continue
            if self.include_provider and req.code not in self.include_provider:
                continue
            try:
                client = Client(req.code, self.args_to_clients)
                self.request_something(client, service, req, output, failed, **kwargs)
            except:
                failed.put(req.request_lines)
                raise

        data = None
        while not output.empty():
            if not data:
                data = output.get()
            else:
                data += output.get()

        retry = []
        while not failed.empty():
            retry.extend(failed.get())
        if retry:
            retry = '\n'.join(retry)
        return data, retry

    def parallel_query_machine(self, request_mgr, service, **kwargs):
        """
        query clients in parallel

        :type request_mgr: :class:`~obspy.clients.fdsn.routers.ResponseManager`
        :param request_mgr:
        :type service: str
        :param service:
        :rtype: tuple (data, list of failed queries)
        :return:
        """

        output = mp.Queue()
        failed = mp.Queue()
        # Setup process for each provider
        processes = []
        for req in request_mgr:
            if self.exclude_provider and req.code in self.exclude_provider:
                continue
            if self.include_provider and req.code not in self.include_provider:
                continue
            try:
                client = Client(req.code, self.args_to_clients)
            except:
                failed.put(req.request_lines)
                raise
            else:
                args = (client, service, req, output, failed)
                processes.append(mp.Process(target=self.request_something,
                                            name=req.code,
                                            args=args, 
                                            kwargs=kwargs))

        # run
        for p in processes:
            p.start()

        # exit completed processes
        for p in processes:
            p.join()

        data = None
        while not output.empty():
            if not data:
                data = output.get()
            else:
                data += output.get()

        retry = []
        while not failed.empty():
            retry.extend(failed.get())

        if retry:
            retry = '\n'.join(retry)
        return data, retry

class ResponseManager(object):
    """
    This class will wrap the response given by routers.  Its primary purpose is
    to divide the response into parcels, each being an XYZResponse containing
    the information required for a single request.

    Input would be the response from the routing service, or a similar text file
    Output is a list of RoutingResponse objects
    """
    def __init__(self, textblock):
        """
        initialize a ResponseManager object

        :type textblock: str or container of RoutingResponse
        :param textblock: text retrieved from routing service
        """
        self.responses = []
        # print("init responsemanager: incoming text is a " + type(textblock).__name__)
        if isinstance(textblock, str):
            self.responses = self.parse_response(textblock)
        elif isinstance(textblock, RoutingResponse):
            self.responses = [textblock]
        elif isinstance(textblock, (tuple, list)):
            self.responses = [v for v in textblock if isinstance(v, RoutingResponse)]

    def __iter__(self):
        return self.responses.__iter__()

    def __len__(self):
        return len(self.responses)

    def __str__(self):
        if not self.responses:
            return "Empty " + type(self).__name__
        responsestr = "\n".join([str(x) for x in self.responses])
        towrite = type(self).__name__ + " with " + str(len(self)) + " items:\n" +responsestr
        return towrite

    def parse_response(self, parameter_list):
        """
        create a list of RoutingResponse objects, one for each provider in response

        :type parameter_list:
        :param parameter_list:
        :rtype:
        :return:
        """
        raise NotImplementedError()

    def get_routing_response(self, code, get_multiple=False):
        """
        retrieve the response for a particular provider, by code

        Set up sample data:
        >>> sed2 = RoutingResponse('SED', raw_requests = ["some_request"])
        >>> fedresps = [RoutingResponse('IRIS'), RoutingResponse('SED'),
        ...             RoutingResponse('RESIF'), sed2]
        >>> fedresps = ResponseManager(fedresps)

        Test methods that return multiple RoutingResponse objects
        >>> print(str(fedresps.get_routing_response('SED')))
        SED, with 0 lines
        >>> ml = fedresps.get_routing_response('SED', get_multiple=True)
        >>> print ([str(x) for x in ml])
        ['SED, with 0 lines', 'SED, with 1 line']

        :type code: str
        :param code: recognized key string for recognized server. see
        :mod:`~obspy.clients.fdsn.client` for a list
        :type get_multiple: bool
        :param get_multiple: determines whether to return a single (first
        matching) RoutingResponse or a list of all matching responses
        :rtype: :class:`~obspy.clients.fdsn.routers.RoutingResponse` or list
        :return:
        """
        if get_multiple:
            return [resp for resp in self.responses if resp.code == code]
        for resp in self.responses:
            if resp.code == code:
                return resp
        return None

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
