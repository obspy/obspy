#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.
"""

from __future__ import print_function
import queue
import sys
import logging
import multiprocessing as mp
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.routers.fedcatalog_parser import (RoutingResponse, FDSNBulkRequests)

# logging facilities swiped from mass_downloader.py
# Setup the logger.
logger = logging.getLogger("obspy.clients.fdsn.routing_client")
logger.setLevel(logging.DEBUG)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

class RoutingClient(object): #no longer inherits from Client.
    """
    This class serves as the user-facing layer for routing requests, and uses
    the Client's methods to communicate with each data center.  Where possible,
    it will also leverage the Client's methods to interact with the federated
    catalog service. The federated catalog's response is passed to the
    RoutingManager.

    The RoutingManager is then repeatedly queried for provider/url/bulk-request
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
                 exclude_provider=None, service_mappings=None, **kwargs):
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
        # super(RoutingClient, self).__init__(self)
        # do not pass initialization on to the Client. its series of checks
        # do this service no good, since this is a non-standard service, and
        # furthermore it is technically an IRIS service, not an FDSN service

        self.use_parallel = use_parallel
        self.args_to_clients = kwargs  # passed to clients as they are initialized
        self.include_provider = include_provider
        self.exclude_provider = exclude_provider
        self._service_mappings = service_mappings

    def __str__(self):
        return "RoutingClient object"

    def _request(self, client, service, route, output, passed, failed, **kwarg):
        """
        :type client: :class:`~obspy.clients.fdsn.Client` or similar
        :param client:
        :type service:
        :param service:
        :type route: 
        :param route:
        :type output: container accepting "put"
        :param output: place where retrieved data go
        :type failed: contdainer accepting "put"
        :param failed: place where list of unretrieved bulk request lines go
        :param filename:
        """
        raise NotImplementedError("define _request() in subclass")

    @property
    def query(self):
        """
        return query based on use_parallel preference
        :rtype: function
        :return: serial_query_machine or parallel_query_machine, with signatures:
                 self.xxxx_query_machine(routing_mgr, service, **kwargs)
                 and return a tuple (data,  FDSNBulkRequests of failed queries)
        """
        if self.use_parallel:
            return self.parallel_query_machine
        return self.serial_query_machine

    def serial_query_machine(self, routing_mgr, service, **kwargs):
        """
        query clients in series

        :type routing_mgr: :class:`~obspy.clients.fdsn.routers.RoutingManager`
        :param routing_mgr:
        :type service: str
        :param service:
        :type keep_unique: bool
        :param keep_unique: once an item of interest is retrieved, remove it
            from all subsequent requests
        :rtype: tuple (data, FDSNBulkRequests of failed queries)
        :return:
        """
        output = queue.Queue()
        passed = queue.Queue()
        failed = queue.Queue()
        all_retrieved = FDSNBulkRequests(None)

        for route in routing_mgr:
            if "keep_unique" in kwargs and kwargs["keep_unique"]:
                route.request_items.difference_update(all_retrieved)

            if self.exclude_provider and route.provider_id in self.exclude_provider:
                logger.info("skipping: " + route.provider_id +
                            " because it is in the exclude_provider list")
                continue
            if self.include_provider and route.provider_id not in self.include_provider:
                logger.info("skipping: " + route.provider_id +
                            " because it isn't in the include_provider list")
                continue
            if not route.request_items:
                logger.info("skipping: " + route.provider_id +
                            " because the retrieval list is empty.")
                continue

            try:
                client = Client(route.provider_id, self.args_to_clients)
                msg = "request to: {0}: {1} items.\n{2}".format(
                    route.provider_id, len(route),
                    routing_mgr.str_details(route.provider_id))
                logger.info("starting " + msg)
                # _request will put data into output and failed queues
                self._request(client=client, service=service,
                              route=route, output=output, passed=passed, failed=failed, **kwargs)
            except:
                failed.put(route.request_items) # FDSNBulkRequests
                raise

            # tricky part here: if anything has passed, we need to remove it
            # from the rest of the requests in the routing manager

            # TODO add description of request here.
        logger.info("all requests completed")
        data = None
        while not output.empty():
            if not data:
                data = output.get()
            else:
                data += output.get()
        
        while not passed.empty():
            # passed = routing_mgr.data_to_request(data)
            all_retrieved.update(passed.get())

        retry = None
        if not failed.empty():
            retry = failed.get()

        while not failed.empty():
            retry.update(failed.get()) # working with a set

        return data, all_retrieved, retry

    def parallel_query_machine(self, routing_mgr, service, **kwargs):
        """
        query clients in parallel

        :type routing_mgr: :class:`~obspy.clients.fdsn.routers.RoutingManager`
        :param routing_mgr:
        :type service: str
        :param service:
        :rtype: tuple tuple (data, FDSNBulkRequests of failed queries)
        :return:

        >>def echoer(**kwargs):
        """
        logging.warn("Parallel query requested, but will perform serial request anyway.")
        return self.serial_query_machine(routing_mgr=routing_mgr, service=service, **kwargs)
        output = mp.Queue()
        passed = mp.Queue()
        failed = mp.Queue()
        # Setup process for each provider
        processes = []
        msgs = []
        for route in routing_mgr:
            if self.exclude_provider and route.provider_id in self.exclude_provider:
                logger.info("skipping: " + route.provider_id)
                continue
            if self.include_provider and route.provider_id not in self.include_provider:
                logger.info("skipping: " + route.provider_id)
                continue
            try:
                client = Client(route.provider_id, self.args_to_clients)
            except:
                failed.put(route.request_items)
                raise
            else:
                p_kwargs = kwargs.copy()
                p_kwargs.update({'client':client, 'service':service,
                                'output':output, 'passed':passed, 'failed':failed,
                                'route':route})
                processes.append(mp.Process(target=self._request,
                                            name=route.provider_id,
                                            kwargs=p_kwargs))
                msgs.append("request to: {0}: {1} items.\n{2}".format(
                     route.provider_id, len(route), 
                     routing_mgr.str_details(route.provider_id)))

        # run
        for p, msg in zip(processes, msgs):
            logger.info("starting " + msg)

            p.start()

        logger.info("processing in parallel, with {0} concurrentish requests".format(len(processes)))
        # exit completed processes
        for p, msg in zip(processes, msgs):
            logger.info("waiting on " + msg)
            p.join()
        logger.info("all processes completed.")

        data = None
        while not output.empty():
            tmp = output.get()
            if not data:
                data = tmp
            else:
                data += tmp

        passed = routing_mgr.data_to_request(data)

        if not failed.empty():
            retry = failed.get()

        while not failed.empty():
            retry.update(failed.get()) # working with a set

        return data, passed, retry

class RoutingManager(object):
    """
    This class will wrap the response given by routers.  Its primary purpose is
    to divide the response into parcels, each being an XYZResponse containing
    the information required for a single request.

    Input would be the response from the routing service, or a similar text file
    Output is a list of RoutingResponse objects
    """
    def __init__(self, textblock, provider_details=None):
        """
        initialize a RoutingManager object

        :type textblock: str or container of RoutingResponse
        :param textblock: text retrieved from routing service
        # :type provider_details: must have member .names, .__str__, and .pretty
        # :param provider_details:
        """
        self.routes = []
        self.provider_details = provider_details
        if isinstance(textblock, str):
            self.routes = self.parse_routing(textblock)
        elif isinstance(textblock, RoutingResponse):
            self.routes = [textblock]
        elif isinstance(textblock, (tuple, list)):
            self.routes = [v for v in textblock if isinstance(v, RoutingResponse)]

    def __iter__(self):
        return self.routes.__iter__()

    def __len__(self):
        return len(self.routes)

    def __str__(self):
        if not self.routes:
            return "Empty " + type(self).__name__
        responsestr = "\n".join([str(x) for x in self.routes])
        towrite = type(self).__name__ + " with " + str(len(self)) + " items:\n" +responsestr
        return towrite

    def data_to_request(self, data):
        raise NotImplementedError

    def provider_ids(self):
        """
        return the provider id's for all retrieved routes
        """
        return [route.provider_id for route in self.routes]

    def str_details(self, provider_id):
        if self.provider_details:
            return self.provider_details.pretty(provider_id)
        return ""

    def parse_routing(self, parameter_list):
        """
        create a list of RoutingResponse objects, one for each provider in response

        :type parameter_list:
        :param parameter_list:
        :rtype:
        :return:
        """
        raise NotImplementedError()

    def get_route(self, provider_id, get_multiple=False):
        """
        retrieve the response for a particular provider, by provider_id

        Set up sample data:
        >>> sed2 = RoutingResponse('SED', raw_requests = ["some_request"])
        >>> fedresps = [RoutingResponse('IRIS'), RoutingResponse('SED'),
        ...             RoutingResponse('RESIF'), sed2]
        >>> fedresps = RoutingManager(fedresps)

        Test methods that return multiple RoutingResponse objects
        >>> print(str(fedresps.get_route('SED')))
        SED, with 0 items
        >>> ml = fedresps.get_route('SED', get_multiple=True)
        >>> print ([str(x) for x in ml])
        ['SED, with 0 items', 'SED, with 1 item']

        :type provider_id: str
        :param provider_id: recognized key string for recognized server. see
        :mod:`~obspy.clients.fdsn.client` for a list
        :type get_multiple: bool
        :param get_multiple: determines whether to return a single (first
        matching) RoutingResponse or a list of all matching routes
        :rtype: :class:`~obspy.clients.fdsn.routers.RoutingResponse` or list
        :return:
        """
        if get_multiple:
            return [route for route in self.routes if route.provider_id == provider_id]
        for route in self.routes:
            if route.provider_id == provider_id:
                return route
        return None

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True, verbose=True)
