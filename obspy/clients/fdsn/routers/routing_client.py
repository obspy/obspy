#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the Routing client and manager for ObsPy.  These classes
are intended to be subclassed.

:class:`~obspy.clients.fdsn.routers.RoutingClient`
This class contains the genericized methods used to query for information.
In general, it will first query the routing service, and depending upon
the response, query the specific services from each data provider.

:class:`~obspy.clients.fdsn.routers.RoutingManager`
This class will wrap the response given by routers in order to provide
an iterabler interface to the response parcels. Each parcel is a subclass of
:class:`~obspy.clients.fdsn.routers.RoutingResponse` and is usable to request
data from the provider.

:Note: Although it was originally intended that RoutingClient inherited from
:class:`~obspy.clients.fdsn.Client`, the Fedcatalog service is technically not
a federated service (yet).  It is running on a different path (../irisws/..),
and fails the many assumptions built into the Client class.

:var ROUTING_LOGGER: logger associated with :mod:`obspy.clients.fdsn.router`
:type ROUTING_LOGGER: logging.getLogger()

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

from future.utils import string_types

import logging
import multiprocessing as mp
try:
    import Queue as queue
except ImportError:
    import queue

from obspy.clients.fdsn import Client
from obspy.clients.fdsn.routers.fedcatalog_parser import (RoutingResponse,
                                                          FDSNBulkRequests)

_DISABLE_PARALLEL_REQESTS = True


def set_up_logger():
    """
    set up a logger associated with the `obspy.clients.fdsn.routers`
    """
    # logging facilities swiped from mass_downloader.py
    # Setup the logger.
    logger = logging.getLogger("obspy.clients.fdsn.routing_client")
    logger.setLevel(logging.DEBUG)
    # Prevent propagating to higher loggers.
    logger.propagate = 0
    # Console log handler.
    clh = logging.StreamHandler()
    clh.setLevel(logging.INFO)
    # Add formatter
    _format = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
    formatter = logging.Formatter(_format)
    clh.setFormatter(formatter)
    logger.addHandler(clh)
    return logger


ROUTING_LOGGER = set_up_logger()


class RoutingClient(object):
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
    source.
    """

    def __init__(self, use_parallel=False, include_provider=None,
                 exclude_provider=None, **kwargs):
        """
        Initialize a RoutingClient

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

        self.use_parallel = use_parallel and not _DISABLE_PARALLEL_REQESTS
        self.args_to_clients = kwargs  # passed to clients during intialization
        if isinstance(include_provider, string_types):
            self.include_provider = (include_provider,)
        else:
            self.include_provider = include_provider

        if isinstance(exclude_provider, string_types):
            self.exclude_provider = (exclude_provider,)
        else:
            self.exclude_provider = exclude_provider

    def __str__(self):
        part1 = "RoutingClient\n"
        part2 = "  request-method: %s\n" % \
                ("parallel" if self.use_parallel else "serial")
        if self.include_provider:
            return part1 + part2 + "  include: %s" % \
                (",".join(self.include_provider))
        if self.exclude_provider:
            return part1 + part2 + "  exclude: %s" %\
                (",".join(self.exclude_provider))
        return part1 + part2

    def _request(self, client, service, route, output, passed,
                 failed, **kwarg):
        """
        ABSTRACT retrieve data from one service endpoint

        :type client: :class:`~obspy.clients.fdsn.Client` or similar
        :param client: The client where this specific request will be routed
        :type service: str
        :param service: name of service to use. This is defined by the subclass
        :type route: something like
            :class:`~obspy.clients.fdsn.routers.FederatedRoute`
        :param route: contains details used to route one set of requests.
        :type output: container accepting "put", like a Queue
        :param output: place where retrieved data go
        :type failed: container accepting "put", like a Queue
        :param failed: place where list of unretrieved bulk request lines go
        :type filename: str or open file handle
        :param filename: filename, or file-like object for writing data out.
        It is possible/probable that the routing routines will prepend some
        provider code to any file name
        """
        raise NotImplementedError("define _request() in subclass")

    def _skip_provider_for_route(self, route):
        """
        helper function

        :type route: route to particular provider
        :param route:
        :rtype: bool
        :return: whether this provider should be skipped

        >>> item = ["* * * * * *"]
        >>> irisRoute = RoutingResponse("IRIS", item)
        >>> sedRoute = RoutingResponse("SED", item)
        >>> rc = RoutingClient(include_provider="IRIS")
        >>> rc._skip_provider_for_route(irisRoute)
        False
        >>> rc._skip_provider_for_route(sedRoute)
        True
        >>> rc = RoutingClient(exclude_provider="IRIS")
        >>> rc._skip_provider_for_route(irisRoute)
        True
        >>> rc._skip_provider_for_route(sedRoute)
        False
        """
        if self.exclude_provider:
            if route.provider_id in self.exclude_provider:
                return True

        if self.include_provider:
            if route.provider_id not in self.include_provider:
                return True

        if not route.request_items:
            return True
        return False

    @property
    def query(self):
        """
        return query method based on user preference (parallel or serial)

        Two query methods currently exist:
        :func:`serial_query_machine`
        :func:`parallel_query_machine`

        These contain the logic that performs the full query, from
        getting a response from the router through requesting data from each
        provider, to re-requesting as necessary.

        :rtype: function
        :returns: serial_ or parallel_ query_machine, with signatures:
                 self.xxxx_query_machine(routing_mgr, service, **kwargs) and
                 return a tuple (data, successful_requests, failed_requests)
        """
        if self.use_parallel:
            return self.parallel_query_machine
        return self.serial_query_machine

    def serial_query_machine(self, routing_mgr, service, keep_unique=False,
                             **kwargs):
        """
        retrieve data from clients, one after the other, based on routes held
        by the routing manager

        Each route held by the routing manager would contain the details needed
        to create a connection to a data provider, along with request details.

        This routine loops through each route held by the manager, opens a
        connection to the clients of interest, requests data from the selected
        service, and then aggregates the resulting data.

        :type routing_mgr: :class:`~obspy.clients.fdsn.routers.RoutingManager`
        :param routing_mgr: contains a collection of Routes
        :type service: str
        :param service:
        :type keep_unique: bool
        :param keep_unique: once an item of interest is retrieved, remove it
            from all subsequent requests
        :rtype: tuple (Inventory or Stream, FDSNBulkRequests, FDSNBulkRequests)
        :returns: (data, successful_requests, failed_requests)

        :Note: This cannot track data requests that are streamed to a file,
        only those that remain in memory.
        """
        output_q = queue.Queue()
        passed_q = queue.Queue()
        failed_q = queue.Queue()
        successful_requests = FDSNBulkRequests(None)  # Note, no generic vers

        for route in routing_mgr:
            if self._skip_provider_for_route(route):
                continue

            if keep_unique and successful_requests:
                route.request_items.difference_update(successful_requests)

            try:
                client = Client(route.provider_id, **self.args_to_clients)

                msg = "request to: {0}: {1} items.\n{2}".format(
                    route.provider_id, len(route),
                    routing_mgr.str_details(route.provider_id))
                ROUTING_LOGGER.info("starting " + msg)
                # _request will put data into output_q, passed_q, failed_q
                self._request(client=client, service=service,
                              route=route, output=output_q, passed=passed_q,
                              failed=failed_q, **kwargs)
            except:
                failed_q.put(route.request_items)  # FDSNBulkRequests
                raise

        ROUTING_LOGGER.info("all requests completed")

        data = None
        while not output_q.empty():
            if not data:
                data = output_q.get()
            else:
                data += output_q.get()

        while not passed_q.empty():
            successful_requests.update(passed_q.get())

        failed_requests = None
        if not failed_q.empty():
            failed_requests = failed_q.get()

        while not failed_q.empty():
            failed_requests.update(failed_q.get())  # working with a set

        return data, successful_requests, failed_requests

    def parallel_query_machine(self, routing_mgr, service, keep_unique=False,
                               **kwargs):
        """
        query clients in parallel

        :type routing_mgr: :class:`~obspy.clients.fdsn.routers.RoutingManager`
        :param routing_mgr: contains a collection of Routes
        :type service: str
        :param service:
        :type keep_unique: bool
        :param keep_unique: once an item of interest is retrieved, remove it
            from all subsequent requests
        :rtype: tuple (Inventory or Stream, FDSNBulkRequests, FDSNBulkRequests)
        :returns: (data, successful_requests, failed_requests)

        :Note: This cannot track data requests that are streamed to a file,
        only those that remain in memory.
        """

        timeout = 120  # seconds
        output_q = mp.Queue()
        passed_q = mp.Queue()
        failed_q = mp.Queue()
        successful_requests = FDSNBulkRequests(None)  # Note, no generic vers

        # Setup process for each provider
        processes = []
        msgs = []
        for route in routing_mgr:
            if self._skip_provider_for_route(route):
                continue

            try:
                client = Client(route.provider_id, **self.args_to_clients)
                p_kwargs = kwargs.copy()
                p_kwargs.update({'client': client, 'service': service,
                                 'output': output_q, 'passed': passed_q,
                                 'failed': failed_q, 'route': route})
                processes.append(mp.Process(target=self._request,
                                            name=route.provider_id,
                                            kwargs=p_kwargs))
                msgs.append("request to: {0}: {1} items.\n{2}".format(
                    route.provider_id, len(route),
                    routing_mgr.str_details(route.provider_id)))
            except Exception as err:
                failed_q.put(route.request_items)
                ROUTING_LOGGER.exception("error: %s", err)

        # run
        for proc, msg in zip(processes, msgs):
            ROUTING_LOGGER.info("starting " + msg)
            proc.start()

        ROUTING_LOGGER.info(
            "processing in parallel, with %d concurrentish requests",
            len(processes))
        # exit completed processes
        for proc, msg in zip(processes, msgs):
            ROUTING_LOGGER.info("waiting on %s", msg)
            proc.join(timeout)
            ROUTING_LOGGER.debug(
                "finished with %s, still have %d to go", msg,
                sum(x.exitcode is not None for x in processes))
        ROUTING_LOGGER.info("all processes completed.")

        data = None
        while not output_q.empty():
            ROUTING_LOGGER.debug("getting 'output' queue")
            if not data:
                data = output_q.get()
            else:
                data += output_q.get()

        while not passed_q.empty():
            ROUTING_LOGGER.debug("getting 'passed' queue")
            successful_requests.update(passed_q.get())

        failed_requests = None
        if not failed_q.empty():
            failed_requests = failed_q.get()

        while not failed_q.empty():
            failed_requests.update(failed_q.get())  # working with a set

        return data, successful_requests, failed_requests


class RoutingManager(object):
    """
    This class will wrap the response given by a router.  Its primary purpose
    is to divide the response into parcels, each being a route that contains
    the information required for a single request.

    Input would be the response from the routing service, or a similar file
    Output is a list of RoutingResponse objects
    """
    def __init__(self, textblock, provider_details=None):
        """
        initializer for RoutingManager

        :type textblock: str, RoutingResponse, or container of RoutingResponse
        :param textblock: text retrieved from routing service
        :type provider_details: must have member .names, .__str__, and .pretty
        :param provider_details:
        """
        self.routes = []
        self.provider_details = provider_details
        if isinstance(textblock, string_types):
            self.routes = self.parse_routing(textblock)
        elif isinstance(textblock, RoutingResponse):
            self.routes = [textblock]
        elif isinstance(textblock, (tuple, list)):
            self.routes = [
                v for v in textblock if isinstance(v, RoutingResponse)]

    def __iter__(self):
        return self.routes.__iter__()

    def __len__(self):
        return len(self.routes)

    def __str__(self):
        if not self.routes:
            return "Empty " + type(self).__name__
        responsestr = "\n".join([str(x) for x in self.routes])
        towrite = type(self).__name__ + " with " + str(len(self)) +\
            " items:\n" + responsestr
        return towrite

    def add_routes(self, route):
        """
        Add a route (RoutingResponse) to the routing manger
        """
        if isinstance(route, RoutingResponse):
            self.routes.append(route)
        elif isinstance(route, (tuple, list)) and len(route)\
                and isinstance(route[0].RoutingResponse):
            self.routes.extend(route)
        elif isinstance(route, RoutingManager):
            self.routes.extend(route.routes)
        else:
            NotImplementedError(
                "Do not know how to add {} to RoutingManager".
                format(route.__class__.__name__))

    def data_to_request(self, data):
        """
        Abstract : convert data to a request
        """
        raise NotImplementedError

    def provider_ids(self):
        """
        return the provider id's for all retrieved routes
        """
        return [route.provider_id for route in self.routes]

    def str_details(self, provider_id):
        """
        provide formatted details for one provider
        """
        if self.provider_details:
            return self.provider_details.pretty(provider_id)
        return ""

    def parse_routing(self, parameter_list):
        """
        ABSTRACT create RoutingResponses, one for each provider in response

        :type parameter_list:
        :param parameter_list:
        :rtype:
        :returns: list of RoutingResponse objects
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
        :returns:
        """
        if get_multiple:
            return [route for route in self.routes
                    if route.provider_id == provider_id]
        for route in self.routes:
            if route.provider_id == provider_id:
                return route
        return None


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
