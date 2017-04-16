#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    ?
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import print_function
import sys
from obspy.clients.fdsn.routers import RoutingClient
from requests.exceptions import HTTPError, Timeout, ConnectionError, TooManyRedirects

class FederatedClient(RoutingClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    """

    def get_waveforms_bulk(self, exclude_datacenter=None, include_datacenter=None,
                           includeoverlaps=False, bulk=None, **kwargs):
        '''
        :param exclude_datacenter: Avoids getting data from datacenters with specified ID code(s)
        :param include_datacenter: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        assert bulk, "No bulk request povided"
        # send request to the FedCatalog
        try:
            resp = query_fedcatalog("dataselect", bulk=bulk)
        except ConnectionError:
            pass
        except HTTPError:
            pass
        except Timeout:
            pass

        # parse the reply into an iterable object
        frm = FederatedResponseManager(resp.text, include_datacenter=include_datacenter,
                                       exclude_datacenter=exclude_datacenter)

        inv, retry = parallel_service_query(submit_waveform_request, frm, kwargs=kwargs)
        return inv

    def get_waveforms(self, exclude_datacenter=None, include_datacenter=None,
                      includeoverlaps=False, **kwargs):
        '''
        :param exclude_datacenter: Avoids getting data from datacenters with specified ID code(s)
        :param include_datacenter: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        # send request to the FedCatalog
        resp = query_fedcatalog("dataselect", params=kwargs)
        # parse the reply into an iterable object
        frm = FederatedResponseManager(resp.text, include_datacenter=include_datacenter,
                                       exclude_datacenter=exclude_datacenter)

        inv, retry = parallel_service_query(submit_waveform_request, frm, kwargs=kwargs)
        return inv

    def get_stations_bulk(self, exclude_datacenter=None, include_datacenter=None,
                          includeoverlaps=False, bulk=None, **kwargs):
        '''
        :param exclude_datacenter: Avoids getting data from datacenters with specified ID code(s)
        :param include_datacenter: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations_bulk
        '''

        assert bulk, "No bulk request provided"
        # send request to the FedCatalog
        resp = query_fedcatalog_bulk("station", bulk=bulk)
        # parse the reply into an iterable object
        frm = FederatedResponseManager(resp.text, include_datacenter=include_datacenter,
                                       exclude_datacenter=exclude_datacenter)

        inv, retry = parallel_service_query(submit_station_request, frm, kwargs=kwargs)
        return inv

    def get_stations(self, exclude_datacenter=None, include_datacenter=None,
                     includeoverlaps=False, **kwargs):
        '''
        This will be the original request for the federated station service
        :param exclude_datacenter: Avoids getting data from datacenters with specified ID code(s)
        :param include_datacenter: limit retrieved data to  datacenters with specified ID code(s)
        :param includeoverlaps: For now, simply leave this False. It will confuse the program.
        other parameters as seen in Client.get_stations

        Warning: If you specify "include_datacenter", you may miss data that the datacenter holds,
                but isn't the primary source
        Warning: If you specify "exclude_datacenter", you may miss data for which that datacenter
                is the primary source

        >>> from requests.packages.urllib3.exceptions import InsecureRequestWarning
        >>> requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        >>> INV = get_stations(network="A?", station="OK*", channel="?HZ", level="station",
        ...                    endtime="2016-12-31")
        '''

        try:
            resp = query_fedcatalog("station", params=kwargs)
        except ConnectionError:
            print("Problem connecting to fedcatalog service", file=sys.stderr)
        except HTTPError:
            print("Error downloading data from fedcatalog service: " + str(resp.status_code), file=sys.stderr)
        except Timeout:
            print("Timeout while waiting for a response from the fedcatalog service")
            pass
        
        frm = FederatedResponseManager(resp.text, include_datacenter=include_datacenter,
                                       exclude_datacenter=exclude_datacenter)

        inv, retry = parallel_service_query(submit_station_request, frm, kwargs=kwargs)

        # level = kwargs["level"]
        # successful, failed = request_exists_in_inventory(inv, datac.request_lines, level)
        # all_inv_set = INV_CONVERTER[level](inv) if not all_inv else all_inv_set.union(INV_CONVERTER[level](inv))
        # resubmit unsuccessful requests to fedservice with includeoverlaps
        return inv