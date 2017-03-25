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


from obspy.clients.fdsn import Client

class RoutingClient(Client):
    """
    This class serves as the user-facing layer for federated requests, and uses
    the Client's methods to communicate with each data center.  Where possible,
    it will also leverage the Client's methods to interact with the federator service.
    The federator's response is passed to the ResponseManager.
    The ResponseManager is then repeatedly queried for datacenter/url/bulk-request parcels,
    which are each routed to the appropriate datacenter using either Client.get_stations_bulk
    or Client.get_waveforms_bulk.  As each parcel of data is requested, the datacenter
    is displayed to console. As each request is fulfilled, then a summary of retrieved data
     is displayed.
    Upon completion, all the waveforms or inventory (station) data are returned in a
    single list /structure as though they were all requested from the same source. It
    appears that the existing obspy.core.inventory module will appropriately attribute
    each station to a datacenter as it is downloaded, but once they are merged into the
    same inventory object, individual station:datacenter identity is lost.
    """
    def get_federator_index(**kwargs):
        print("getting federator index")

        url = self._create_url_from_parameters(
            "federator", DEFAULT_PARAMETERS['station'], kwargs)
        # DEFAULT_PARAMETERS probably needs a federator version
        
        data_stream = self._download(url)
        data_stream.seek(0, 0)
        
        federator_index = read_federator_response(data_stream)
        data_stream.close()
        return federator_index

    def get_federator_index_bulk(**kwargs):
        r"""
        Query the station service of the client. Bulk request.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_stations_bulk()`
        method.
        """
        if "federator" not in self.services:
            msg = "The current client does not have a dataselect service."
            raise ValueError(msg)

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly,
            level=level,
            includerestriced=includerestricted,
            includeavailability=includeavailability
        )
        bulk = self._get_bulk_string(bulk, arguments)

        url = self._build_url("federator", "query")

        data_stream = self._download(url,
                                     data=bulk.encode('ascii', 'strict'))
        data_stream.seek(0, 0)
  
        federator_index = read_federator_response(data_stream)
        data_stream.close()
        return federator_index

class ResponseManager():
    """
    This class will wrap the response given by the federator.  Its primary purpose is to
    divide the response into parcels, each being a FederatorResponse containing the information
    required for a single request.
    Input would be the response from the federator, or a similar text file Output is a list
    of FederatorResponse objects
    """
