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
from obspy.clients.routers import RoutingClient
from .header import (DEFAULT_PARAMETERS)

class FederatorClient(RoutingClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.client.Client.__init__()`
    method.
    """

    def get_stations(self, starttime=None, endtime=None, startbefore=None,
                     startafter=None, endbefore=None, endafter=None,
                     network=None, station=None, location=None, channel=None,
                     minlatitude=None, maxlatitude=None, minlongitude=None,
                     maxlongitude=None, latitude=None, longitude=None,
                     minradius=None, maxradius=None, level=None,
                     includerestricted=None, includeavailability=None,
                     updatedafter=None, matchtimeseries=None, filename=None,
                     format="xml", **kwargs):
        """
        Query the station service of the FDSN client.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_stations()`
        method.
        """

        #send request to fedrator
        if "federator" not in self.services:
            msg = "The current client does not have a station service."
            raise ValueError(msg)

        locs = locals()
        setup_query_dict('station', locs, kwargs)

        fedresp = get_federator_index(kwargs)

        for i in fedresp:
            #make request to appropriate service
            #work on response
            #if to be written to file, do some hocus-pocus
        url = self._create_url_from_parameters(
            "federator", DEFAULT_PARAMETERS['station'], kwargs)

        if filename:
            # write to file
        else:
            return inventory

    def get_waveforms(self, network, station, location, channel, starttime,
                      endtime, quality=None, minimumlength=None,
                      longestonly=None, filename=None, attach_response=False,
                      **kwargs):
        """
        Query the dataselect service of the client.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_waveforms()`
        method.
        """
        if "federator" not in self.services:
            msg = "The current client does not have a federator service."
            raise ValueError(msg)

        locs = locals()
        setup_query_dict('dataselect', locs, kwargs)

        # Special location handling. Convert empty strings to "--".
        if "location" in kwargs and not kwargs["location"]:
            kwargs["location"] = "--"

        url = self._create_url_from_parameters(
            "federator", DEFAULT_PARAMETERS['dataselect'], kwargs)

        # Gzip not worth it for MiniSEED and most likely disabled for this
        # route in any case.
        data_stream = self._download(url, use_gzip=False)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            if attach_response:
                self._attach_responses(st)
            return st


    def get_waveforms_bulk(self, bulk, quality=None, minimumlength=None,
                           longestonly=None, filename=None,
                           attach_response=False, **kwargs):
        r"""
        Query the federator service of the client for dataselect style data. Bulk request.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_waveforms_bulk()`
        method.
        """
        if "federator" not in self.services:
            msg = "The current client does not have a federator service."
            raise ValueError(msg)

        arguments = OrderedDict(
            quality=quality,
            minimumlength=minimumlength,
            longestonly=longestonly
        )
        bulk = self._get_bulk_string(bulk, arguments)

        url = self._build_url("federator", "query")

        data_stream = self._download(url,
                                     data=bulk.encode('ascii', 'strict'))
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            if attach_response:
                self._attach_responses(st)
            return st

    def get_stations_bulk(self, bulk, level=None, includerestricted=None,
                          includeavailability=None, filename=None, **kwargs):
        r"""
        Query the station service of the client. Bulk request.

        For details see the :meth:`~obspy.clients.fdsn.client.Client.get_stations_bulk()`
        method.
        """
        if "federator" not in self.services:
            msg = "The current client does not have a federator service."
            raise ValueError(msg)

        arguments = OrderedDict(
            level=level,
            includerestriced=includerestricted,
            includeavailability=includeavailability
        )
        bulk = self._get_bulk_string(bulk, arguments)

        url = self._build_url("station", "query")

        data_stream = self._download(url,
                                     data=bulk.encode('ascii', 'strict'))
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
            return
        else:
            inv = obspy.read_inventory(data_stream, format="stationxml")
            data_stream.close()
            return inv
