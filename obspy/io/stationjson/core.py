#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions dealing with reading and writing StationJSON (incomplete).

:copyright:
    Mathijs Koymans (koymans@knmi.nl, 2016)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import json

import obspy
from obspy import UTCDateTime


# Define some constants for writing StationXML files.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "https://www.obspy.org"
SCHEMA_VERSION = "1.0"

class JSONXMLEncoder(json.JSONEncoder):
    """
    Custom encoder capable of dealing with ObsPy types.
    """
    def default(self, obj):
        if isinstance(obj, UTCDateTime):
            return str(obj)
        else:
            return super(JSONXMLEncoder, self).default(obj)


def _write_stationjson(inventory, file_or_file_object, validate=False,
                   **kwargs):
    """
    Writes ObsPy inventory to stationJSON
    Currently implemented until the channel level
    See: https://github.com/chad-iris/StationJSON

    :type inventory: :class:`~obspy.core.inventory.Inventory`
    :param inventory: The inventory instance to be written.
    :param file_or_file_object: The file or file-like object to be written to.
    :param validate: If True, the created document will be validated with the
        StationXML schema before being written. Useful for debugging or if you
        don't trust ObsPy. Defaults to False.
    """

    jsonXML = {}
    jsonXML['network'] = [_write_network(network) for network in
                          inventory.networks]

    with open(file_or_file_object, 'w') as file:
      file.write(unicode(json.dumps(jsonXML, cls=JSONXMLEncoder, indent=4)))

def _write_network(network):
    """
    Helper function to write network subinventories
    """
    networkObject = {}
    networkObject['code'] = network.code
    networkObject['startTime'] = network.start_date
    networkObject['endTime'] = network.end_date
    networkObject['description'] = network.description
    networkObject['restrictedStatus'] = network.restricted_status
    networkObject['totalNumberStations'] = network.total_number_of_stations
    networkObject['selectedNumberStations'] = network.\
                                              selected_number_of_stations

    networkObject['stations'] = [_write_station(station) for station in
                                 network.stations]

    return networkObject

def _write_site(site):
    """
    Helper function to write station site
    """
    siteObject = {}
    siteObject['name'] = site.name
    siteObject['description'] = site.description

    return siteObject


def _write_station(station):
    """
    Helper function to write station subinventories
    """
    stationObject = {}
    stationObject['code'] = station.code;
    stationObject['startTime'] = station.start_date
    stationObject['endTime'] = station.end_date
    stationObject['latitude'] = station.latitude
    stationObject['longitude'] = station.longitude
    stationObject['elevation'] = station.elevation

    stationObject['site'] = _write_site(station.site)

    stationObject['restrictedStatus'] = station.restricted_status
    stationObject['totalNumberChannels'] = station.total_number_of_channels
    stationObject['selectedNumberChannels'] = station.\
                                              selected_number_of_channels
    stationObject['channels'] = [_write_channel(channel) for channel in
                                 station.channels] 

    return stationObject


def _write_data_availability(data_availability):
    """
    Helper function to write data availability
    Currently not implemented
    """
    return None


def _write_channel(channel):
    """
    Helper function to write channel subinventories
    """
    channelObject = {}

    channelObject['code'] = channel.code
    channelObject['location'] = channel.location_code
    channelObject['startTime'] = channel.start_date
    channelObject['endTime'] = channel.end_date
    channelObject['latitude'] = channel.latitude
    channelObject['longitude'] = channel.longitude
    channelObject['elevation'] = channel.elevation
    channelObject['depth'] = channel.depth
    channelObject['azimuth'] = channel.azimuth
    channelObject['dip'] = channel.dip
    channelObject['sensorDescription'] = channel.sensor.description
    channelObject['sampleRate'] = channel.sample_rate

    # Take channel scale (gain?) from first resp stage
    first_stage = channel.response.response_stages[0]
    channelObject['scale'] = first_stage.normalization_factor
    channelObject['scaleUnits'] = first_stage.input_units
    channelObject['scaleFrequency'] = first_stage.normalization_frequency

    channelObject['restrictedStatus'] = channel.restricted_status

    channelObject['dataAvailability'] = _write_data_availability(
                                            channel.data_availability)

    return channelObject


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
