#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parsing of the text files from the FDSN station web services.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from future import standard_library
with standard_library.hooks():
    import collections

import csv
import io
import re
import warnings

import obspy
from obspy import UTCDateTime
from obspy.core.inventory import (Inventory, Network, Station, Channel,
                                  Response, Equipment, Site,
                                  InstrumentSensitivity)


def float_or_none(value):
    if not value:
        return None
    else:
        return float(value)


def utcdatetime_or_none(value):
    if not value:
        return None
    else:
        return obspy.UTCDateTime(value)


# The header fields of the text files at the different request levels.
network_components = ("network", "description", "starttime", "endtime",
                      "totalstations")
network_types = (str, str, obspy.UTCDateTime, utcdatetime_or_none, int)
station_components = ("network", "station", "latitude", "longitude",
                      "elevation", "sitename", "starttime", "endtime")
station_types = (str, str, float, float, float, str, obspy.UTCDateTime,
                 utcdatetime_or_none)
channel_components = ("network", "station", "location", "channel", "latitude",
                      "longitude", "elevation", "depth", "azimuth", "dip",
                      "sensordescription", "scale", "scalefreq", "scaleunits",
                      "samplerate", "starttime", "endtime")
channel_types = (str, str, str, str, float, float, float, float_or_none,
                 float_or_none, float_or_none, str, float_or_none,
                 float_or_none, str, float_or_none, obspy.UTCDateTime,
                 utcdatetime_or_none)
all_components = (network_components, station_components, channel_components)


def unicode_csv_reader(unicode_csv_data, **kwargs):
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
    for row in csv_reader:
        try:
            yield [str(cell, "utf8") for cell in row]
        except Exception:
            yield [str(cell) for cell in row]


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        if isinstance(line, native_str):
            yield line
        else:
            yield line.encode('utf-8')


def is_fdsn_station_text_file(path_or_file_object):
    """
    Simple function checking if the passed object contains a valid FDSN
    station text file.

    :param path_or_file_object: File name or file like object.
    """
    try:
        if hasattr(path_or_file_object, "readline"):
            cur_pos = path_or_file_object.tell()
            first_line = path_or_file_object.readline()
        else:
            with open(path_or_file_object, "rt", encoding="utf8") as fh:
                first_line = fh.readline()
    except Exception:
        return False

    # Attempt to move the file pointer to the old position.
    try:
        path_or_file_object.seek(cur_pos, 0)
    except Exception:
        pass

    first_line = first_line.strip()

    # Attempt to decode.
    try:
        first_line = first_line.decode("utf-8")
    except Exception:
        pass

    if not first_line.startswith("#"):
        return False
    first_line = first_line.lstrip("#").strip()
    if not first_line:
        return False

    # IRIS currently has a wrong header name. Just map it.
    first_line = first_line.replace("Instrument", "SensorDescription")

    components = tuple(_i.strip().lower() for _i in first_line.split("|"))
    if components in all_components:
        return True
    return False


def read_fdsn_station_text_file(path_or_file_object):
    """
    Function reading a FDSN station text file to an inventory object.

    :param path_or_file_object: File name or file like object.
    """
    def _read(obj):
        r = unicode_csv_reader(obj, delimiter=native_str("|"))
        header = next(r)
        header[0] = header[0].lstrip("#")
        header = [_i.strip().lower() for _i in header]
        # IRIS currently has a wrong header name. Just map it.
        header = [_i.replace("instrument", "sensordescription") for _i in
                  header]

        all_lines = []
        for line in r:
            # Skip comment lines.
            if line[0].startswith("#"):
                continue
            all_lines.append([_i.strip() for _i in line])
        return {"header": tuple(header), "content": all_lines}

    # Enable reading from files and buffers opened in binary mode.
    if (hasattr(path_or_file_object, "mode") and
            "b" in path_or_file_object.mode) or \
            isinstance(path_or_file_object, io.BytesIO):
        buf = io.StringIO(path_or_file_object.read().decode("utf-8"))
        buf.seek(0, 0)
        path_or_file_object = buf

    if hasattr(path_or_file_object, "read"):
        content = _read(path_or_file_object)
    else:
        with open(path_or_file_object, "rt", newline="",
                  encoding="utf8") as fh:
            content = _read(fh)

    # Figure out the type.
    if content["header"] == network_components:
        level = "network"
        filetypes = network_types
    elif content["header"] == station_components:
        level = "station"
        filetypes = station_types
    elif content["header"] == channel_components:
        level = "channel"
        filetypes = channel_types
    else:
        raise ValueError("Unknown type of header.")

    content = content["content"]
    converted_content = []
    # Convert all types.
    for line in content:
        converted_content.append([
            v_type(value) for value, v_type in zip(line, filetypes)])

    # Now convert to an inventory object.
    inv = Inventory(networks=[], source=None)

    if level == "network":
        for net in converted_content:
            network = Network(
                code=net[0],
                description=net[1],
                start_date=net[2],
                end_date=net[3],
                total_number_of_stations=net[4])
            inv.networks.append(network)
    elif level == "station":
        networks = collections.OrderedDict()
        for sta in converted_content:
            site = Site(name=sta[5])
            station = Station(
                code=sta[1], latitude=sta[2], longitude=sta[3],
                elevation=sta[4], site=site, start_date=sta[6],
                end_date=sta[7])
            if sta[0] not in networks:
                networks[sta[0]] = []
            networks[sta[0]].append(station)
        for network_code, stations in networks.items():
            net = Network(code=network_code, stations=stations)
            inv.networks.append(net)
    elif level == "channel":
        networks = collections.OrderedDict()
        stations = collections.OrderedDict()

        for channel in converted_content:
            net, sta, loc, chan, lat, lng, ele, dep, azi, dip, inst, scale, \
                scale_freq, scale_units, s_r, st, et = channel

            if net not in networks:
                networks[net] = Network(code=net)

            if (net, sta) not in stations:
                station = Station(code=sta, latitude=lat,
                                  longitude=lng, elevation=ele)
                networks[net].stations.append(station)
                stations[(net, sta)] = station

            sensor = Equipment(type=inst)
            if scale is not None and scale_freq is not None:
                resp = Response(
                    instrument_sensitivity=InstrumentSensitivity(
                        value=scale, frequency=scale_freq,
                        input_units=scale_units, output_units=None))
            else:
                resp = None
            try:
                channel = Channel(
                    code=chan, location_code=loc, latitude=lat, longitude=lng,
                    elevation=ele, depth=dep, azimuth=azi, dip=dip,
                    sensor=sensor, sample_rate=s_r, start_date=st,
                    end_date=et, response=resp)
            except Exception as e:
                warnings.warn(
                    "Failed to parse channel %s.%s.%s.%s due to: %s" % (
                        net, sta, loc, chan, str(e)),
                    UserWarning)
                continue
            stations[(net, sta)].channels.append(channel)
        inv.networks.extend(list(networks.values()))
    else:
        # Cannot really happen - just a safety measure.
        raise NotImplementedError("Unkown level: " + str(level))
    return inv


def inventory_to_station_text(inventory_or_network):
    """
    Function to convert inventory or network to station text representation.

    :type inventory_or_network:
        :class:`~obspy.core.inventory.inventory.Inventory` or
        :class:`~obspy.core.inventory.network.Network`
    """
    if isinstance(inventory_or_network, Inventory):
        networks = inventory_or_network.networks
    elif isinstance(inventory_or_network, Network):
        networks = [inventory_or_network.networks]
    else:
        raise TypeError()

    items = []

    # assemble items to write
    for net in networks:
        if not net.stations:
            items.append((net, None, None))
            continue
        for sta in net.stations:
            if not sta.channels:
                items.append((net, sta, None))
                continue
            for cha in sta.channels:
                items.append((net, sta, cha))

    def _to_str(item):
        if item is None:
            return ""
        x = str(item)
        if isinstance(item, UTCDateTime):
            x = x.rstrip("Z")
            x = re.sub(r'\.0+$', '', x)
            x = re.sub(r'T00:00:00$', '', x)
        return x

    # depending on the items write with channel/station/network detail level
    # write with channel detail level
    if any(cha is not None for net, sta, cha in items):
        header = ("#Network|Station|Location|Channel|Latitude|Longitude|"
                  "Elevation|Depth|Azimuth|Dip|SensorDescription|Scale|"
                  "ScaleFrequency|ScaleUnits|SampleRate|StartTime|EndTime")
        lines = [header]
        for net, sta, cha in items:
            # omit stations with no channels and warn
            if cha is None:
                msg = ("Writing inventory with channel detail level and "
                       "encountered some networks/stations without channels. "
                       "These will be omitted in the output.")
                warnings.warn(msg)
                continue
            resp = cha and cha.response
            sensitivity = resp and resp.instrument_sensitivity
            line = "|".join(_to_str(x) for x in (
                net.code, sta.code, cha.location_code, cha.code,
                cha.latitude is not None and cha.latitude or sta.latitude,
                cha.longitude is not None and cha.longitude or sta.longitude,
                cha.elevation is not None and cha.elevation or sta.elevation,
                cha.depth, cha.azimuth, cha.dip,
                cha.sensor and cha.sensor.type,
                sensitivity and sensitivity.value,
                sensitivity and sensitivity.frequency,
                sensitivity and sensitivity.input_units,
                cha.sample_rate, cha.start_date, cha.end_date))
            lines.append(line)
    # write with station detail level
    elif any(sta is not None for net, sta, cha in items):
        header = ("#Network|Station|Latitude|Longitude|Elevation|StartTime|"
                  "EndTime")
        lines = [header]
        for net, sta, cha in items:
            # omit networks with no stations and warn
            if sta is None:
                msg = ("Writing inventory with station detail level and "
                       "encountered some networks without stations. "
                       "These will be omitted in the output.")
                warnings.warn(msg)
                continue
            line = "|".join(_to_str(x) for x in (
                net.code, sta.code, sta.latitude, sta.longitude, sta.elevation,
                sta.site and sta.site.name, sta.start_date, sta.end_date))
            lines.append(line)
    # write with network detail level
    else:
        header = "#Network|Description|StartTime|EndTime|TotalStations"
        lines = [header]
        for net, sta, cha in items:
            line = "|".join(_to_str(x) for x in (
                net.code, net.description, net.start_date, net.end_date,
                net.total_number_of_stations))
            lines.append(line)

    return "\n".join(lines)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
