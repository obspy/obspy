#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parsing of the text files from the FDSN station web services.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
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
import warnings

import obspy
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
        except:
            yield [str(cell) for cell in row]


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        if isinstance(line, native_str):
            yield line
        else:
            yield line.encode('utf-8')


def is_FDSN_station_text_file(path_or_file_object):
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
    except:
        return False

    # Attempt to move the file pointer to the old position.
    try:
        path_or_file_object.seek(cur_pos, 0)
    except:
        pass

    first_line = first_line.strip()

    # Attempt to decode.
    try:
        first_line = first_line.decode("utf-8")
    except:
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


def read_FDSN_station_text_file(path_or_file_object):
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
