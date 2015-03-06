# -*- coding: utf-8 -*-
"""
Header files for the FDSN webservice.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import platform

from obspy import UTCDateTime, __version__


class FDSNException(Exception):
    pass


# A curated list collecting some implementations:
# http://www.fdsn.org/webservices/datacenters/
URL_MAPPINGS = {"IRIS": "http://service.iris.edu",
                "ORFEUS": "http://www.orfeus-eu.org",
                "USGS": "http://comcat.cr.usgs.gov",
                "RESIF": "http://ws.resif.fr",
                "NCEDC": "http://service.ncedc.org",
                "USP": "http://sismo.iag.usp.br",
                "GFZ": "http://geofon.gfz-potsdam.de",
                "NERIES": "http://www.seismicportal.eu",
                "SCEC": "http://www.data.scec.org",
                "GEONET": "http://service.geonet.org.nz",
                "INGV": "http://webservices.rm.ingv.it",
                "BGR": "http://eida.bgr.de",
                }

FDSNWS = ("dataselect", "event", "station")

# The default User Agent that will be sent with every request.
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (__version__,
                                                   platform.platform(),
                                                   platform.python_version())


# The default parameters. Different services can choose to add more. It always
# contains the long name first and the short name second. If it has no short
# name, it is simply a tuple with only one entry.
DEFAULT_DATASELECT_PARAMETERS = [
    "starttime", "endtime", "network", "station", "location", "channel",
    "quality", "minimumlength", "longestonly"]

DEFAULT_STATION_PARAMETERS = [
    "starttime", "endtime", "startbefore", "startafter", "endbefore",
    "endafter", "network", "station", "location", "channel", "minlatitude",
    "maxlatitude", "minlongitude", "maxlongitude", "latitude", "longitude",
    "minradius", "maxradius", "level", "includerestricted",
    "includeavailability", "updatedafter", "matchtimeseries"]

DEFAULT_EVENT_PARAMETERS = [
    "starttime", "endtime", "minlatitude", "maxlatitude", "minlongitude",
    "maxlongitude", "latitude", "longitude", "minradius", "maxradius",
    "mindepth", "maxdepth", "minmagnitude", "maxmagnitude", "magnitudetype",
    "includeallorigins", "includeallmagnitudes", "includearrivals",
    "eventid", "limit", "offset", "orderby", "catalog", "contributor",
    "updatedafter"]

DEFAULT_PARAMETERS = {
    "dataselect": DEFAULT_DATASELECT_PARAMETERS,
    "event": DEFAULT_EVENT_PARAMETERS,
    "station": DEFAULT_STATION_PARAMETERS}

PARAMETER_ALIASES = {
    "net": "network",
    "sta": "station",
    "loc": "location",
    "cha": "channel",
    "start": "starttime",
    "end": "endtime",
    "minlat": "minlatitude",
    "maxlat": "maxlatitude",
    "minlon": "minlongitude",
    "maxlon": "maxlongitude",
    "lat": "latitude",
    "lon": "longitude",
    "minmag": "minmagnitude",
    "maxmag": "maxmagnitude",
    "magtype": "magnitudetype",
}


# The default types if none are given. If the parameter can not be found in
# here and has no specified type, the type will be assumed to be a string.
DEFAULT_TYPES = {
    "starttime": UTCDateTime,
    "endtime": UTCDateTime,
    "network": str,
    "station": str,
    "location": str,
    "channel": str,
    "quality": str,
    "minimumlength": float,
    "longestonly": bool,
    "startbefore": UTCDateTime,
    "startafter": UTCDateTime,
    "endbefore": UTCDateTime,
    "endafter": UTCDateTime,
    "maxlongitude": float,
    "minlongitude": float,
    "longitude": float,
    "maxlatitude": float,
    "minlatitude": float,
    "latitude": float,
    "maxdepth": float,
    "mindepth": float,
    "maxmagnitude": float,
    "minmagnitude": float,
    "magnitudetype": str,
    "maxradius": float,
    "minradius": float,
    "level": str,
    "includerestricted": bool,
    "includeavailability": bool,
    "includeallorigins": bool,
    "includeallmagnitudes": bool,
    "includearrivals": bool,
    "matchtimeseries": bool,
    "eventid": str,
    "limit": int,
    "offset": int,
    "orderby": str,
    "catalog": str,
    "contributor": str,
    "updatedafter": UTCDateTime}

# This list collects WADL parameters that will not be parsed because they are
# not useful for the ObsPy client.
# Current the nodata parameter used by IRIS is part of that list. The ObsPy
# client relies on the HTTP codes.
# Furthermore the format parameter is part of that list. ObsPy relies on the
# default format.
WADL_PARAMETERS_NOT_TO_BE_PARSED = ["nodata", "format"]
