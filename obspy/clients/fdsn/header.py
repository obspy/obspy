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
from future.utils import PY2

import platform
import sys

from obspy import UTCDateTime, __version__


class FDSNException(Exception):
    def __init__(self, value, server_info=None):
        if server_info is not None:
            value = "\n".join([value, "Detailed response of server:", "",
                               server_info])
        super(FDSNException, self).__init__(value)


# A curated list collecting some implementations:
# http://www.fdsn.org/webservices/datacenters/
# http://www.orfeus-eu.org/eida/eida_odc.html
URL_MAPPINGS = {
    "BGR": "http://eida.bgr.de",
    "ETH": "http://eida.ethz.ch",
    "GEONET": "http://service.geonet.org.nz",
    "GFZ": "http://geofon.gfz-potsdam.de",
    "INGV": "http://webservices.rm.ingv.it",
    "IPGP": "http://eida.ipgp.fr",
    "IRIS": "http://service.iris.edu",
    "ISC": "http://isc-mirror.iris.washington.edu",
    "KOERI": "http://eida.koeri.boun.edu.tr",
    "LMU": "http://erde.geophysik.uni-muenchen.de",
    "NCEDC": "http://service.ncedc.org",
    "NIEP": "http://eida-sc3.infp.ro",
    "NERIES": "http://www.seismicportal.eu",
    "ODC": "http://www.orfeus-eu.org",
    "ORFEUS": "http://www.orfeus-eu.org",
    "RESIF": "http://ws.resif.fr",
    "SCEDC": "http://service.scedc.caltech.edu",
    "USGS": "http://earthquake.usgs.gov",
    "USP": "http://sismo.iag.usp.br",
    }

FDSNWS = ("dataselect", "event", "station")

if PY2:
    platform_ = platform.platform().decode("ascii", "ignore")
else:
    encoding = sys.getdefaultencoding() or "UTF-8"
    platform_ = platform.platform().encode(encoding).decode("ascii", "ignore")
# The default User Agent that will be sent with every request.
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (
    __version__, platform_, platform.python_version())


# The default parameters. Different services can choose to add more. It always
# contains the long name first and the short name second. If it has no short
# name, it is simply a tuple with only one entry.
DEFAULT_DATASELECT_PARAMETERS = [
    "starttime", "endtime", "network", "station", "location", "channel"]

OPTIONAL_DATASELECT_PARAMETERS = [
    "quality", "minimumlength", "longestonly"]

DEFAULT_STATION_PARAMETERS = [
    "starttime", "endtime", "network", "station", "location", "channel",
    "minlatitude", "maxlatitude", "minlongitude", "maxlongitude", "level"]

OPTIONAL_STATION_PARAMETERS = [
    "startbefore", "startafter", "endbefore", "endafter", "latitude",
    "longitude", "minradius", "maxradius", "includerestricted",
    "includeavailability", "updatedafter", "matchtimeseries", "format"]

DEFAULT_EVENT_PARAMETERS = [
    "starttime", "endtime", "minlatitude", "maxlatitude", "minlongitude",
    "maxlongitude", "mindepth", "maxdepth", "minmagnitude", "maxmagnitude",
    "orderby"]

OPTIONAL_EVENT_PARAMETERS = [
    "latitude", "longitude", "minradius", "maxradius", "magnitudetype",
    "includeallorigins", "includeallmagnitudes", "includearrivals", "eventid",
    "limit", "offset", "catalog", "contributor", "updatedafter"]

DEFAULT_PARAMETERS = {
    "dataselect": DEFAULT_DATASELECT_PARAMETERS,
    "event": DEFAULT_EVENT_PARAMETERS,
    "station": DEFAULT_STATION_PARAMETERS}

OPTIONAL_PARAMETERS = {
    "dataselect": OPTIONAL_DATASELECT_PARAMETERS,
    "event": OPTIONAL_EVENT_PARAMETERS,
    "station": OPTIONAL_STATION_PARAMETERS}

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
WADL_PARAMETERS_NOT_TO_BE_PARSED = ["nodata"]
