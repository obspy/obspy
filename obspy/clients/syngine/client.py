#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy client for the IRIS syngine service.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import io

import numpy as np

import obspy
from obspy.core import AttribDict

from ..base import WaveformClient, HTTPClient, DEFAULT_USER_AGENT, \
    ClientHTTPException


class Client(WaveformClient, HTTPClient):
    """
    FDSN Web service request client.

    For details see the :meth:`~obspy.clients.fdsn.clients.Client.__init__()`
    method.
    """
    # Caching the database information, thus repeatedly initializing the
    # client is cheap.
    __cache = {}

    def __init__(self, base_url="http://service.iris.edu/irisws/syngine/1",
                 user_agent=DEFAULT_USER_AGENT, debug=False, timeout=20):
        """
        Initializes a Syngine Client.
        """
        HTTPClient.__init__(self, debug=debug, timeout=timeout,
                            user_agent=user_agent)

        # Make sure the base_url does not end with a slash.
        base_url = base_url.rstrip("/")
        self._base_url = base_url

    def _get_url(self, path):
        return "/".join([self._base_url.rstrip("/"), path])

    def _handle_requests_http_error(self, r):
        msg = "HTTP code %i when downloading '%s':\n\n%s" % (
            r.status_code, r.url, r.text)
        raise ClientHTTPException(msg.strip())

    def get_model_info(self, model_name):
        """
        Get some information about a particular model.

        :param model_name: The name of the model. Case insensitive.
        """
        model_name = model_name.strip().lower()
        key = "model_" + model_name
        self.__cache.setdefault(self._base_url, {})
        if key not in self.__cache[self._base_url]:
            r = self._download(self._get_url("info"),
                               params={"model": model_name})
            info = AttribDict(r.json())
            # Convert slip and sliprate into a numpy array for easier handling.
            info.slip = np.array(info.slip, dtype=np.float64)
            info.sliprate = np.array(info.sliprate, dtype=np.float64)
            self.__cache[self._base_url][key] = info
        return self.__cache[self._base_url][key]

    def get_service_version(self):
        self.__cache.setdefault(self._base_url, {})
        if "version" not in self.__cache[self._base_url]:
            r = self._download(self._get_url("version"))
            # Decoding and what not is handled by the requests library.
            self.__cache[self._base_url]["version"] = r.text
        return self.__cache[self._base_url]["version"]

    def get_waveforms(
            self, model, network=None, station=None,
            receiverlatitude=None, receiverlongitude=None,
            networkcode=None, stationcode=None, locationcode=None,
            eventid=None, sourcelatitude=None, sourcelongitude=None,
            sourcedepthinmeters=None, sourcemomenttensor=None,
            sourcedoublecouple=None, sourceforce=None, origintime=None,
            starttime=None, endtime=None, label=None, components="ZRT",
            units="velocity", scale=1.0, dt=None, kernelwidth=8,
            format="miniseed"):
        """
        """
        model = str(model).strip().lower()

        # Error handling is mostly delegated to the actual syngine service.
        # Here we just check that the types are compatible.
        str_arguments = ["network", "station", "networkcode", "stationcode",
                         "locationcode", "eventid", "label", "components",
                         "units", "format"]
        float_arguments = ["receiverlatitude", "receiverlongitude",
                           "sourcelatitude", "sourcelongitude",
                           "sourcedepthinmeters", "scale", "dt"]
        int_arguments = ["kernelwidth"]
        time_arguments = ["origintime"]

        for keys, t in ((str_arguments, str),
                       (float_arguments, float),
                       (int_arguments, int),
                       (time_arguments, obspy.UTCDateTime)):
            for key in keys:
                value = locals()[key]
                if value is None:
                    continue
                locals()[key] = t(value)

        # These can be absolute times, relative times or phase relative times.
        # jo
        temporal_bounds = ["starttime", "endtime"]
        for key in temporal_bounds:
            value = locals()[key]
            if value is None:
                continue
            # If a number, convert to a float.
            elif isinstance(value, (int, float)):
                value = float(value)
            # If a string like object, attempt to parse it to a datetime
            # object, otherwise assume its a phase-relative time and let the
            # syngine service deal with the erorr handling.
            elif isinstance(value, (str, native_str)):
                try:
                    value = obspy.UTCDateTime(value)
                except:
                    value = str(value)
            # Last but not least just try to pass it to the datetime
            # constructor without catching the error.
            else:
                value = obspy.UTCDateTime(value)
            locals()[0] = value

        # These all have to be lists of floats. Otherwise it fails.
        source_mecs = ["sourcemomenttensor",
                       "sourcedoublecouple",
                       "sourceforce"]
        for key in source_mecs:
            value = locals()[key]
            if value is None:
                continue
            value = [float(_i) for _i in value]
            locals()[key] = value

        # Now simply use all arguments to construct the query.
        all_arguments = ["model"]
        all_arguments.extend(str_arguments)
        all_arguments.extend(float_arguments)
        all_arguments.extend(int_arguments )
        all_arguments.extend(time_arguments)
        all_arguments.extend(temporal_bounds)
        all_arguments.extend(source_mecs)

        params = {}
        for arg in all_arguments:
            value = locals()[arg]
            if value is None:
                continue
            params[arg] = value

        r = self._download(url=self._get_url("query"), params=params)
        with io.BytesIO(r.content) as buf:
            st = obspy.read(buf)
        return st
