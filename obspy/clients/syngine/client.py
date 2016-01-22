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

import collections
import copy
import io
import pprint

import numpy as np

import obspy
from obspy.core import AttribDict

from ..base import WaveformClient, HTTPClient, DEFAULT_USER_AGENT, \
    ClientHTTPException


class Client(WaveformClient, HTTPClient):
    """
    Client for the IRIS syngine service.
    """
    def __init__(self, base_url="http://service.iris.edu/irisws/syngine/1",
                 user_agent=DEFAULT_USER_AGENT, debug=False, timeout=20):
        """
        Initializes a Syngine Client.

        :param base_url: The base URL of the service.
        :type base_url: str
        :param user_agent: The user agent sent along the HTTP request.
        :type user_agent: str
        :param debug: Debug on/off.
        :type debug: bool
        :param timeout: The socket timeout.
        :type timeout: float
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
        :type model_name: str
        :returns: A dictionary with more information about any model.
        :rtype: :class:`obspy.core.attribdict.AttribDict`
        """
        model_name = model_name.strip().lower()
        r = self._download(self._get_url("info"),
                           params={"model": model_name})
        info = AttribDict(r.json())
        # Convert slip and sliprate into numpy arrays for easier handling.
        info.slip = np.array(info.slip, dtype=np.float64)
        info.sliprate = np.array(info.sliprate, dtype=np.float64)
        return info

    def get_available_models(self):
        """
        Get information about all available velocity models.
        """
        return self._download(self._get_url("models")).json()

    def print_model_information(self):
        """
        Print information about the available models.
        """
        pprint.pprint(self.get_available_models())

    def get_service_version(self):
        """
        Get the service version of the remote syngine server.
        """
        r = self._download(self._get_url("version"))
        # Decoding and what not is handled by the requests library.
        return r.text

    def _convert_parameters(self, model, **kwargs):
        model = model.strip().lower()
        if not model:
            raise ValueError("Model must be given.")

        params = {"model": model}

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
                try:
                    value = kwargs[key]
                except KeyError:
                    continue
                if value is None:
                    continue
                value = t(value)
                # String arguments are stripped and empty strings are not
                # allowed.
                if t is str:
                    value = value.strip()
                    if not value:
                        raise ValueError("String argument '%s' must not be "
                                         "an empty string." % key)
                params[key] = t(value)

        # These can be absolute times, relative times or phase relative times.
        # jo
        temporal_bounds = ["starttime", "endtime"]
        for key in temporal_bounds:
            try:
                value = kwargs[key]
            except KeyError:
                continue
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
            params[key] = value

        # These all have to be lists of floats. Otherwise it fails.
        source_mecs = ["sourcemomenttensor",
                       "sourcedoublecouple",
                       "sourceforce"]
        for key in source_mecs:
            try:
                value = kwargs[key]
            except KeyError:
                continue
            if value is None:
                continue
            value = ",".join(["%g" % float(_i) for _i in value])
            params[key] = value

        return params


    def get_waveforms(
            self, model, network=None, station=None,
            receiverlatitude=None, receiverlongitude=None,
            networkcode=None, stationcode=None, locationcode=None,
            eventid=None, sourcelatitude=None, sourcelongitude=None,
            sourcedepthinmeters=None, sourcemomenttensor=None,
            sourcedoublecouple=None, sourceforce=None, origintime=None,
            starttime=None, endtime=None, label=None, components=None,
            units=None, scale=None, dt=None, kernelwidth=None,
            format="miniseed", filename=None):
        """
        Request waveforms using the Syngine service.

        This method is strongly tied to the actual implementation on the
        server side. The default values and all the exception handling are
        deferred to the service. Please see `the syngine documentation
        <http://ds.iris.edu/ds/products/syngine/>`_ for more details and the
        default values of all parameters.

        :param model: Specify the model.
        :type model: str
        :param network: Specify a network code combined with ``station`` to
            identify receiver coordinates of a operating station.
        :type network: str
        :param station: Specify a network code combined with ``network`` to
            identify receiver coordinates of a operating station.
        :type station: str
        :param receiverlatitude: Specify the receiver latitude in degrees.
        :type receiverlatitude: float
        :param receiverlongitude: Specify the receiver longitude in degrees.
        :type receiverlongitude: float
        :param networkcode: Specify the network code for synthetic. Optional
            when using ``receiverlatitude`` and ``receiverlongitude``.
        :type networkcode: str
        :param stationcode: Specify the station code for synthetic. Optional
            when using ``receiverlatitude`` and ``receiverlongitude``.
        :type stationcode: str
        :param locationcode: Specify the location code for synthetic. Optional
            in any usage.
        :type locationcode: str
        :param eventid: Specify an event identifier in the form
            [catalog]:[eventid]. The centroid time and location and moment
            tensor of the solution will be used as the source.
        :type eventid: str
        :param sourcelatitude: Specify the source latitude.
        :type sourcelatitude: float
        :param sourcelongitude: Specify the source longitude.
        :type sourcelongitude: float
        :param sourcedepthinmeters: Specify the source depth in meters from
            0 to 700000.
        :type sourcedepthinmeters: float
        :param sourcemomenttensor: Specify a source in moment tensor
            components as a list: ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``,
            ``Mtp`` with values in Newton meters (*Nm*).
        :type sourcemomenttensor: list of floats
        :param sourcedoublecouple: Specify a source as a double couple. The
            list of values are ``strike``, ``dip``, ``rake`` [,``M0``],
            where strike, dip and rake are in degrees and M0 is the scalar
            seismic moment in Newton meters (Nm). If not specified, a value
            of *1e19* will be used as the scalar moment.
        :type sourcedoublecouple: list of floats
        :param sourceforce: Specify a force source as a list of ``Fr``, ``Ft``,
            ``Fp`` in units of Newtons (N).
        :type sourceforce: list of floats
        :param origintime: Specify the source origin time. This must be
            specified as an absolute date and time.
        :type origintime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specifies the desired start time for the synthetic
            trace(s). This may be specified as either:

            * an absolute date and time
            * a phase-relative offset
            * an offset from origin time in seconds

            If the value is recognized as a date and time, it is interpreted
            as an absolute time. If the value is in the form
            ``phase[+-]offset`` it is interpreted as a phase-relative time,
            for example **P-10** (meaning Pwave arrival time minus 10
            seconds). If the value is a numerical value it is interpreted as an
            offset, in seconds, from the ``origintime``.
        :type starttime::class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specifies the desired end time for the synthetic
            trace(s). This may be specified as either:

            * an absolute date and time
            * a phase-relative offset
            * an offset from origin time in seconds

            If the value is recognized as a date and time, it is interpreted
            as an absolute time. If the value is in the form
            ``phase[+-]offset`` it is interpreted as a phase-relative time,
            for example **P-10** (meaning Pwave arrival time minus 10
            seconds). If the value is a numerical value it is interpreted as an
            offset, in seconds, from the ``starttime``.
        :type endtime::class:`~obspy.core.utcdatetime.UTCDateTime`
        :param label: Specify a label to be included in file names and HTTP
            file name suggestions.
        :type label: str
        :param components: Specify the orientation of the synthetic
            seismograms as a list of any combination of ``Z`` (vertical),
            ``N`` (north), ``E`` (east), ``R`` (radial), ``T`` (transverse)
        :type components: str or list of strings.
        :param units: Specify either ``displacement``, ``velocity`` or
            ``acceleration`` for the synthetics. The length unit is meters.
        :type units: str
        :param scale: Specify an amplitude scaling factor. The default
            amplitude length unit is meters.
        :type scale: float
        :param dt: Specify the sampling interval in seconds. Only upsampling
            is allowed so this value must be larger than the intrinsic interval
            of the model database.
        :type dt: float
        :param kernelwidth: Specify the width of the sinc kernel used for
            resampling to requested sample interval (``dt``), relative to the
            original sampling rate.
        :type kernelwidth: int
        :param format: Specify output file to be either miniSEED or a ZIP
            archive of SAC files, either ``miniseed`` or ``saczip``.
        :type format: str
        :param filename: Will download directly to the given file. If given,
            this method will return nothing.
        :type filename: str or file-like object
        """
        arguments = {
            "network": network,
            "station": station,
            "receiverlatitude": receiverlatitude,
            "receiverlongitude": receiverlongitude,
            "networkcode": networkcode,
            "stationcode": stationcode,
            "locationcode": locationcode,
            "eventid": eventid,
            "sourcelatitude": sourcelatitude,
            "sourcelongitude": sourcelongitude,
            "sourcedepthinmeters": sourcedepthinmeters,
            "sourcemomenttensor": sourcemomenttensor,
            "sourcedoublecouple": sourcedoublecouple,
            "sourceforce": sourceforce,
            "origintime": origintime,
            "starttime": starttime,
            "endtime": endtime,
            "label": label,
            "components": components,
            "units": units,
            "scale": scale,
            "dt": dt,
            "kernelwidth": kernelwidth,
            "format": format,
            "filename": filename}

        params = self._convert_parameters(model=model, **arguments)

        r = self._download(url=self._get_url("query"), params=params,
                           filename=filename)

        # A given filename will write directly to a file.
        if filename:
            return

        with io.BytesIO(r.content) as buf:
            st = obspy.read(buf)
        return st

    def get_waveforms_bulk(
            self, model, bulk,
            eventid=None, sourcelatitude=None, sourcelongitude=None,
            sourcedepthinmeters=None, sourcemomenttensor=None,
            sourcedoublecouple=None, sourceforce=None, origintime=None,
            starttime=None, endtime=None, label=None, components=None,
            units=None, scale=None, dt=None, kernelwidth=None,
            format="miniseed", filename=None):
        """
        Request multiple waveforms using the Syngine service at once.

        This method is strongly tied to the actual implementation on the
        server side. The default values and all the exception handling are
        deferred to the service. Please see `the syngine documentation
        <http://ds.iris.edu/ds/products/syngine/>`_ for more details and the
        default values of all parameters.

        This method used the POST functionalities of the syngine service.

        :param model: Specify the model.
        :type model: str
        :param bulk: Specify the receivers to download in bulk.
        :type bulk: list of lists, tuples, or dictionaries
        :param eventid: Specify an event identifier in the form
        [catalog]:[eventid]. The centroid time and location and moment
        tensor of the solution will be used as the source.
        :type eventid: str
        :param sourcelatitude: Specify the source latitude.
        :type sourcelatitude: float
        :param sourcelongitude: Specify the source longitude.
        :type sourcelongitude: float
        :param sourcedepthinmeters: Specify the source depth in meters from
        0 to 700000.
        :type sourcedepthinmeters: float
        :param sourcemomenttensor: Specify a source in moment tensor
        components as a list: ``Mrr``, ``Mtt``, ``Mpp``, ``Mrt``, ``Mrp``,
        ``Mtp`` with values in Newton meters (*Nm*).
        :type sourcemomenttensor: list of floats
        :param sourcedoublecouple: Specify a source as a double couple. The
        list of values are ``strike``, ``dip``, ``rake`` [,``M0``],
        where strike, dip and rake are in degrees and M0 is the scalar
        seismic moment in Newton meters (Nm). If not specified, a value
        of *1e19* will be used as the scalar moment.
        :type sourcedoublecouple: list of floats
        :param sourceforce: Specify a force source as a list of ``Fr``, ``Ft``,
        ``Fp`` in units of Newtons (N).
        :type sourceforce: list of floats
        :param origintime: Specify the source origin time. This must be
        specified as an absolute date and time.
        :type origintime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specifies the desired start time for the synthetic
            trace(s). This may be specified as either:

        * an absolute date and time
        * a phase-relative offset
        * an offset from origin time in seconds

        If the value is recognized as a date and time, it is interpreted
        as an absolute time. If the value is in the form
        ``phase[+-]offset`` it is interpreted as a phase-relative time,
        for example **P-10** (meaning Pwave arrival time minus 10
        seconds). If the value is a numerical value it is interpreted as an
        offset, in seconds, from the ``origintime``.
        :type starttime::class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Specifies the desired end time for the synthetic
        trace(s). This may be specified as either:

        * an absolute date and time
        * a phase-relative offset
        * an offset from origin time in seconds

        If the value is recognized as a date and time, it is interpreted
        as an absolute time. If the value is in the form
        ``phase[+-]offset`` it is interpreted as a phase-relative time,
        for example **P-10** (meaning Pwave arrival time minus 10
        seconds). If the value is a numerical value it is interpreted as an
        offset, in seconds, from the ``starttime``.
        :type endtime::class:`~obspy.core.utcdatetime.UTCDateTime`
        :param label: Specify a label to be included in file names and HTTP
        file name suggestions.
        :type label: str
        :param components: Specify the orientation of the synthetic
        seismograms as a list of any combination of ``Z`` (vertical),
        ``N`` (north), ``E`` (east), ``R`` (radial), ``T`` (transverse)
        :type components: str or list of strings.
        :param units: Specify either ``displacement``, ``velocity`` or
        ``acceleration`` for the synthetics. The length unit is meters.
        :type units: str
        :param scale: Specify an amplitude scaling factor. The default
        amplitude length unit is meters.
        :type scale: float
        :param dt: Specify the sampling interval in seconds. Only upsampling
        is allowed so this value must be larger than the intrinsic interval
        of the model database.
        :type dt: float
        :param kernelwidth: Specify the width of the sinc kernel used for
            resampling to requested sample interval (``dt``), relative to the
            original sampling rate.
        :type kernelwidth: int
        :param format: Specify output file to be either miniSEED or a ZIP
        archive of SAC files, either ``miniseed`` or ``saczip``.
        :type format: str
        :param filename: Will download directly to the given file. If given,
        this method will return nothing.
        :type filename: str or file-like object
        """
        if not bulk:
            raise ValueError("Some bulk download information must be given.")

        arguments = {
            "eventid": eventid,
            "sourcelatitude": sourcelatitude,
            "sourcelongitude": sourcelongitude,
            "sourcedepthinmeters": sourcedepthinmeters,
            "sourcemomenttensor": sourcemomenttensor,
            "sourcedoublecouple": sourcedoublecouple,
            "sourceforce": sourceforce,
            "origintime": origintime,
            "starttime": starttime,
            "endtime": endtime,
            "label": label,
            "components": components,
            "units": units,
            "scale": scale,
            "dt": dt,
            "kernelwidth": kernelwidth,
            "format": format,
            "filename": filename}

        params = self._convert_parameters(model=model, **arguments)

        # Assemble the bulk file.
        with io.BytesIO() as buf:
            # Write the header.
            buf.write("model={model}\n".format(**params).encode())
            del params["model"]
            for key in sorted(params.keys()):
                value = params[key]
                buf.write("{key}={value}\n".format(key=key,
                                                   value=value).encode())

            # Write the bulk content.
            for item in bulk:
                # Dictionary like items.
                if isinstance(item, collections.Mapping):
                    if "latitude" in item or "longitude" in item:
                        if not ("latitude" in item and "longitude" in item):
                            raise ValueError(
                                "Item '%s' in bulk must contain both "
                                "latitude and longitude if either is given." %
                                str(item))
                        bulk_item = "{latitude} {longitude}".format(**item)
                        for _i in ("netcode", "stacode", "loccode"):
                            if _i in item:
                                bulk_item += " %s=%s" % (_i.upper(), item[_i])
                    elif "stacode" in item and "netcode" in item:
                        bulk_item = "{netcode} {stacode}".format(**item)
                    else:
                        raise ValueError("Item '%s' in bulk is malformed." %
                                         str(item))
                # Iterable items.
                elif isinstance(item, collections.Container):
                    if len(item) != 2:
                        raise ValueError("Item '%s' in bulk must have two "
                                         "entries." % str(item))
                    bulk_item = "%s %s" % (item[0], item[1])
                else:
                    raise ValueError("Item '%s' in bulk cannot be parsed." %
                                     str(item))

                buf.write((bulk_item.strip() + "\n").encode())

            buf.seek(0, 0)

            r = self._download(url=self._get_url("query"),
                               data=buf, filename=filename)

            # A given filename will write directly to a file.
            if filename:
                return

            with io.BytesIO(r.content) as buf:
                st = obspy.read(buf)
            return st
