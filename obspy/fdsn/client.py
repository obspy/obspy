#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from io import BytesIO
import obspy
from obspy.fdsn.wadl_parser import WADLParser
from obspy.fdsn.header import DEFAULT_USER_AGENT, \
    DEFAULT_DATASELECT_PARAMETERS, DEFAULT_STATION_PARAMETERS, \
    DEFAULT_EVENT_PARAMETERS, URL_MAPPINGS

import Queue
import threading
import urllib
import urllib2
import warnings


class FDSNException(Exception):
    pass


class Client(object):
    """
    FDSN Web service request client.
    """
    def __init__(self, base_url="IRIS", major_version=1, user=None,
                 password=None, user_agent=DEFAULT_USER_AGENT, debug=False):
        """
        Initializes the IRIS Web service client.

        :type base_url: str
        :param base_url: Base URL of FDSN web service compatible server or key
            string for recognized server (currently "IRIS", "USGS", "RESIF",
            "NCEDC")
        :type major_version: int
        :param major_version: Major version number of server to access.
        :type user: str
        :param user: User name of HTTP Digest Authentication for access to
            restricted data.
        :type password: str
        :param password: Password of HTTP Digest Authentication for access to
            restricted data.
        :type user_agent: str
        :param user_agent: The user agent for all requests.
        :type debug: bool
        :param debug: Debug flag.
        """
        self.debug = debug
        #if user and password:
            ## Create an OpenerDirector for HTTP Digest Authentication
            #password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            #password_mgr.add_password(None, base_url, user, password)
            #auth_handler = urllib2.HTTPDigestAuthHandler(password_mgr)
            #opener = urllib2.build_opener(auth_handler)
            ## install globally
            #urllib2.install_opener(opener)

        if base_url.upper() in URL_MAPPINGS:
            self.base_url = URL_MAPPINGS[base_url.upper()]
        else:
            self.base_url = base_url
        # Make sure the base_url does not end with a slash.
        self.base_url = self.base_url.strip("/")

        self.request_headers = {"User-Agent": user_agent}
        self.major_version = major_version

        if self.debug is True:
            print "Base URL: %s" % self.base_url
            print "Request Headers: %s" % str(self.request_headers)

        self._discover_services()

    def get_event(self, starttime=None, endtime=None, minlatitude=None,
                  maxlatitude=None, minlongitude=None, maxlongitude=None,
                  latitude=None, longitude=None, minradius=None,
                  maxradius=None, mindepth=None, maxdepth=None,
                  minmagnitude=None, maxmagnitude=None, magnitudetype=None,
                  includeallorigins=None, includeallmagnitudes=None,
                  includearrivals=None, eventid=None, limit=None, offset=None,
                  orderby=None, catalog=None, contributor=None,
                  updatedafter=None, filename=None, **kwargs):
        """
        Query the event service of the client.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit to events on or after the specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit to events on or before the specified end time.
        :type minlatitude: float, optional
        :param minlatitude: Limit to events with a latitude larger than the
            specified minimum.
        :type maxlatitude: float, optional
        :param maxlatitude: Limit to events with a latitude smaller than the
            specified maximum.
        :type minlongitude: float, optional
        :param minlongitude: Limit to events with a longitude larger than the
            specified minimum.
        :type maxlongitude: float, optional
        :param maxlongitude: Limit to events with a longitude smaller than the
            specified maximum.
        :type latitude: float, optional
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float, optional
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float, optional
        :param minradius: Limit to events within the specified minimum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type maxradius: float, optional
        :param maxradius: Limit to events within the specified maximum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type mindepth: float, optional
        :param mindepth: Limit to events with depth more than the specified
            minimum.
        :type maxdepth: float, optional
        :param maxdepth: Limit to events with depth less than the specified
            maximum.
        :type minmagnitude: float, optional
        :param minmagnitude: Limit to events with a magnitude larger than the
            specified minimum.
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.
        :type includeallorigins: bool, optional
        :param includeallorigins: Specify if all origins for the event should
            be included, default is data center dependent but is suggested to
            be the preferred origin only.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: Specify if all magnitudes for the event
            should be included, default is data center dependent but is
            suggested to be the preferred magnitude only.
        :type includearrivals: bool, optional
        :param includearrivals: Specify if phase arrivals should be included.
        :type eventid: str or int (dependent on data center), optional
        :param eventid: Select a specific event by ID; event identifiers are
            data center specific.
        :type limit: int, optional
        :param limit: Limit the results to the specified number of events.
        :type offset: int, optional
        :param offset: Return results starting at the event count specified,
            starting at 1.
        :type orderby: str, optional
        :param orderby: Order the result by time or magnitude with the
            following possibilities:
                * time: order by origin descending time
                * time-asc: order by origin ascending time
                * magnitude: order by descending magnitude
                * magnitude-asc: order by ascending magnitude
        :type catalog: str, optional
        :param catalog: Limit to events from a specified catalog
        :type contributor: str, optional
        :param contributor: Limit to events contributed by a specified
            contributor.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Limit to events updated after the specified time.
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.


        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "event" not in self.services:
            msg = "The current client does not have an event service."
            raise ValueError(msg)

        locs = locals()
        for param in ("starttime", "endtime", "minlatitude", "maxlatitude",
                      "minlongitude", "maxlongitude", "latitude", "longitude",
                      "minradius", "maxradius", "mindepth", "maxdepth",
                      "minmagnitude", "maxmagnitude", "magnitudetype",
                      "includeallorigins", "includeallmagnitudes",
                      "includearrivals", "eventid", "limit", "offset",
                      "orderby", "catalog", "contributor", "updatedafter"):
            value = locs[param]
            if value is not None:
                kwargs[param] = value

        url = self._create_url_from_parameters(
            "event", DEFAULT_DATASELECT_PARAMETERS, kwargs)

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            cat = obspy.readEvents(data_stream, format="quakeml")
            data_stream.close()
            return cat

    def get_waveform(self, starttime, endtime, network, station, location,
                     channel, quality=None, minimumlength=None,
                     longestonly=None, filename=None, **kwargs):
        """
        Query the dataselect service of the client.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Limit results to time series samples on or after the
            specified start time
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Limit results to time series samples on or before the
            specified end time
        :type network: str
        :param network: Select one or more network codes. Can be SEED network
            codes or data center defined codes. Multiple codes are
            comma-separated. Wildcards are allowed.
        :type station: str
        :param station: Select one or more SEED station codes. Multiple codes
            are comma-separated. Wildcards are allowed.
        :type location: str
        :param location: Select one or more SEED location identifiers. Multiple
            identifiers are comma-separated. Wildcards are allowed.
        :type channel: str
        :param channel: Select one or more SEED channel codes. Multiple codes
            are comma-separated.
        :type quality: str, optional
        :param quality: Select a specific SEED quality indicator, handling is
            data center dependent.
        :type minimumlength: float, optional
        :param minimumlength: Limit results to continuous data segments of a
            minimum length specified in seconds.
        :type longestonly: bool, optional
        :param longestonly: Limit results to the longest continuous segment per
            channel.
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.

        Any additional keyword arguments will be passed to the webservice as
        additional arguments. If you pass one of the default parameters and the
        webservice does not support it, a warning will be issued. Passing any
        non-default parameters that the webservice does not support will raise
        an error.
        """
        if "dataselect" not in self.services:
            msg = "The current client does not have a dataselect service."
            raise ValueError(msg)

        # Combine all parameters in the kwargs dictionary.
        kwargs["starttime"] = starttime
        kwargs["endtime"] = endtime
        kwargs["network"] = network
        kwargs["station"] = station
        kwargs["location"] = location
        kwargs["channel"] = channel
        if quality is not None:
            kwargs["quality"] = quality
        if minimumlength is not None:
            kwargs["minimumlength"] = minimumlength
        if longestonly is not None:
            kwargs["longestonly"] = longestonly
        url = self._create_url_from_parameters(
            "dataselect", DEFAULT_DATASELECT_PARAMETERS, kwargs)

        data_stream = self._download(url)
        data_stream.seek(0, 0)
        if filename:
            self._write_to_file_object(filename, data_stream)
            data_stream.close()
        else:
            st = obspy.read(data_stream, format="MSEED")
            data_stream.close()
            return st

    def _write_to_file_object(filename_or_object, data_stream):
        if hasattr(filename_or_object, "write"):
            filename_or_object.write(data_stream.read())
            return
        with open(filename_or_object, "wb") as fh:
            fh.write(data_stream.read())

    def _create_url_from_parameters(self, service, default_params, parameters):
        """
        """
        service_params = self.services[service]
        # Get all required parameters and make sure they are available!
        required_parameters = [
            key for key, value in service_params.iteritems()
            if value["required"] is True]
        for req_param in required_parameters:
            if req_param not in parameters:
                msg = "Parameter '%s' is required." % req_param
                raise TypeError(msg)

        final_parameter_set = {}

        # Now loop over all parameters, convert them and make sure they are
        # accepted by the service.
        for key, value in parameters.iteritems():
            if key not in service_params:
                # If it is not in the service but in the default parameters
                # raise a warning.
                if key in default_params:
                    msg = ("The standard parameter '%s' is not supporte by "
                           "the webservice. It will be silently ignored." %
                           key)
                    warnings.warn(msg)
                    continue
                # Otherwise raise an error.
                else:
                    msg = \
                        "The parameter '%s' is not supported by the service." \
                        % key
                    raise TypeError(msg)
            # Now attempt to convert the parameter to the correct type.
            this_type = service_params[key]["type"]
            try:
                value = this_type(value)
            except:
                msg = "'%s' could not be converted to type '%s'." % (
                    str(value), this_type.__name__)
                raise TypeError(msg)
            # Now convert to a string that is accepted by the webservice.
            final_parameter_set[key] = convert_to_string(value)

        return self._build_url(service, "query",
                               parameters=final_parameter_set)

    def __str__(self):
        ret = (
            "FDSN Webservice Client (base url: {url})\n"
            "Available Services: {services}".format(
            url=self.base_url,
            services=", ".join(self.services.keys())))
        return ret

    def help(self, service):
        """
        Print a more extensive help for a given service.

        This will use the already parsed WADL files and be specific for each
        data center and always up-to-date.
        """
        if service not in self.services:
            msg = "Service '%s' not available for current client." % service
            raise ValueError(msg)

        if service == "dataselect":
            SERVICE_DEFAULT = DEFAULT_DATASELECT_PARAMETERS
        elif service == "event":
            SERVICE_DEFAULT = DEFAULT_EVENT_PARAMETERS
        elif service == "station":
            SERVICE_DEFAULT = DEFAULT_STATION_PARAMETERS
        else:
            raise NotImplementedError

        print "Parameter description for the '%s' service of '%s':" % (
            service, self.base_url)

        # Loop over all parameters and group them in three list: available
        # default parameters, missing default parameters and additional
        # parameters
        available_default_parameters = []
        missing_default_parameters = []
        additional_parameters = []
        long_default_names = [_i[0] for _i in SERVICE_DEFAULT]
        for name in long_default_names:
            if name in self.services[service]:
                available_default_parameters.append(name)
            else:
                missing_default_parameters.append(name)

        for name in self.services[service].iterkeys():
            if name not in long_default_names:
                additional_parameters.append(name)

        def _print_param(name):
            param = self.services[service][name]
            name = "%s (%s)" % (name, param["type"].__name__)
            req_def = ""
            if param["required"]:
                req_def = "Required Parameter"
            elif param["default_value"]:
                req_def = "Default value: %s" % str(param["default_value"])
            if param["options"]:
                req_def += "Choices: %s" % \
                    ", ".join(map(str, param["options"]))
            if req_def:
                req_def = ", %s" % req_def
            if param["doc_title"]:
                doc_title = "\n        %s" % param["doc_title"]
            else:
                doc_title = ""

            print "    {name}{req_def}{doc_title}".format(
                name=name, req_def=req_def, doc_title=doc_title)

        for name in available_default_parameters:
            _print_param(name)

        if additional_parameters:
            print "The service offers the following non-standard parameters:"
            for name in additional_parameters:
                _print_param(name)

        if missing_default_parameters:
            print("WARNING: The service does not offer the following "
                  "standard parameters: %s" %
                  ", ".join(missing_default_parameters))

    def _download(self, url):
        code, data = download_url(url, headers=self.request_headers,
                                  debug=self.debug, return_string=False)
        # No data.
        if code == 204:
            raise FDSNException("No data available for request.")
        elif code == 400:
            msg = "Bad request. Please contact the developers."
            raise NotImplementedError(msg)
        elif code == 401:
            raise FDSNException("Unauthorized, authentication required.")
        elif code == 403:
            raise FDSNException("Authentication failed.")
        elif code == 413:
            raise FDSNException("Request would result in too much data. "
                                "Denied by the datacenter. Split the request "
                                "in smaller parts")
        # Request URI too large.
        elif code == 414:
            msg = ("The request URI is too large. Please contact the ObsPy "
                   "developers.")
            raise NotImplementedError(msg)
        elif code == 500:
            raise FDSNException("Service responds: Internal server error")
        elif code == 503:
            raise FDSNException("Service temporarily unavailable")

        return data

    def _build_url(self, resource_type, service, parameters={}):
        """
        Builds a correct URL.
        """
        return build_url(self.base_url, self.major_version, resource_type,
                         service, parameters)

    def _discover_services(self):
        """
        Automatically discovers available services.

        They are discovered by downloading the corresponding WADL files. If a
        WADL does not exist, the services are assumed to be non-existent.
        """
        dataselect_url = self._build_url("dataselect", "application.wadl")
        station_url = self._build_url("station", "application.wadl")
        event_url = self._build_url("event", "application.wadl")
        urls = (dataselect_url, station_url, event_url)

        # Request all three WADL files in parallel.
        wadl_queue = Queue.Queue()

        headers = self.request_headers
        debug = self.debug

        def get_download_thread(url):
            class ThreadURL(threading.Thread):
                def run(self):
                    code, data = download_url(url, headers=headers,
                                              debug=debug)
                    if code == 200:
                        wadl_queue.put((url, data))
                    else:
                        wadl_queue.put((url, None))
            return ThreadURL()

        threads = map(get_download_thread, urls)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)

        self.services = {}
        for _ in range(wadl_queue.qsize()):
            item = wadl_queue.get()
            url, wadl = item
            if wadl is None:
                continue
            if "dataselect" in url:
                self.services["dataselect"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered dataselect service"
            elif "event" in url:
                self.services["event"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered event service"
            elif "station" in url:
                self.services["station"] = WADLParser(wadl).parameters
                if self.debug is True:
                    print "Discovered station service"
        if not self.services:
            msg = ("No FDSN services could be discoverd at '%s'. This could "
                   "be due to a temporary service outage or an invalid FDSN "
                   "service address." % self.base_url)
            raise FDSNException(msg)


def convert_to_string(value):
    """
    Takes any value and converts it to a string compliant with the FDSN
    webservices.

    Will raise a ValueError if the value could not be converted.

    >>> convert_to_string("abcd")
    'abcd'
    >>> convert_to_string(1)
    '1'
    >>> convert_to_string(1.2)
    '1.2'
    >>> convert_to_string(UTCDateTime(2012, 1, 2, 3, 4, 5, 666666))
    '2012-01-02T03:04:05.666666'
    >>> convert_to_string(True)
    'true'
    >>> convert_to_string(False)
    'false'
    """
    if isinstance(value, basestring):
        return value
    # Boolean test must come before integer check!
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return str(value)
    elif isinstance(value, obspy.UTCDateTime):
        return str(value).replace("Z", "")


def build_url(base_url, major_version, resource_type, service, parameters={}):
    """
    URL builder for the FDSN webservices.

    Built as a separate function to enhance testability.

    >>> build_url("http://service.iris.edu", 1, "dataselect", \
                  "application.wadl")
    'http://service.iris.edu/fdsnws/dataselect/1/application.wadl'

    >>> build_url("http://service.iris.edu", 1, "dataselect", \
                  "query", {"cha": "EHE"})
    'http://service.iris.edu/fdsnws/dataselect/1/query?cha=EHE'
    """
    # Only allow certain resource types.
    if resource_type not in ["dataselect", "event", "station"]:
        msg = "Resource type '%s' not allowed. Allowed resource types: \n%s" %\
            (resource_type, ",".join(("dataselect", "event", "station")))
        raise ValueError(msg)

    url = "/".join((base_url, "fdsnws", resource_type,
                    str(major_version), service))
    if parameters:
        url = "?".join((url, urllib.urlencode(parameters)))
    return url


def download_url(url, timeout=10, headers={}, debug=False, return_string=True):
    """
    Returns a pair of tuples.

    The first one is the returned HTTP code and the second the data as
    string.

    Will return a touple of Nones if the service could not be found.
    """
    if debug is True:
        print "Downloading %s" % url

    try:
        url_obj = urllib2.urlopen(urllib2.Request(url=url, headers=headers),
                                  timeout=timeout)
    except urllib2.URLError:
        if debug is True:
            print "Error while downloading: %s" % url
        return None, None

    code = url_obj.getcode()
    if return_string is False:
        data = BytesIO(url_obj.read())
    else:
        data = url_obj.read()

    if debug is True:
        print "Downloaded %s with HTTP code: %i" % (url, code)

    return code, data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
