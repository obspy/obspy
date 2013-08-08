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
from obspy.fdsn.wadl_parser import WADLParser
from obspy.fdsn.header import DEFAULT_USER_AGENT, \
    DEFAULT_DATASELECT_PARAMETERS, DEFAULT_STATION_PARAMETERS, \
    DEFAULT_EVENT_PARAMETERS, URL_MAPPINGS

import Queue
import threading
import urllib
import urllib2


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


def download_url(url, timeout=10, headers={}, debug=False):
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

    code, data = url_obj.getcode(), url_obj.read()

    if debug is True:
        print "Downloaded %s with HTTP code: %i" % (url, code)

    return code, data


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
