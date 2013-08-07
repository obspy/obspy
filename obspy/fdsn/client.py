# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import __version__

import platform
import Queue
import threading
import urllib
import urllib2

URL_MAPPINGS = {"IRIS": "http://service.iris.edu",
                "USGS": "http://comcat.cr.usgs.gov",
                "RESIF": "http://ws.resif.fr",
                "NCEDC": "http://service.ncedc.org",
                }
SERVICES = ("dataselect", "station", "event")

DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (__version__,
                                                   platform.platform(),
                                                   platform.python_version())


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
        self.debug = True
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
                    code, data = download_url(dataselect_url, headers=headers,
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

        services = {}
        for _ in range(wadl_queue.qsize()):
            item = wadl_queue.get()
            url, wadl = item
            if wadl is None:
                continue
            if "dataselect" in url:
                services["dataselect"] = wadl
                if self.debug is True:
                    print "Discovered dataselect service"
            elif "event" in url:
                services["event"] = wadl
                if self.debug is True:
                    print "Discovered event service"
            elif "station" in url:
                services["station"] = wadl
                if self.debug is True:
                    print "Discovered station service"


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
