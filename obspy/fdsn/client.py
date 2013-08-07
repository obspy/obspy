# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import urllib
import urllib2

URL_MAPPINGS = {"IRIS": "http://service.iris.edu",
                "USGS": "http://comcat.cr.usgs.gov",
                "RESIF": "http://ws.resif.fr",
                "NCEDC": "http://service.ncedc.org",
                }
SERVICES = ("dataselect", "station", "event")


class Client(object):
    """
    FDSN Web service request client.
    """
    def __init__(self, base_url="IRIS", major_version=1, user=None,
                 password=None):
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
        """
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

        self.major_version = major_version

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
        """
        self._build_url("dataselect", "application.wadl")


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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
