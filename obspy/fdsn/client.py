# -*- coding: utf-8 -*-
"""
FDSN Web service client for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
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
    def __init__(self, base_url="IRIS", majorversion=1, user=None,
                 password=None):
        """
        Initializes the IRIS Web service client.

        :type base_url: str
        :param base_url: Base URL of FDSN web service compatible server or key
            string for recognized server (currently "IRIS", "USGS", "RESIF")
        :type majorversion: int
        :param majorversion: Major version number of server to access.
        :type user: str
        :param user: User name of HTTP Digest Authentication for access to
            restricted data.
        :type password: str
        :param password: Password of HTTP Digest Authentication for access to
            restricted data.
        """
        if user and password:
            # Create an OpenerDirector for HTTP Digest Authentication
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, base_url, user, password)
            auth_handler = urllib2.HTTPDigestAuthHandler(password_mgr)
            opener = urllib2.build_opener(auth_handler)
            # install globally
            urllib2.install_opener(opener)
        for service in SERVICES:
            url = "/".join([base_url, "fdsnws", service, str(majorversion),
                            "query"])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
