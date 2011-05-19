# -*- coding: utf-8 -*-
"""
NERIES web service client for ObsPy.

See: http://www.seismicportal.eu/jetspeed/portal/web-services.psml

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util import _getVersionString
import platform
import sys
import urllib
import urllib2
try:
    import json
    if not getattr(json, "loads", None):
        json.loads = json.read
except ImportError:
    import simplejson as json


MAP = {'min_datetime': "dateMin", 'max_datetime': "dateMax",
       'min_latitude': "latMin", 'max_latitude': "latMax",
       'min_longitude': "lonMin", 'max_longitude': "lonMax",
       'min_depth': "depthMin", 'max_depth': "depthMax",
       'min_magnitude': "magMin", 'max_magnitude': "magMax",
       'magnitude_type': "magType", 'author': "auth",
       'max_results': "limit", 'sort_by': "sort", 'sort_direction': "dir",
       'format': "format", 'datetime': "datetime", 'depth': "depth",
       'flynn_region': "flynn_region", 'latitude': "lat",
       'longitude': "lon", 'magnitude': "mag", 'origin_id': "orid",
       'event_id': "unid"}

MAP_INVERSE = dict([(value, key) for key, value in MAP.iteritems()])
# in results the "magType" key is all lowercase, so add it to..
MAP_INVERSE['magtype'] = "magnitude_type"

VERSION = _getVersionString("obspy.neries")
DEFAULT_USER_AGENT = "ObsPy %s (%s, Python %s)" % (VERSION, platform.platform(),
                                                   platform.python_version())


class Client(object):
    """
    NERIES web service request client.
    """
    def __init__(self, base_url="http://www.seismicportal.eu", user="",
                 password="", timeout=10, debug=False,
                 user_agent=DEFAULT_USER_AGENT):
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        self.user_agent = user_agent
        # Create an OpenerDirector for Basic HTTP Authentication
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, base_url, user, password)
        auth_handler = urllib2.HTTPBasicAuthHandler(password_mgr)
        opener = urllib2.build_opener(auth_handler)
        # install globally
        urllib2.install_opener(opener)

    def _fetch(self, url, headers={}, **params):
        """
        Send a HTTP request via urllib2.

        :type url: String
        :param url: Complete URL of resource
        :type headers: dict
        :param headers: Additional header information for request
        """
        headers['User-Agent'] = self.user_agent
        # replace special characters 
        remoteaddr = self.base_url + url + '?' + urllib.urlencode(params)
        if self.debug:
            print('\nRequesting %s' % (remoteaddr))
        # timeout exists only for Python >= 2.6
        if sys.hexversion < 0x02060000:
            response = urllib2.urlopen(remoteaddr)
        else:
            response = urllib2.urlopen(remoteaddr, timeout=self.timeout)
        doc = response.read()
        return doc

    def getEvents(self, min_datetime=None, max_datetime=None,
                  min_longitude=None, max_longitude=None, min_latitude=None,
                  max_latitude=None, min_depth=None, max_depth=None,
                  min_magnitude=None, max_magnitude=None, magnitude_type=None,
                  author=None, sort_by="datetime", sort_direction="ASC",
                  max_results=100, format="list", **kwargs):
        """
        Gets a list of events.

        Also see: http://www.seismicportal.eu/services/event/search/info

        Example
        -------
        >>> from obspy.neries import Client
        >>> client = Client()
        >>> events = client.getEvents(min_datetime="2004-12-01",
        ...                           max_datetime="2005-01-01",
        ...                           min_magnitude=9)
        >>> print len(events)
        1
        >>> print events #doctest: +NORMALIZE_WHITESPACE 
        [{'author': u'CSEM', 'event_id': u'20041226_0000148', 
          'origin_id': 127773, 'longitude': 95.724, 
          'datetime': u'2004-12-26T00:58:50Z', 'depth': -10.0, 'magnitude': 9.3,
          'magnitude_type': u'mw', 'latitude': 3.498,
          'flynn_region': u'OFF W COAST OF NORTHERN SUMATRA'}]

        Parameters
        ----------
        min_datetime : str, optional
            Earliest date and time for search.
            ISO 8601-formatted, in UTC: yyyy-MM-dd['T'HH:mm:ss].
            e.g.: "2002-05-17" or "2002-05-17T05:24:00"
        max_datetime : str, optional
            Latest date and time for search.
        min_latitude : int or float, optional
            Minimum latitude for search. Format: +/- 90 decimal degrees.
        max_latitude : int or float, optional
            Maximum latitude for search.
        min_longitude : int or float, optional
            Minimum ("left-side") longitude for search.
            Format: +/- 180 decimal degrees.
        max_longitude : int or float, optional
            Maximum ("right-side") longitude for search.
        min_depth : int or float, optional
            Minimum event depth. Format: in km, negative down.
        max_depth : int or float, optional
            Maximum event depth.
        min_magnitude : int or float, optional
            Minimum event magnitude.
        max_magnitude : int or float, optional
            Maximum event magnitude.
        magnitude_type : str, optional
            Magnitude scale type. Example: "mw", "mb".
        author : str, optional
            Origin author. Example: "CSEM", "LDG", ...
        max_results : int (maximum: 2500)
            Maximum number of returned results.
        sort_by : str
            Field to sort by. Options: "datetime", "magnitude", "flynn_region",
            "depth". Only available if format="list".
        sort_direction : str
            Sort direction. Format: "ASC" or "DESC".
        format : str
            format of returned results. Either "list" or "xml" (default).

        Returns
        -------
            List of event dictionaries or QuakeML string.
        """
        # NOTHING goes ABOVE this line!
        for key, value in locals().iteritems():
            if value and key not in ["self", "kwargs"]:
                key = MAP[key]
                kwargs[key] = value
        # map request format string "list" -> "json"
        if kwargs.get("format") == "list":
            kwargs['format'] = "json"
        # switch depth to positive down
        if kwargs.get("depthMin"):
            kwargs['depthMin'] = -kwargs['depthMin']
        if kwargs.get("depthMax"):
            kwargs['depthMax'] = -kwargs['depthMax']
        # fetch data
        url = "/services/event/search"
        results = self._fetch(url, **kwargs)
        # format output
        if format == "list":
            results = json.loads(results)
            events = []
            float_keys = ('depth', 'latitude', 'longitude', 'magnitude')
            for result in results['unids']:
                event = dict([(MAP_INVERSE[k], v)
                              for k, v in result.iteritems()])
                for k in float_keys:
                    event[k] = float(event[k])
                event['magnitude_type'] = event['magnitude_type'].lower()
                # convention in ObsPy: all depths negative down
                event['depth'] = -event['depth']
                events.append(event)
        else:
            events = results
        return events

    def getEventDetail(self, uri, format="xml"):
        """
        Gets event detail information.

        Also see: http://www.seismicportal.eu/services/event/detail/info

        Example
        -------
        >>> from obspy.neries import Client
        >>> client = Client()
        >>> result = client.getEventDetail("19990817_0000001", 'list')

        Number of calculated origins for the requested event
        >>> print len(result)
        12

        Details about first calculated origin of the requested event
        >>> print result[0] #doctest: +NORMALIZE_WHITESPACE 
        {'author': u'EMSC', 'event_id': u'19990817_0000001',
         'origin_id': 1465935, 'longitude': 29.972,
         'datetime': u'1999-08-17T00:01:35Z', 'depth': -10.0, 'magnitude': 6.7,
         'magnitude_type': u'mw', 'latitude': 40.749}

        Parameters
        ----------
        uri : str
            event identifier as either a EMSC event unique identifier, e.g. 
            "19990817_0000001" or a QuakeML-formatted event URI, e.g.
            "quakeml:eu.emsc/event#19990817_0000001"
        format : str, optional
            format of returned results. Either "list" or "xml" (default).

        Returns
        -------
            List of origin dictionaries or QuakeML string.
        """
        # parse parameters
        kwargs = {}
        if format == 'list':
            kwargs['format'] = 'json'
        else:
            kwargs['format'] = 'xml'
        if str(uri).startswith('quakeml:'):
            # QuakeML-formatted event URI
            kwargs['uri'] = str(uri)
        else:
            # EMSC event unique identifier 
            kwargs['unid'] = str(uri)
        # fetch data
        url = "/services/event/detail"
        results = self._fetch(url, **kwargs)
        # format output
        if format == "list":
            results = json.loads(results)
            events = []
            float_keys = ('depth', 'latitude', 'longitude', 'magnitude')
            for result in results['unids']:
                event = dict([(MAP_INVERSE[k], v)
                              for k, v in result.iteritems()])
                for k in float_keys:
                    event[k] = float(event[k])
                event['magnitude_type'] = event['magnitude_type'].lower()
                # convention in ObsPy: all depths negative down
                event['depth'] = -event['depth']
                events.append(event)
        else:
            events = results
        return events


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
