# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from future.builtins import str  # NOQA


def origin__geo_interface__(self):
    """
    __geo_interface__ method for GeoJSON-type GIS protocol

    :return: dict of valid GeoJSON

    Reference
    ---------
    Python geo_interface specifications:
    https://gist.github.com/sgillies/2217756

    """
    ts = None
    update_ts = None
    # Convert UTCDateTime objects to float timestamps
    if hasattr(self.time, 'timestamp'):
        ts = self.time.timestamp
    if hasattr(self.creation_info, 'creation_time') and \
       hasattr(self.creation_info.creation_time, 'timestamp'):
        update_ts = self.creation_info.creation_time.timestamp

    point = {
        "type": "Point",
        "coordinates": (self.longitude, self.latitude, self.depth),
        "id": str(self.resource_id),
        }
    props = {
        "time": ts,
        "updated": update_ts,
        }
    return {"type": "Feature", "properties": props, "geometry": point}


def event__geo_interface__(self):
    """
    __geo_interface__ method for GeoJSON-type GIS protocol

    :return: dict of valid GeoJSON

    Reference
    ---------
    Python geo_interface specifications:
    https://gist.github.com/sgillies/2217756

    Schema loosely based on the USGS GeoJSON format
    http://earthquake.usgs.gov/earthquakes/feed/v1.0/GeoJSON.php

    """
    # This will throw an error if no preferreds are set
    o = self.preferred_origin()
    m = self.preferred_magnitude()

    if o is None or m is None:
        raise ValueError("Preferred origin/magnitude not set!")

    gj = o.__geo_interface__
    gj['properties'].update({
        "mag": m.mag,
        "magtype": m.magnitude_type,
        "type": self.event_type,
        "url": str(self.resource_id),
        })
    return gj


def station__geo_interface__(self):
    """
    __geo_interface__ method for GeoJSON-type GIS protocol

    :return: dict of valid GeoJSON

    Reference
    ---------
    Python geo_interface specifications:
    https://gist.github.com/sgillies/2217756

    """
    # Convert UTCDateTime objects to float timestamps
    times = dict([(a, getattr(self, a).timestamp) for a in ('start_date',
                 'end_date', 'creation_date', 'termination_date')
                 if hasattr(getattr(self, a), 'timestamp')])

    point = {
        "type": "Point",
        "coordinates": (self.longitude, self.latitude, self.elevation),
        "id": self.code,
        }
    props = {
        "start_time": times.get('start_date'),
        "end_time": times.get('end_date'),
        "creation_time": times.get('creation_date'),
        "termination_time": times.get('termination_date'),
        }
    return {"type": "Feature", "properties": props, "geometry": point}

##############################################################################
# TESTING
##############################################################################
# TODO: break out to testing individual classes?
from obspy.core.event import Event, Origin, CreationInfo, UTCDateTime, \
    Magnitude
from obspy.station import Station
import unittest

Event.__geo_interface__ = property(event__geo_interface__)
Origin.__geo_interface__ = property(origin__geo_interface__)
Station.__geo_interface__ = property(station__geo_interface__)


class GeoTestCase(unittest.TestCase):
    def setUp(self):
        # Read in an Event from QuakeML or just make one
        ci = CreationInfo(
            creation_time=UTCDateTime(2012, 01, 02, 3, 45, 56, 789000)
            )
        self.origin = Origin(
            latitude=44.5, longitude=-120.56, depth=10505.0,
            time=UTCDateTime(2010, 02, 15, 13, 24, 34, 345000),
            creation_info=ci
            )
        self.mag = Magnitude(mag=6.7, magnitude_type='mw')
        # Add to event
        self.event = Event(
            origins=[self.origin],
            creation_info=ci,
            )
        # mod creation time
        self.event.creation_info.creation_time += 324.32

        # Read in a Station from StationXML or just make one
        self.station = Station(
            code="FOO", latitude=34.567, longitude=-119.234, elevation=45671,
            creation_date=UTCDateTime(2005, 01, 02, 11, 30, 35, 000000),
            start_date=UTCDateTime(2005, 01, 02, 12, 45, 21, 340000),
            end_date=UTCDateTime(2013, 12, 30, 23, 59, 59, 999999),
            termination_date=UTCDateTime(2013, 12, 31, 8, 0, 0, 000000),
            )

    def test_origin_geo_interface(self):
        d = self.origin.__geo_interface__
        self.assertIsInstance(d, dict)
        # TODO: verify stuff in the dictionary
        # TODO: try to read in as json?
        pass

    def test_event_geo_interface(self):
        # TODO: assert raises ValueError before setting this
        self.event.preferred_origin_id = self.origin.resource_id.id
        self.event.preferred_magnitude_id = self.mag.resource_id.id
        d = self.event.__geo_interface__
        self.assertIsInstance(d, dict)
        # TODO: verify stuff in the dictionary
        pass

    def test_station_geo_interface(self):
        d = self.station.__geo_interface__
        self.assertIsInstance(d, dict)
        # TODO: verify stuff in the dictionary
        # TODO: try to read in as json?
        pass


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(GeoTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
