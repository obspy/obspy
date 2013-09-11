# -*- coding: utf-8 -*-

from obspy.core.event import ResourceIdentifier, WaveformStreamID, Magnitude, \
    Origin, Event, Tensor, MomentTensor, FocalMechanism, Catalog, readEvents
from obspy.mchedr.mchedr import readMchedr
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.decorator import skipIf
import StringIO
import difflib
import math
import os
import unittest
import warnings


class mchedrTestCase(unittest.TestCase):
    """
    Test suite for obspy.mchedr
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    #def test_readMchedr(self):
    #    """
    #    """
    #    filename = os.path.join(self.path, 'mchedr.dat')
    #    catalog = readMchedr(filename)
    #    self.assertEqual(len(catalog), 2)
    #    self.assertEqual(
    #        catalog[0].resource_id,
    #        ResourceIdentifier(resource_id="20120101052755.98"))
    #    self.assertEqual(
    #        catalog[1].resource_id,
    #        ResourceIdentifier(resource_id="20120110183659.08"))

    def test_event(self):
        """
        Tests Event object.
        """
        filename = os.path.join(self.path, 'mchedr.dat')
        catalog = readMchedr(filename)
        self.assertEqual(len(catalog), 2)
        event = catalog[0]
        self.assertEqual(
            event.resource_id,
            ResourceIdentifier(resource_id="20120101052755.98"))
        # enums
        self.assertEqual(event.event_type, None)
        self.assertEqual(event.event_type_certainty, None)
        # comments
        self.assertEqual(len(event.comments), 1)
        c = event.comments
        self.assertEqual(c[0].text, 'MW 6.8 (WCMT), 6.8 (UCMT), 6.8 (GCMT). \
Felt (V) at Chiba; (IV) at Fussa, Kawasaki, Saitama, Tokyo, \
Yokohama and Yokosuka; (III) at Ebina, Zama and Zushi; (II) \
at Misawa and Narita, Honshu. Recorded (4 JMA) in Chiba, Fukushima, \
Gumma, Ibaraki, Kanagawa, Miyagi, Saitama, Tochigi and Tokyo.  ')
        # event descriptions
        self.assertEqual(len(event.event_descriptions), 2)
        d = event.event_descriptions
        self.assertEqual(d[0].text, 'SOUTHEAST OF HONSHU, JAPAN')
        self.assertEqual(d[0].type, 'region name')
        self.assertEqual(d[1].text, '211')
        self.assertEqual(d[1].type, 'Flinn-Engdahl region')
        # creation info
        self.assertEqual(event.creation_info, None)

    def test_origin(self):
        """
        Tests Origin object.
        """
        filename = os.path.join(self.path, 'mchedr.dat')
        catalog = readMchedr(filename)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(len(catalog[0].origins), 4)
        origin = catalog[0].origins[0]
        self.assertEqual(origin.time, UTCDateTime(2012, 1, 1, 5, 27, 55, 980000))
        self.assertEqual(origin.latitude, 31.456)
        self.assertEqual(origin.latitude_errors.uncertainty, 1.72)
        self.assertEqual(origin.longitude, 138.072)
        self.assertEqual(origin.longitude_errors.uncertainty, 1.64)
        self.assertEqual(origin.depth, 365.3)
        self.assertEqual(origin.depth_errors.uncertainty, 2.7)
        self.assertEqual(origin.depth_type, 'from location')
        self.assertEqual(origin.method_id, None)
        self.assertEqual(origin.time_fixed, None)
        self.assertEqual(origin.epicenter_fixed, None)
        self.assertEqual(
            origin.earth_model_id,
            ResourceIdentifier(resource_id='smi:ISC/emid=AK135'))
        self.assertEqual(origin.evaluation_mode, None)
        self.assertEqual(origin.evaluation_status, None)
        self.assertEqual(origin.origin_type, None)
        # composite times
        self.assertEqual(len(origin.composite_times), 0)
        # quality
        self.assertEqual(origin.quality.used_station_count, 628)
        self.assertEqual(origin.quality.standard_error, 0.84)
        self.assertEqual(origin.quality.azimuthal_gap, 10.8)
        self.assertEqual(origin.quality.maximum_distance, 128.07)
        self.assertEqual(origin.quality.minimum_distance, 2.22)
        self.assertEqual(origin.quality.associated_phase_count, 878)
        self.assertEqual(origin.quality.associated_station_count, 628)
        self.assertEqual(origin.quality.depth_phase_count, 0)
        self.assertEqual(origin.quality.secondary_azimuthal_gap, None)
        self.assertEqual(origin.quality.ground_truth_level, None)
        self.assertEqual(origin.quality.median_distance, None)
        # comments
        self.assertEqual(len(origin.comments), 0)
        # creation info
        self.assertEqual(origin.creation_info.author, None)
        self.assertEqual(origin.creation_info.agency_id, 'USGS-NEIC')
        self.assertEqual(origin.creation_info.author_uri, None)
        self.assertEqual(origin.creation_info.agency_uri, None)
        self.assertEqual(origin.creation_info.creation_time, None)
        self.assertEqual(origin.creation_info.version, None)
        # origin uncertainty
        u = origin.origin_uncertainty
        self.assertEqual(u.preferred_description, 'confidence ellipsoid')
        self.assertEqual(u.horizontal_uncertainty, None)
        self.assertEqual(u.min_horizontal_uncertainty, None)
        self.assertEqual(u.max_horizontal_uncertainty, None)
        self.assertEqual(u.azimuth_max_horizontal_uncertainty, None)
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        self.assertEqual(c.semi_intermediate_axis_length, 2.75)
        #c.major_axis_rotation is computed during file reading:
        self.assertLessEqual(abs(c.major_axis_rotation - 80.505), 1e-3)
        self.assertEqual(c.major_axis_plunge, 76.06)
        self.assertEqual(c.semi_minor_axis_length, 2.21)
        self.assertEqual(c.semi_major_axis_length, 4.22)
        self.assertEqual(c.major_axis_azimuth, 292.79)


def suite():
    return unittest.makeSuite(mchedrTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
