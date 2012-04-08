# -*- coding: utf-8 -*-

from obspy.core.quakeml import readQuakeML, _catalogToXML
from obspy.core.utcdatetime import UTCDateTime
import os
import unittest


class QuakeMLTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.quakeml
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.join(os.path.dirname(__file__), 'data')

    def test_readQuakeML(self):
        """
        """
        # iris
        filename = os.path.join(self.path, 'iris_events.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 2)
        self.assertEquals(catalog[0].id,
                          'smi:www.iris.edu/ws/event/query?eventId=3279407')
        self.assertEquals(catalog[1].id,
                          'smi:www.iris.edu/ws/event/query?eventId=2318174')
        # neries
        filename = os.path.join(self.path, 'neries_events.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 3)
        self.assertEquals(catalog[0].id,
                          'quakeml:eu.emsc/event/20120404_0000041')
        self.assertEquals(catalog[1].id,
                          'quakeml:eu.emsc/event/20120404_0000038')
        self.assertEquals(catalog[2].id,
                          'quakeml:eu.emsc/event/20120404_0000039')

    def test_event(self):
        """
        Tests Event object.
        """
        filename = os.path.join(self.path, 'quakeml_event.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        event = catalog[0]
        self.assertEquals(event.id, 'smi:ch.ethz.sed/event/historical/1165')
        self.assertEquals(event.public_id,
            'smi:ch.ethz.sed/event/historical/1165')
        self.assertEquals(event.preferred_origin_id,
            'smi:ch.ethz.sed/origin/2054')
        self.assertEquals(event.preferred_magnitude_id,
            'smi:ch.ethz.sed/magnitude/1015')
        self.assertEquals(event.preferred_focal_mechanism_id,
            'smi:ch.ethz.sed/fm/23151')
        # enums
        self.assertEquals(event.type, 'earthquake')
        self.assertEquals(event.type_certainty, 'suspected')
        # comments
        self.assertEquals(len(event.comments), 2)
        c = event.comments
        self.assertEquals(c[0].text, 'Relocated after re-evaluation')
        self.assertEquals(c[0].id, None)
        self.assertEquals(c[0].creation_info.agency_id, 'EMSC')
        self.assertEquals(c[1].text, 'Another comment')
        self.assertEquals(c[1].id, '#3')
        self.assertEquals(c[1].creation_info.author, None)
        # event descriptions
        self.assertEquals(len(event.descriptions), 3)
        d = event.descriptions
        self.assertEquals(d[0].text, '1906 San Francisco Earthquake')
        self.assertEquals(d[0].type, 'earthquake name')
        self.assertEquals(d[1].text, 'NEAR EAST COAST OF HONSHU, JAPAN')
        self.assertEquals(d[1].type, 'Flinn-Engdahl region')
        self.assertEquals(d[2].text, 'free-form string')
        self.assertEquals(d[2].type, None)
        # creation info
        self.assertEquals(event.creation_info.author, "Erika Mustermann")
        self.assertEquals(event.creation_info.agency_id, "EMSC")
        self.assertEquals(event.creation_info.author_uri,
            "smi:smi-registry/organization/EMSC")
        self.assertEquals(event.creation_info.agency_uri,
            "smi:smi-registry/organization/EMSC")
        self.assertEquals(event.creation_info.creation_time,
            UTCDateTime("2012-04-04T16:40:50+00:00"))
        self.assertEquals(event.creation_info.version, "1.0.1")
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        original = ''.join([l.strip() for l in original.splitlines()[2:]])
        processed = _catalogToXML(catalog)
        processed = ''.join([l.strip() for l in processed.splitlines()[2:]])
        self.assertEquals(original, processed)

    def test_origin(self):
        """
        Tests Origin object.
        """
        filename = os.path.join(self.path, 'quakeml_origin.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].origins), 1)
        origin = catalog[0].origins[0]
        self.assertEquals(origin.public_id,
                          'smi:www.iris.edu/ws/event/query?originId=7680412')
        self.assertEquals(origin.time.value,
                          UTCDateTime("2011-03-11T05:46:24.1200"))
        self.assertEquals(origin.latitude.value, 38.297)
        self.assertEquals(origin.latitude.lower_uncertainty, None)
        self.assertEquals(origin.longitude.value, 142.373)
        self.assertEquals(origin.longitude.uncertainty, None)
        self.assertEquals(origin.depth.value, 29.0)
        self.assertEquals(origin.depth.confidence_level, 50.0)
        self.assertEquals(origin.depth_type, "from location")
        self.assertEquals(origin.method_id, "method/NA")
        self.assertEquals(origin.time_fixed, None)
        self.assertEquals(origin.epicenter_fixed, False)
        self.assertEquals(origin.reference_system_id, "muh")
        self.assertEquals(origin.earth_model_id, "maeh")
        self.assertEquals(origin.evaluation_mode, "manual")
        self.assertEquals(origin.evaluation_status, "preliminary")
        self.assertEquals(origin.type, "hypocenter")
        # composite times
        self.assertEquals(len(origin.composite_times), 2)
        c = origin.composite_times
        self.assertEquals(c[0].year.value, 2029)
        self.assertEquals(c[0].month.value, None)
        self.assertEquals(c[0].day.value, None)
        self.assertEquals(c[0].hour.value, 12)
        self.assertEquals(c[0].minute.value, None)
        self.assertEquals(c[0].second.value, None)
        self.assertEquals(c[1].year.value, None)
        self.assertEquals(c[1].month.value, None)
        self.assertEquals(c[1].day.value, None)
        self.assertEquals(c[1].hour.value, 1)
        self.assertEquals(c[1].minute.value, None)
        self.assertEquals(c[1].second.value, 29.124234)
        # quality
        self.assertEquals(origin.quality.used_station_count, 16)
        self.assertEquals(origin.quality.standard_error, 0)
        self.assertEquals(origin.quality.azimuthal_gap, 231)
        self.assertEquals(origin.quality.maximum_distance, 53.03)
        self.assertEquals(origin.quality.minimum_distance, 2.45)
        self.assertEquals(origin.quality.associated_phase_count, None)
        self.assertEquals(origin.quality.associated_station_count, None)
        self.assertEquals(origin.quality.depth_phase_count, None)
        self.assertEquals(origin.quality.secondary_azimuthal_gap, None)
        self.assertEquals(origin.quality.ground_truth_level, None)
        self.assertEquals(origin.quality.median_distance, None)
        # comments
        self.assertEquals(len(origin.comments), 2)
        c = origin.comments
        self.assertEquals(c[0].text, 'Some comment')
        self.assertEquals(c[0].id, "muh")
        self.assertEquals(c[0].creation_info.author, 'EMSC')
        self.assertEquals(c[1].creation_info.creation_time, None)
        self.assertEquals(c[1].text, 'Another comment')
        self.assertEquals(c[1].id, None)
        self.assertEquals(c[1].creation_info.agency_id, None)
        # creation info
        self.assertEquals(origin.creation_info.author, "NEIC")
        self.assertEquals(origin.creation_info.agency_id, None)
        self.assertEquals(origin.creation_info.author_uri, None)
        self.assertEquals(origin.creation_info.agency_uri, None)
        self.assertEquals(origin.creation_info.creation_time, None)
        self.assertEquals(origin.creation_info.version, None)
        # origin uncertainty
        u = origin.origin_uncertainty
        self.assertEquals(u.preferred_description, "uncertainty ellipse")
        self.assertEquals(u.horizontal_uncertainty, 9000)
        self.assertEquals(u.min_horizontal_uncertainty, 6000)
        self.assertEquals(u.max_horizontal_uncertainty, 10000)
        self.assertEquals(u.azimuth_max_horizontal_uncertainty, 80.0)
        # confidence ellipsoid
        c = u.confidence_ellipsoid
        self.assertEquals(c.semi_intermediate_axis_length, 2.123)
        self.assertEquals(c.major_axis_rotation, 5.123)
        self.assertEquals(c.major_axis_plunge, 3.123)
        self.assertEquals(c.semi_minor_axis_length, 1.123)
        self.assertEquals(c.semi_major_axis_length, 0.123)
        self.assertEquals(c.major_axis_azimuth, 4.123)
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        original = ''.join([l.strip() for l in original.splitlines()[2:]])
        processed = _catalogToXML(catalog)
        processed = ''.join([l.strip() for l in processed.splitlines()[2:]])
        self.assertEquals(original, processed)

    def test_magnitude(self):
        """
        Tests Magnitude object.
        """
        filename = os.path.join(self.path, 'quakeml_magnitude.xml')
        catalog = readQuakeML(filename)
        self.assertEquals(len(catalog), 1)
        self.assertEquals(len(catalog[0].magnitudes), 1)
        mag = catalog[0].magnitudes[0]
        self.assertEquals(mag.public_id, 'smi:ch.ethz.sed/magnitude/37465')
        self.assertEquals(mag.mag.value, 5.5)
        self.assertEquals(mag.mag.uncertainty, 0.1)
        self.assertEquals(mag.type, 'MS')
        self.assertTrue(mag.method_id.startswith('smi:ch.ethz.sed/magnitude'))
        self.assertEquals(mag.station_count, 8)
        self.assertEquals(mag.evaluation_status, 'preliminary')
        # comments
        self.assertEquals(len(mag.comments), 2)
        c = mag.comments
        self.assertEquals(c[0].text, 'Some comment')
        self.assertEquals(c[0].id, "muh")
        self.assertEquals(c[0].creation_info.author, 'EMSC')
        self.assertEquals(c[1].creation_info.creation_time, None)
        self.assertEquals(c[1].text, 'Another comment')
        self.assertEquals(c[1].id, None)
        self.assertEquals(c[1].creation_info.agency_id, None)
        # creation info
        self.assertEquals(mag.creation_info.author, "NEIC")
        self.assertEquals(mag.creation_info.agency_id, None)
        self.assertEquals(mag.creation_info.author_uri, None)
        self.assertEquals(mag.creation_info.agency_uri, None)
        self.assertEquals(mag.creation_info.creation_time, None)
        self.assertEquals(mag.creation_info.version, None)
        # exporting back to XML should result in the same document
        original = open(filename, "rt").read()
        original = ''.join([l.strip() for l in original.splitlines()[2:]])
        processed = _catalogToXML(catalog)
        processed = ''.join([l.strip() for l in processed.splitlines()[2:]])
        self.assertEquals(original, processed)


#    def test_writeQuakeML(self):
#        """
#        Tests writing a QuakeML document.
#        """
#        filename = os.path.join(self.path, 'quakeml_in.xml')
#        catalog = readQuakeML(filename)
#        print writeQuakeML(catalog, 'test')


def suite():
    return unittest.makeSuite(QuakeMLTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
