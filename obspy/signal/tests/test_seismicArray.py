from unittest import TestCase

from obspy.signal.array_analysis import SeismicArray, correct_with_3dplane
from obspy.core.inventory.station import Station
from obspy.core.inventory.network import Network
import numpy as np


class TestSeismicArray(TestCase):
    """
    Test cases for the Seismic Array class.
    """

    def setUp(self):
        codes = "bl br m ul ur".split()
        lat = [0, 0, 1, 2, 2]
        long = [0, 2, 1, 0, 2]
        elev = [0, 0, 0, 0, 0]
        stns = [Station(codes[i], lat[i], long[i], elev[i]) for i in
                range(len(codes))]
        self.simpleinv = [Network("5pt", stations=stns)]

        #todo: this:
        self.simpleinv_nochannels = [[[]]]

        self.testarray = SeismicArray()
        self.testarray.add_inventory(self.simpleinv)

        self.geo_exp = {'5pt.bl': {'absolute_height_in_km': 0.0,
                                   'latitude': 0.0, 'longitude': 0.0},
                        '5pt.br': {'absolute_height_in_km': 0.0,
                                   'latitude': 0.0, 'longitude': 2.0},
                        '5pt.m': {'absolute_height_in_km': 0.0,
                                  'latitude': 1.0, 'longitude': 1.0},
                        '5pt.ul': {'absolute_height_in_km': 0.0,
                                   'latitude': 2.0, 'longitude': 0.0},
                        '5pt.ur': {'absolute_height_in_km': 0.0,
                                   'latitude': 2.0, 'longitude': 2.0}}
        self.geox_exp = {'5pt.bl': {'x': -111.31564682647114, 'y':
                                    -110.5751633754653, 'z': 0.0},
                         '5pt.br': {'x': 111.31564682647114, 'y':
                                    -110.5751633754653, 'z': 0.0},
                         '5pt.m': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                         '5pt.ul': {'x': -111.28219117308639, 'y':
                                    110.5751633754653, 'z': 0.0},
                         '5pt.ur': {'x': 111.28219117308639, 'y':
                                    110.5751633754653, 'z': 0.0}}


    def test_get_geometry_xyz(self):
        """
        Test get_geometry_xyz and, implicitly, _get_geometry (necessary because
        self.geometry is a property and can't be set).
        """
        geox = self.testarray.get_geometry_xyz(1, 1, 0,
                                               correct_3dplane=False)
        self.assertEqual(geox, self.geox_exp)
        geox = self.testarray.get_geometry_xyz(1, 1, 0,
                                               correct_3dplane=True)
        # todo: make a better test for this case?
        self.assertEqual(self.geox_exp, geox)

    def test__get_geometry(self):
        geo = self.testarray.geometry
        self.assertEqual(geo, self.geo_exp)
        # test for both inventories (or fake inventories) with and w/o channels

    def test_center_of_gravity(self):
        self.assertEqual(self.testarray.center_of_gravity,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_geometrical_center(self):
        self.assertEqual(self.testarray.geometrical_center,
                         {'absolute_height_in_km': 0.0,
                          'latitude': 1.0, 'longitude': 1.0})

    def test_correct_with_3dplane(self):
        self.assertEqual(self.geo_exp,
                         correct_with_3dplane(self.testarray.geometry))