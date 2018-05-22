# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import os
import unittest

from obspy import read_events, Catalog, UTCDateTime
from obspy.core.event import Event
from obspy.io.focmec.core import _is_focmec, _read_focmec


lst_file_first_comment = '\n'.join((
    "",
    "     Dip,Strike,Rake     76.43    59.08   -64.23",
    "     Dip,Strike,Rake     28.90   174.99  -150.97   Auxiliary Plane",
    "     Lower Hem. Trend, Plunge of A,N     84.99    61.10   329.08    "
    "13.57",
    "     Lower Hem. Trend, Plunge of P,T    358.82    51.71   128.90    "
    "26.95",
    "     B trend, B plunge, Angle:  232.62  25.00 105.00",
    "",
    "          Log10(Ratio)                              Ratio     S Polarity",
    "     Observed  Calculated    Difference  Station     Type     Obs.  "
    "Calc. Flag",
    "      0.8847      0.8950      -0.0103      BLA        SH       R      "
    "R       ",
    "      1.1785      1.0810       0.0975      COR        SH       R      "
    "R       ",
    "      0.6013      0.5442       0.0571      HRV        SH       R      "
    "R       ",
    "      0.3287      0.3666      -0.0379      KEV        SH       L      "
    "L       ",
    "      0.8291      0.9341      -0.1050      KIP        SH       R      "
    "R       ",
    "      0.8033      0.7815       0.0218      KIP        SV       B      "
    "B       ",
    "      1.0783      1.1857      -0.1074      PAS        SH       R      "
    "R       ",
    "      0.2576      0.2271       0.0305      TOL        SH       L      "
    "L       ",
    "     -0.2762     -0.4076       0.1314      TOL        SS       F      "
    "F    NUM",
    "     -0.4283     -0.4503       0.0220      HRV        SS       F      "
    "F       ",
    "     -0.0830      0.0713      -0.1543      KEV        SS       B      "
    "B       ",
    "",
    "Total number of ratios used is  11",
    "RMS error for the 11 acceptable solutions is 0.0852",
    "  Highest absolute velue of diff for those solutions is  0.1543"))
out_file_first_comment = (
    "    Dip   Strike   Rake    Pol: P     SV    SH  AccR/TotR  RMS RErr  "
    "AbsMaxDiff\n   76.43   59.08  -64.23       0.00  0.00  0.00    11/11    "
    "0.0852    0.1543")


class FOCMECTestCase(unittest.TestCase):
    """
    Test everything related to reading FOCMEC files
    """
    def setUp(self):
        # Directory where the test files are located
        self.path = os.path.dirname(__file__)
        self.lst_file = os.path.join(self.path, 'data', 'focmec_8sta.lst')
        self.out_file = os.path.join(self.path, 'data', 'focmec_8sta.out')
        with open(self.out_file, 'rb') as fh:
            header = []
            for i in range(15):
                header.append(fh.readline().decode('ASCII'))
            self.out_file_header = ''.join(header).rstrip()
        with open(self.lst_file, 'rb') as fh:
            header = []
            for i in range(48):
                header.append(fh.readline().decode('ASCII'))
            self.lst_file_header = ''.join(header).rstrip()

    def _assert_cat_common_parts(self, cat):
        self.assertTrue(isinstance(cat, Catalog))
        self.assertEqual(len(cat), 1)
        event = cat[0]
        self.assertTrue(isinstance(event, Event))
        self.assertEqual(event.creation_info.creation_time,
                         UTCDateTime(2017, 9, 8, 14, 54, 58))
        self.assertEqual(len(event.focal_mechanisms), 4)
        expected_dip_strike_rake = (
            (76.43, 59.08, -64.23),
            (77.05, 54.08, -59.13),
            (77.05, 59.89, -59.13),
            (77.76, 54.50, -54.06))
        for focmec, (dip, strike, rake) in zip(
                event.focal_mechanisms, expected_dip_strike_rake):
            plane1 = focmec.nodal_planes.nodal_plane_1
            self.assertEqual(plane1.strike, strike)
            self.assertEqual(plane1.dip, dip)
            self.assertEqual(plane1.rake, rake)
            self.assertEqual(focmec.nodal_planes.preferred_plane, 1)
        for focmec in cat[0].focal_mechanisms:
            self.assertEqual(focmec.station_polarity_count, 23)

    def _assert_cat_out(self, cat):
        self._assert_cat_common_parts(cat)
        self.assertEqual(cat[0].comments[0].text, self.out_file_header)
        self.assertEqual(cat[0].focal_mechanisms[0].comments[0].text,
                         out_file_first_comment)
        for focmec in cat[0].focal_mechanisms:
            # misfit should be None, because the file specifies that polarity
            # errors are weighted and in the out file format we can't know how
            # many individual errors there are
            self.assertEqual(focmec.misfit, None)
            # we can't tell the gap from the out format
            self.assertEqual(focmec.azimuthal_gap, None)

    def _assert_cat_lst(self, cat):
        self._assert_cat_common_parts(cat)
        self.assertEqual(cat[0].comments[0].text, self.lst_file_header)
        self.assertEqual(cat[0].focal_mechanisms[0].comments[0].text,
                         lst_file_first_comment)
        for focmec in cat[0].focal_mechanisms:
            # misfit should be 0.0, because in the lst file we can count the
            # number of individual errors (and there's no polarity errors)
            self.assertEqual(focmec.misfit, 0.0)
            # we can't tell the gap from the out format
            self.assertEqual(focmec.azimuthal_gap, 236.7)

    def test_is_focmec(self):
        for file_ in (self.lst_file, self.out_file):
            self.assertTrue(_is_focmec(file_))

    def test_read_focmec_out(self):
        cat = _read_focmec(self.out_file)
        self._assert_cat_out(cat)

    def test_read_focmec_lst(self):
        cat = _read_focmec(self.lst_file)
        self._assert_cat_lst(cat)

    def test_read_focmec_out_through_plugin(self):
        cat = read_events(self.out_file)
        self._assert_cat_out(cat)

    def test_read_focmec_lst_through_plugin(self):
        cat = read_events(self.lst_file)
        self._assert_cat_lst(cat)


def suite():
    return unittest.makeSuite(FOCMECTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
