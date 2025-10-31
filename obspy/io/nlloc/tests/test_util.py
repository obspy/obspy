#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from obspy.io.nlloc.util import read_nlloc_scatter


def _coordinate_conversion(x, y, z):
    try:
        transformer = pyproj.Transformer.from_crs("epsg:31468", "epsg:4326",
                                                  always_xy=True)
    except Exception:
        # pyproj version < 2.2
        proj_wgs84 = pyproj.Proj(init="epsg:4326")
        proj_gk4 = pyproj.Proj(init="epsg:31468")
        x, y = pyproj.transform(proj_gk4, proj_wgs84, x * 1e3, y * 1e3)
    else:
        x, y = transformer.transform(x * 1e3, y * 1e3)
    return x, y, z


class TestNLLOC():
    """
    Test suite for obspy.io.nlloc
    """
    def test_read_nlloc_scatter_plain(self, testdata):
        """
        Test reading NLLoc scatter file without coordinate conversion.
        """
        # test without coordinate manipulation
        filename = testdata['nlloc.scat']
        got = read_nlloc_scatter(filename)
        filename = testdata['nlloc_scat.npy']
        expected = np.load(filename)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.skipif(not HAS_PYPROJ, reason='pyproj not installed')
    def test_read_nlloc_scatter_coordinate_conversion(self, testdata):
        """
        Test reading NLLoc scatter file including coordinate conversion.
        """
        # test including coordinate manipulation
        filename = testdata['nlloc.scat']
        got = read_nlloc_scatter(
            filename, coordinate_converter=_coordinate_conversion)
        filename = testdata['nlloc_scat_converted.npy']
        expected = np.load(filename)
        for field in expected.dtype.fields:
            np.testing.assert_allclose(got[field], expected[field], rtol=1e-5)
