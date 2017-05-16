# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import itertools
import os
import unittest

from obspy.io.xseed.core import _is_resp, _is_xseed, _is_seed, _read_resp, \
    _read_seed


class CoreTestCase(unittest.TestCase):
    """
    Test integration with ObsPy's inventory objects.
    """
    def setUp(self):
        self.data_path = os.path.join(os.path.dirname(__file__), "data")

        self.seed_files = [
            "AI.ESPZ._.BHE.dataless",
            "AI.ESPZ._.BH_.dataless",
            "BN.LPW._.BHE.dataless",
            "CL.AIO.dataless",
            "G.SPB.dataless",
            "arclink_full.seed",
            "bug165.dataless",
            "dataless.seed.BW_DHFO",
            "dataless.seed.BW_FURT",
            "dataless.seed.BW_MANZ",
            "dataless.seed.BW_RJOB",
            "dataless.seed.BW_ROTZ",
            "dataless.seed.BW_ZUGS",
            "dataless.seed.II_COCO"
        ]
        self.xseed_files = ["dataless.seed.BW_FURT.xml"]

        self.resp_files = ["RESP.BW.FURT..EHZ",
                           "RESP.XX.NR008..HHZ.130.1.100",
                           "RESP.XX.NS085..BHZ.STS2_gen3.120.1500"]
        self.other_files = ["II_COCO_three_channel_borehole.mseed",
                            "xml-seed-1.0.xsd",
                            "xml-seed-1.1.xsd"]

        self.seed_files = [
            os.path.join(self.data_path, _i) for _i in self.seed_files]
        self.xseed_files = [
            os.path.join(self.data_path, _i) for _i in self.xseed_files]
        self.resp_files = [
            os.path.join(self.data_path, _i) for _i in self.resp_files]
        self.other_files = [
            os.path.join(self.data_path, _i) for _i in self.other_files]

        for _i in itertools.chain.from_iterable([
                self.seed_files, self.xseed_files, self.resp_files,
                self.other_files]):
            assert os.path.exists(_i), _i

    def test_is_seed(self):
        for filename in self.seed_files:
            self.assertTrue(_is_seed(filename), filename)

        for filename in self.xseed_files:
            self.assertFalse(_is_seed(filename), filename)

        for filename in self.resp_files:
            self.assertFalse(_is_seed(filename), filename)

        for filename in self.other_files:
            self.assertFalse(_is_seed(filename), filename)

    def test_is_xseed(self):
        for filename in self.seed_files:
            self.assertFalse(_is_xseed(filename), filename)

        for filename in self.xseed_files:
            self.assertTrue(_is_xseed(filename), filename)

        for filename in self.resp_files:
            self.assertFalse(_is_xseed(filename), filename)

        for filename in self.other_files:
            self.assertFalse(_is_xseed(filename), filename)

    def test_is_resp(self):
        for filename in self.seed_files:
            self.assertFalse(_is_resp(filename), filename)

        for filename in self.xseed_files:
            self.assertFalse(_is_resp(filename), filename)

        for filename in self.resp_files:
            self.assertTrue(_is_resp(filename), filename)

        for filename in self.other_files:
            self.assertFalse(_is_resp(filename), filename)

    def test_read_resp(self):
        f = self.resp_files[1]
        _read_resp(f)

    def test_read_seed(self):
        for f in self.seed_files:
            print("FILENAME:", f)
            _read_seed(f)


def suite():
    return unittest.makeSuite(CoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
