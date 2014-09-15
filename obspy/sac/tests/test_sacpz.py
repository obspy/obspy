# -*- coding: utf-8 -*-
"""
The sac.sacpz test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest
import os
from obspy import read_inventory
from obspy.core.util import NamedTemporaryFile


class SACPZTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        # directory where the test files are located
        self.path = os.path.dirname(__file__)
        # these files were checked against data given by IRIS SACPZ web service
        # http://service.iris.edu/irisws/sacpz/1/
        #                                query?net=IU&loc=*&cha=BH?&sta=ANMO
        # DIP seems to be systematically different in SACPZ output compared to
        # StationXML served by IRIS...
        self.file1 = os.path.join(self.path, 'data', 'IU_ANMO_00_BHZ.sacpz')
        self.file2 = os.path.join(self.path, 'data', 'IU_ANMO_BH.sacpz')

    def test_write_SACPZ_single_channel(self):
        """
        """
        inv = read_inventory("/path/to/IU_ANMO_00_BHZ.xml")
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            inv.write(tempfile, format='SACPZ')
            with open(tempfile) as fh:
                got = fh.read()
        with open(self.file1) as fh:
            expected = fh.read()
        # remove CREATED line that changes
        got = [l for l in got.split("\n") if "CREATED" not in l]
        expected = [l for l in expected.split("\n") if "CREATED" not in l]
        self.assertEqual(got, expected)

    def test_write_SACPZ_multiple_channels(self):
        """
        """
        inv = read_inventory("/path/to/IU_ANMO_BH.xml")
        with NamedTemporaryFile() as tf:
            tempfile = tf.name
            inv.write(tempfile, format='SACPZ')
            with open(tempfile) as fh:
                got = fh.read()
        with open(self.file2) as fh:
            expected = fh.read()
        # remove CREATED line that changes
        got = [l for l in got.split("\n") if "CREATED" not in l]
        expected = [l for l in expected.split("\n") if "CREATED" not in l]
        self.assertEqual(got, expected)


def suite():
    return unittest.makeSuite(SACPZTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
