#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import unittest
import warnings

from obspy import read_events
from obspy.io.cnv.core import _write_cnv
from obspy.core.util import NamedTemporaryFile


class CNVTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.cnv
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_write_cnv(self):
        """
        Test writing CNV catalog summary file.
        """
        # load QuakeML file to generate CNV file from it
        filename = os.path.join(self.datapath, "obspyck_20141020150701.xml")
        cat = read_events(filename, format="QUAKEML")

        # read expected OBS file output
        filename = os.path.join(self.datapath, "obspyck_20141020150701.cnv")
        with open(filename, "rb") as fh:
            expected = fh.read().decode()

        # write via plugin
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="CNV")
            tf.seek(0)
            got = tf.read().decode()

        self.assertEqual(expected.splitlines(), got.splitlines())

        # write manually
        with NamedTemporaryFile() as tf:
            _write_cnv(cat, tf)
            tf.seek(0)
            got = tf.read().decode()

        self.assertEqual(expected.splitlines(), got.splitlines())

        # write via plugin and with phase_mapping
        with NamedTemporaryFile() as tf:
            cat.write(tf, format="CNV", phase_mapping={"P": "P", "S": "S"})
            tf.seek(0)
            got = tf.read().decode()

        self.assertEqual(expected.splitlines(), got.splitlines())

        # write via plugin and with phase_mapping with only P
        # read expected OBS file output
        filename = os.path.join(self.datapath, "obspyck_20141020150701_P.cnv")
        with open(filename, "rb") as fh:
            expected = fh.read().decode()

        with NamedTemporaryFile() as tf:
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")
                cat.write(tf, format="CNV", phase_mapping={"P": "P"})
                tf.seek(0)
                got = tf.read().decode()
                # There should be 4 S warnings for the 4 S phases:
                self.assertEqual(len(w), 4)
                assert "with unmapped phase hint: S" in str(w[-1].message)

        self.assertEqual(expected.splitlines(), got.splitlines())
