#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import io
import os
import unittest

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.cmtsolution.core import _is_cmtsolution


class CmtsolutionTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.cmtsolution.

    The tests usually directly utilize the registered function with the
    read_events() to also test the integration.
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_read_and_write_cmtsolution_from_files(self):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION")
        with open(filename, "rb") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)

        with NamedTemporaryFile() as tf:
            temp_filename = tf.name

        try:
            cat.write(temp_filename, format="CMTSOLUTION")
            with open(temp_filename, "rb") as fh:
                new_data = fh.read()
        finally:
            try:
                os.remove(temp_filename)
            except Exception:
                pass

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_write_no_preferred_focal_mechanism(self):
        """
        Tests that writing a CMTSOLUTION file with no preferred (but at least
        one) focal mechanism works, see #1303.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION")
        with open(filename, "rb") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)
        cat[0].preferred_focal_mechanism_id = None

        with NamedTemporaryFile() as tf:
            temp_filename = tf.name

        try:
            cat.write(temp_filename, format="CMTSOLUTION")
            with open(temp_filename, "rb") as fh:
                new_data = fh.read()
        finally:
            try:
                os.remove(temp_filename)
            except Exception:
                pass

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_read_and_write_cmtsolution_from_open_files(self):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        This time it tests reading from and writing to open files.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION")
        with open(filename, "rb") as fh:
            data = fh.read()
            fh.seek(0, 0)
            cat = obspy.read_events(fh)

        with NamedTemporaryFile() as tf:
            cat.write(tf, format="CMTSOLUTION")
            tf.seek(0, 0)
            new_data = tf.read()

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_read_and_write_cmtsolution_from_bytes_io(self):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION")
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            data = buf.read()
            buf.seek(0, 0)

        with buf:
            buf.seek(0, 0)
            cat = obspy.read_events(buf)

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_read_and_write_cmtsolution_explosion(self):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        Tests another file.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION_EXPLOSION")
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())

        data = buf.read()
        buf.seek(0, 0)

        with buf:
            cat = obspy.read_events(buf)

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_is_cmtsolution(self):
        """
        Tests the is_cmtsolution function.
        """
        good_files = [os.path.join(self.datapath, "CMTSOLUTION"),
                      os.path.join(self.datapath, "CMTSOLUTION_EXPLOSION")]

        bad_files = [
            os.path.join(self.datapath, os.path.pardir, "test_core.py"),
            os.path.join(self.datapath, os.path.pardir, "__init__.py")]

        for filename in good_files:
            self.assertTrue(_is_cmtsolution(filename))
        for filename in bad_files:
            self.assertFalse(_is_cmtsolution(filename))

    def test_read_and_write_multiple_cmtsolution_from_files(self):
        """
        Tests that reading and writing a CMTSOLUTION file with multiple
        events does not change anything.
        """
        filename = os.path.join(self.datapath, "MULTIPLE_EVENTS")
        with open(filename, "rb") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)

        self.assertEqual(len(cat), 4)

        with NamedTemporaryFile() as tf:
            temp_filename = tf.name

        try:
            cat.write(temp_filename, format="CMTSOLUTION")
            with open(temp_filename, "rb") as fh:
                new_data = fh.read()
        finally:
            try:
                os.remove(temp_filename)
            except Exception:
                pass

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_read_and_write_multiple_events_from_bytes_io(self):
        """
        Tests that reading and writing a CMTSOLUTION file with multiple
        events does not change anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = os.path.join(self.datapath, "MULTIPLE_EVENTS")
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            data = buf.read()
            buf.seek(0, 0)

        with buf:
            buf.seek(0, 0)
            cat = obspy.read_events(buf)

            self.assertEqual(len(cat), 4)

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        self.assertEqual(data.decode().splitlines(),
                         new_data.decode().splitlines())

    def test_reading_newer_cmtsolution_files(self):
        """
        The format changed a bit. Make sure these files can also be read.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION_NEW")
        cat = obspy.read_events(filename)

        self.assertEqual(len(cat), 3)

        # Test the hypocentral origins as the "change" to the format only
        # affected the first line.
        self.assertEqual(cat[0].origins[1].latitude, 55.29)
        self.assertEqual(cat[0].origins[1].longitude, 163.06)

        self.assertEqual(cat[1].origins[1].latitude, -13.75)
        self.assertEqual(cat[1].origins[1].longitude, -111.75)

        self.assertEqual(cat[2].origins[1].latitude, -13.68)
        self.assertEqual(cat[2].origins[1].longitude, -111.93)
