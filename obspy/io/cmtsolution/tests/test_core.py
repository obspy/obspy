#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

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
            except:
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
            except:
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


def suite():
    return unittest.makeSuite(CmtsolutionTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
