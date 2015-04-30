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
        Tests that reading and writing CMTSOLUTIONS file does not change
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

        for line1, line2 in zip(data.decode().splitlines(),
                new_data.decode().splitlines()):
            self.assertEqual(line1, line2)

    def test_read_and_write_cmtsolution_from_open_files(self):
        """
        Tests that reading and writing CMTSOLUTIONS file does not change
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

        for line1, line2 in zip(data.decode().splitlines(),
                new_data.decode().splitlines()):
            self.assertEqual(line1, line2)

    def test_read_and_write_cmtsolution_from_bytes_io(self):
        """
        Tests that reading and writing CMTSOLUTIONS file does not change
        anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION")
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            fh.seek(0, 0)
            data = fh.read()

        with buf:
            buf.seek(0, 0)
            cat = obspy.read_events(buf)

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        for line1, line2 in zip(data.decode().splitlines(),
                new_data.decode().splitlines()):
            self.assertEqual(line1, line2)

    def test_read_and_write_cmtsolution_explosion(self):
        """
        Tests that reading and writing CMTSOLUTIONS file does not change
        anything.

        Tests another file.
        """
        filename = os.path.join(self.datapath, "CMTSOLUTION_EXPLOSION")
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            fh.seek(0, 0)
            data = fh.read()

        with buf:
            buf.seek(0, 0)
            cat = obspy.read_events(buf)

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        for line1, line2 in zip(data.decode().splitlines(),
                                new_data.decode().splitlines()):
            self.assertEqual(line1, line2)


def suite():
    return unittest.makeSuite(CmtsolutionTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
