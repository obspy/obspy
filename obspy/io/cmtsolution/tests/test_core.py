#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.cmtsolution.core import _is_cmtsolution


class TestCmtsolution():
    """
    Test suite for obspy.io.cmtsolution.

    The tests usually directly utilize the registered function with the
    read_events() to also test the integration.
    """
    def test_read_and_write_cmtsolution_from_files(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.
        """
        filename = testdata['CMTSOLUTION']
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

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_write_no_preferred_focal_mechanism(self, testdata):
        """
        Tests that writing a CMTSOLUTION file with no preferred (but at least
        one) focal mechanism works, see #1303.
        """
        filename = testdata['CMTSOLUTION']
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

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_read_and_write_cmtsolution_from_open_files(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        This time it tests reading from and writing to open files.
        """
        filename = testdata['CMTSOLUTION']
        with open(filename, "rb") as fh:
            data = fh.read()
            fh.seek(0, 0)
            cat = obspy.read_events(fh)

        with NamedTemporaryFile() as tf:
            cat.write(tf, format="CMTSOLUTION")
            tf.seek(0, 0)
            new_data = tf.read()

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_read_and_write_cmtsolution_from_bytes_io(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = testdata['CMTSOLUTION']
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

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_read_and_write_cmtsolution_explosion(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file does not change
        anything.

        Tests another file.
        """
        filename = testdata['CMTSOLUTION_EXPLOSION']
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

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_is_cmtsolution(self, testdata, datapath):
        """
        Tests the is_cmtsolution function.
        """
        good_files = [testdata['CMTSOLUTION'],
                      testdata['CMTSOLUTION_EXPLOSION']]

        bad_files = [datapath.parent / "test_core.py",
                     datapath.parent / "__init__.py"]

        for filename in good_files:
            assert _is_cmtsolution(filename)
        for filename in bad_files:
            assert not _is_cmtsolution(filename)

    def test_read_and_write_multiple_cmtsolution_from_files(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file with multiple
        events does not change anything.
        """
        filename = testdata['MULTIPLE_EVENTS']
        with open(filename, "rb") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)

        assert len(cat) == 4

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

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_read_and_write_multiple_events_from_bytes_io(self, testdata):
        """
        Tests that reading and writing a CMTSOLUTION file with multiple
        events does not change anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = testdata['MULTIPLE_EVENTS']
        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            data = buf.read()
            buf.seek(0, 0)

        with buf:
            buf.seek(0, 0)
            cat = obspy.read_events(buf)

            assert len(cat) == 4

            with io.BytesIO() as buf2:
                cat.write(buf2, format="CMTSOLUTION")
                buf2.seek(0, 0)
                new_data = buf2.read()

        assert data.decode().splitlines() == new_data.decode().splitlines()

    def test_reading_newer_cmtsolution_files(self, testdata):
        """
        The format changed a bit. Make sure these files can also be read.
        """
        filename = testdata['CMTSOLUTION_NEW']
        cat = obspy.read_events(filename)

        assert len(cat) == 3

        # Test the hypocentral origins as the "change" to the format only
        # affected the first line.
        assert cat[0].origins[1].latitude == 55.29
        assert cat[0].origins[1].longitude == 163.06

        assert cat[1].origins[1].latitude == -13.75
        assert cat[1].origins[1].longitude == -111.75

        assert cat[2].origins[1].latitude == -13.68
        assert cat[2].origins[1].longitude == -111.93
