#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import numpy as np
import warnings

import obspy
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.scardec.core import _is_scardec


class TestScardec():
    """
    Test suite for obspy.io.scardec.

    The tests usually directly utilize the registered function with the
    read_events() to also test the integration.
    """
    def test_read_and_write_scardec_from_files(self, testdata):
        """
        Tests that reading and writing a SCARDECfile does not change
        anything.
        Note: The test file is not one from the catalogue, since it was
              impossible to recreate the number formatting. Therefore, the test
              file has been created with ObsPy, but was manually checked to be
              consistent with the original file
        """
        filename = testdata['test.scardec']
        with open(filename, "rb") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)

        with NamedTemporaryFile() as tf:
            temp_filename = tf.name

        try:
            # raises two UserWarnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', UserWarning)

                cat.write(temp_filename, format="SCARDEC")

                assert len(w) == 2
                assert w[0].category == UserWarning
                assert 'No moment wave magnitude found' in str(w[0])
                assert w[1].category == UserWarning
                assert 'No derived origin attached' in str(w[1])

            with open(temp_filename, "rb") as fh:
                new_data = fh.read()
        finally:
            try:
                os.remove(temp_filename)
            except Exception:
                pass

        # Test file header
        assert data.decode().splitlines()[0:2] == \
            new_data.decode().splitlines()[0:2]

        for line_data, line_new in zip(data.decode().splitlines()[2:],
                                       new_data.decode().splitlines()[2:]):
            # Compare time stamps
            assert np.allclose(float(line_data.split()[0]),
                               float(line_new.split()[0]))
            # Compare moment rate values
            assert np.allclose(float(line_data.split()[1]),
                               float(line_new.split()[1]))

    def test_read_and_write_scardec_from_open_files(self, testdata):
        """
        Tests that reading and writing a SCARDEC file does not change
        anything.

        This time it tests reading from and writing to open files.
        """
        filename = testdata['test.scardec']
        with open(filename, "rb") as fh:
            data = fh.read()
            fh.seek(0, 0)
            cat = obspy.read_events(fh)

        with NamedTemporaryFile() as tf:
            # raises two UserWarnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', UserWarning)

                cat.write(tf, format="SCARDEC")
                tf.seek(0, 0)
                new_data = tf.read()

                assert len(w) == 2
                assert w[0].category == UserWarning
                assert 'No moment wave magnitude found' in str(w[0])
                assert w[1].category == UserWarning
                assert 'No derived origin attached' in str(w[1])

        # Test file header
        assert data.decode().splitlines()[0:2] == \
            new_data.decode().splitlines()[0:2]

        for line_data, line_new in zip(data.decode().splitlines()[2:],
                                       new_data.decode().splitlines()[2:]):
            # Compare time stamps
            assert np.allclose(float(line_data.split()[0]),
                               float(line_new.split()[0]))
            # Compare moment rate values
            assert np.allclose(float(line_data.split()[1]),
                               float(line_new.split()[1]))

    def test_read_and_write_scardec_from_bytes_io(self, testdata):
        """
        Tests that reading and writing a SCARDEC file does not change
        anything.

        This time it tests reading from and writing to BytesIO objects.
        """
        filename = testdata['test.scardec']

        with open(filename, "rb") as fh:
            buf = io.BytesIO(fh.read())
            data = buf.read()
            buf.seek(0, 0)

            with buf:
                buf.seek(0, 0)
                cat = obspy.read_events(buf)

                # raises two UserWarnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', UserWarning)

                    with io.BytesIO() as buf2:
                        cat.write(buf2, format="SCARDEC")
                        buf2.seek(0, 0)
                        new_data = buf2.read()

                assert len(w) == 2
                assert w[0].category == UserWarning
                assert 'No moment wave magnitude found' in str(w[0])
                assert w[1].category == UserWarning
                assert 'No derived origin attached' in str(w[1])

        # Test file header
        assert data.decode().splitlines()[0:2] == \
            new_data.decode().splitlines()[0:2]

        for line_data, line_new in zip(data.decode().splitlines()[2:],
                                       new_data.decode().splitlines()[2:]):
            # Compare time stamps
            assert np.allclose(float(line_data.split()[0]),
                               float(line_new.split()[0]))
            # Compare moment rate values
            assert np.allclose(float(line_data.split()[1]),
                               float(line_new.split()[1]))

    def test_is_scardec(self, testdata, datapath):
        """
        Tests the is_scardec function.
        """
        good_files = [testdata['test.scardec'],
                      testdata['test2.scardec']]

        bad_files = [
            datapath.parent / "test_core.py",
            datapath.parent / "__init__.py"]

        for filename in good_files:
            assert _is_scardec(filename)
        for filename in bad_files:
            assert not _is_scardec(filename)
