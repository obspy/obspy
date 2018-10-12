# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import copy
import shutil
import unittest

from obspy.core.compatibility import mock
from obspy.core.util.base import (NamedTemporaryFile, get_dependency_version,
                                  download_to_file, sanitize_filename,
                                  create_empty_data_chunk, ComparingObject)
from obspy.core.util.testing import ImageComparison, ImageComparisonException

import numpy as np
from requests import HTTPError


class UtilBaseTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.base
    """
    def test_get_matplotlib_version(self):
        """
        Tests for the get_matplotlib_version() function as it continues to
        cause problems.
        """
        versions = (("1.2.3", [1, 2, 3]), ("0.9.11", [0, 9, 11]),
                    ("0.9.svn", [0, 9, 0]), ("1.1.1~rc1-1", [1, 1, 1]),
                    ("1.2.x", [1, 2, 0]), ("1.3.1rc2", [1, 3, 1]))

        for version_string, expected in versions:
            with mock.patch('pkg_resources.get_distribution') as p:
                class _D(object):
                    version = version_string
                p.return_value = _D()
                got = get_dependency_version('matplotlib')
            self.assertEqual(expected, got)

    def test_named_temporay_file__context_manager(self):
        """
        Tests the automatic closing/deleting of NamedTemporaryFile using the
        context manager.
        """
        content = b"burn after writing"
        # write something to tempfile and check closing/deletion afterwards
        with NamedTemporaryFile() as tf:
            filename = tf.name
            tf.write(content)
        self.assertFalse(os.path.exists(filename))
        # write something to tempfile and check that it is written correctly
        with NamedTemporaryFile() as tf:
            filename = tf.name
            tf.write(content)
            tf.close()
            with open(filename, 'rb') as fh:
                tmp_content = fh.read()
        self.assertEqual(content, tmp_content)
        self.assertFalse(os.path.exists(filename))
        # check that closing/deletion works even when nothing is done with file
        with NamedTemporaryFile() as tf:
            filename = tf.name
        self.assertFalse(os.path.exists(filename))

    def test_image_comparison(self):
        """
        Tests the image comparison mechanism with an expected fail and an
        expected passing test.
        Also tests that temporary files are deleted after both passing and
        failing tests.
        """
        path = os.path.join(os.path.dirname(__file__), "images")
        img_basename = "image.png"
        img_ok = os.path.join(path, "image_ok.png")
        img_fail = os.path.join(path, "image_fail.png")

        # image comparison that should pass
        with ImageComparison(path, img_basename) as ic:
            shutil.copy(img_ok, ic.name)
            self.assertTrue(os.path.exists(ic.name))
        # check that temp file is deleted
        self.assertFalse(os.path.exists(ic.name))

        # image comparison that should raise
        # avoid uploading the staged test fail image
        # (after an estimate of 10000 uploads of it.. ;-))
        with self.assertRaises(ImageComparisonException):
            with ImageComparison(path, img_basename, adjust_tolerance=False,
                                 no_uploads=True) as ic:
                shutil.copy(img_fail, ic.name)

        # check that temp file is deleted
        self.assertFalse(os.path.exists(ic.name))

    def test_mock_read_inventory_http_errors(self):
        """
        Tests HTTP Error on 204, 400, and 500
        """
        url = "http://obspy.org"
        for response_tuple in [("204", "No Content"), ("400", "Bad Request"),
                               ("500", "Internal Server Error")]:
            code = response_tuple[0]
            reason = response_tuple[1]
            with mock.patch("requests.get") as mocked_get:
                mocked_get.return_value.status_code = code
                mocked_get.return_value.reason = reason
                with self.assertRaises(HTTPError) as e:
                    download_to_file(url, None)
                self.assertEqual(e.exception.args[0],
                                 "%s HTTP Error: %s for url: %s" %
                                 (code, reason, url))

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("example.mseed"),
                         "example.mseed")
        self.assertEqual(sanitize_filename("Example.mseed"),
                         "Example.mseed")
        self.assertEqual(sanitize_filename("example.mseed?raw=True"),
                         "example.mseedrawTrue")
        self.assertEqual(sanitize_filename("Example.mseed?raw=true"),
                         "Example.mseedrawtrue")

    def test_create_empty_data_chunk(self):
        out = create_empty_data_chunk(3, 'int', 10)
        self.assertIsInstance(out, np.ndarray)
        # The default dtype for an integer (np.int_) is a `C long` which is
        # only 32 bits on windows. Thus we have to allow both.
        self.assertIn(out.dtype, (np.int32, np.int64))
        np.testing.assert_allclose(out, [10, 10, 10])

        out = create_empty_data_chunk(6, np.complex128, 0)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.complex128)
        np.testing.assert_allclose(out, np.zeros(6, dtype=np.complex128))

        # Fully masked output.
        out = create_empty_data_chunk(3, 'f')
        self.assertIsInstance(out, np.ma.MaskedArray)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out.mask, [True, True, True])

    def test_comparing_object_eq(self):
        co = ComparingObject()
        # Compare to other types
        self.assertNotEqual(co, 5)
        self.assertNotEqual(co, None)
        self.assertNotEqual(co, object())
        # Compare same type, different instance, with attributes
        co.at = 3
        deep_copy = copy.deepcopy(co)
        self.assertEqual(co, deep_copy)
        deep_copy.at = 0
        self.assertNotEqual(co, deep_copy)


def suite():
    return unittest.makeSuite(UtilBaseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
