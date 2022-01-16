# -*- coding: utf-8 -*-
import os
import copy
from unittest import mock

import numpy as np
from requests import HTTPError
import pytest

from obspy.core.util.base import (NamedTemporaryFile, get_dependency_version,
                                  download_to_file, sanitize_filename,
                                  create_empty_data_chunk, ComparingObject)


class TestUtilBase:
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
            assert expected == got

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
        assert not os.path.exists(filename)
        # write something to tempfile and check that it is written correctly
        with NamedTemporaryFile() as tf:
            filename = tf.name
            tf.write(content)
            tf.close()
            with open(filename, 'rb') as fh:
                tmp_content = fh.read()
        assert content == tmp_content
        assert not os.path.exists(filename)
        # check that closing/deletion works even when nothing is done with file
        with NamedTemporaryFile() as tf:
            filename = tf.name
        assert not os.path.exists(filename)

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
                msg = "%s HTTP Error: %s for url: %s" % (code, reason, url)
                with pytest.raises(HTTPError, match=msg):
                    download_to_file(url, None)

    def test_sanitize_filename(self):
        assert sanitize_filename("example.mseed") == \
               "example.mseed"
        assert sanitize_filename("Example.mseed") == \
               "Example.mseed"
        assert sanitize_filename("example.mseed?raw=True") == \
               "example.mseedrawTrue"
        assert sanitize_filename("Example.mseed?raw=true") == \
               "Example.mseedrawtrue"

    def test_create_empty_data_chunk(self):
        out = create_empty_data_chunk(3, 'int', 10)
        assert isinstance(out, np.ndarray)
        # The default dtype for an integer (np.int_) is a `C long` which is
        # only 32 bits on windows. Thus we have to allow both.
        assert out.dtype in (np.int32, np.int64)
        np.testing.assert_allclose(out, [10, 10, 10])

        out = create_empty_data_chunk(6, np.complex128, 0)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.complex128
        np.testing.assert_allclose(out, np.zeros(6, dtype=np.complex128))

        # Fully masked output.
        out = create_empty_data_chunk(3, 'f')
        assert isinstance(out, np.ma.MaskedArray)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out.mask, [True, True, True])

    def test_comparing_object_eq(self):
        co = ComparingObject()
        # Compare to other types
        assert co != 5
        assert co is not None
        assert co != object()
        # Compare same type, different instance, with attributes
        co.at = 3
        deep_copy = copy.deepcopy(co)
        assert co == deep_copy
        deep_copy.at = 0
        assert co != deep_copy
