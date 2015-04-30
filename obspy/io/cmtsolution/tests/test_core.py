#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
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
        with open(filename, "rt") as fh:
            data = fh.read()

        cat = obspy.read_events(filename)

        with NamedTemporaryFile() as tf:
            temp_filename = tf.name

        try:
            cat.write(temp_filename, format="CMTSOLUTION")
            with open(temp_filename, "rt") as fh:
                new_data = fh.read()
        finally:
            try:
                os.remove(temp_filename)
            except:
                pass

        self.assertEqual(data, new_data)


def suite():
    return unittest.makeSuite(CmtsolutionTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
