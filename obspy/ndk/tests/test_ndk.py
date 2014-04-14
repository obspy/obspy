#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
import unittest
import warnings

from obspy import readEvents
from obspy.ndk.core import is_ndk, read_ndk


class NDKTestCase(unittest.TestCase):
    """
    Test suite for obspy.ndk
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")

    def test_read_single_ndk(self):
        """
        Test reading a single event from and NDK file and comparing it to a
        QuakeML file that has been manually checked to contain all the
        information in the NDK file.
        """
        filename = os.path.join(self.datapath, "C200604092050A.ndk")
        cat = read_ndk(filename)

        reference = os.path.join(self.datapath, "C200604092050A.xml")
        ref_cat = readEvents(reference)

        self.assertEqual(cat, ref_cat)

    def test_read_multiple_events(self):
        """
        Tests the reading of multiple events in one file. The file has been
        edited to test a variety of settings.
        """
        filename = os.path.join(self.datapath, "multiple_events.ndk")
        cat = read_ndk(filename)

        self.assertEqual(len(cat), 6)

        # Test the type of moment tensor inverted for.
        self.assertEqual([i.focal_mechanisms[0].moment_tensor.inversion_type
                          for i in cat],
                         ["general", "zero trace", "double couple"] * 2)

        # Test the type and duration of the moment rate function.
        self.assertEqual(
            [i.focal_mechanisms[0].moment_tensor.source_time_function.type
             for i in cat],
            ["triangle", "box car"] * 3)
        self.assertEqual(
            [i.focal_mechanisms[0].moment_tensor.source_time_function.duration
             for i in cat],
            [2.6, 7.4, 9.0, 1.8, 2.0, 1.6])

        # Test the type of depth setting.
        self.assertEqual([i.preferred_origin().depth_type for i in cat],
                         ["from moment tensor inversion", "from location",
                          "from modeling of broad-band P waveforms"] * 2)

        # Check the solution type.
        for event in cat[:3]:
            self.assertTrue("Standard" in
                            event.focal_mechanisms[0].comments[0].text)
        for event in cat[3:]:
            self.assertTrue("Quick" in
                            event.focal_mechanisms[0].comments[0].text)

    def test_is_ndk(self):
        """
        Test for the the is_ndk() function.
        """
        valid_files = [os.path.join(self.datapath, "C200604092050A.ndk"),
                       os.path.join(self.datapath, "multiple_events.ndk")]
        invalid_files = []
        for filename in os.listdir(self.path):
            if filename.endswith(".py"):
                invalid_files.append(os.path.join(self.path, filename))
        self.assertTrue(len(invalid_files) > 0)

        for filename in valid_files:
            self.assertTrue(is_ndk(filename))
        for filename in invalid_files:
            self.assertFalse(is_ndk(filename))

    def test_file_with_faulty_timestamp(self):
        """
        Tests a file with timestamp 'O-0000000000000'. While this is wrong
        according to the specification it is unfortunately present in the
        GlobalCMT catalog and thus only a warning should be raised.
        """
        filename = os.path.join(self.datapath,
                                "file_with_faulty_timestamp.ndk")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cat = read_ndk(filename)

        self.assertEqual(len(w), 1)
        self.assertTrue("Invalid CMT timestamp 'O-000000" in str(w[0]))
        self.assertTrue("Unknown" in
                        cat[0].focal_mechanisms[0].comments[0].text)

    def test_reading_using_obspy_plugin(self):
        """
        Checks that reading with the readEvents() function works correctly.
        """
        filename = os.path.join(self.datapath, "C200604092050A.ndk")
        cat = readEvents(filename)

        reference = os.path.join(self.datapath, "C200604092050A.xml")
        ref_cat = readEvents(reference)

        self.assertEqual(cat, ref_cat)


def suite():
    return unittest.makeSuite(NDKTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
