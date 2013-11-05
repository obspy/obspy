# -*- coding: utf-8 -*-
import unittest
from obspy.core.util.decorator import map_example_filename
from obspy.core.util import getExampleFile


class TestCase(unittest.TestCase):
    def test_map_example_filename(self):
        """
        Tests the @map_example_filename decorator
        """
        dummy = "abc"
        example_file = "example.npz"
        path = "/path/to/" + example_file
        path_mapped = getExampleFile(example_file)

        def unchanged(a, b="", **kwargs):
            return map(str, (a, b, kwargs))

        @map_example_filename("a")
        def changed(a, b="", **kwargs):
            return map(str, (a, b, kwargs))
        self.assertEqual(
            changed(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed(path, dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed(dummy, path), unchanged(dummy, path))
        self.assertEqual(
            changed(a=path, b=dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed(path, b=dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed(path, b=path, x=path),
            unchanged(path_mapped, path, x=path))

        @map_example_filename("b")
        def changed(a, b="", **kwargs):
            return map(str, (a, b, kwargs))
        self.assertEqual(
            changed(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed(path, dummy), unchanged(path, dummy))
        self.assertEqual(
            changed(dummy, path), unchanged(dummy, path_mapped))
        self.assertEqual(
            changed(a=path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed(path, b=path), unchanged(path, path_mapped))
        self.assertEqual(
            changed(path, b=path, x=path),
            unchanged(path, path_mapped, x=path))

        @map_example_filename("x")
        def changed(a, b="", **kwargs):
            return map(str, (a, b, kwargs))
        self.assertEqual(
            changed(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed(path, dummy), unchanged(path, dummy))
        self.assertEqual(
            changed(dummy, path), unchanged(dummy, path))
        self.assertEqual(
            changed(a=path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed(path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed(path, b=path, x=path),
            unchanged(path, path, x=path_mapped))


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
