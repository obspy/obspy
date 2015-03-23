# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import unittest

from obspy.core.util import getExampleFile
from obspy.core.util.decorator import map_example_filename


class TestCase(unittest.TestCase):
    def test_map_example_filename(self):
        """
        Tests the @map_example_filename decorator
        """
        dummy = "abc"
        example_file = "slist.ascii"
        path = "/path/to/" + example_file
        path_mapped = getExampleFile(example_file)

        def unchanged(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))

        @map_example_filename("a")
        def changed1(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        self.assertEqual(
            changed1(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed1(path, dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed1(dummy, path), unchanged(dummy, path))
        self.assertEqual(
            changed1(a=path, b=dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed1(path, b=dummy), unchanged(path_mapped, dummy))
        self.assertEqual(
            changed1(path, b=path, x=path),
            unchanged(path_mapped, path, x=path))

        @map_example_filename("b")
        def changed2(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        self.assertEqual(
            changed2(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed2(path, dummy), unchanged(path, dummy))
        self.assertEqual(
            changed2(dummy, path), unchanged(dummy, path_mapped))
        self.assertEqual(
            changed2(a=path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed2(path, b=path), unchanged(path, path_mapped))
        self.assertEqual(
            changed2(path, b=path, x=path),
            unchanged(path, path_mapped, x=path))

        @map_example_filename("x")
        def changed3(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        self.assertEqual(
            changed3(dummy, dummy), unchanged(dummy, dummy))
        self.assertEqual(
            changed3(path, dummy), unchanged(path, dummy))
        self.assertEqual(
            changed3(dummy, path), unchanged(dummy, path))
        self.assertEqual(
            changed3(a=path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed3(path, b=dummy), unchanged(path, dummy))
        self.assertEqual(
            changed3(path, b=path, x=path),
            unchanged(path, path, x=path_mapped))


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
