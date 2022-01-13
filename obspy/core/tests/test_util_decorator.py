# -*- coding: utf-8 -*-
from obspy.core.util import get_example_file
from obspy.core.util.decorator import map_example_filename


class TestUtilDecorator:
    def test_map_example_filename(self):
        """
        Tests the @map_example_filename decorator
        """
        dummy = "abc"
        example_file = "example.npz"
        path = "/path/to/" + example_file
        path_mapped = get_example_file(example_file)

        def unchanged(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))

        @map_example_filename("a")
        def changed1(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        assert changed1(dummy, dummy) == unchanged(dummy, dummy)
        assert changed1(path, dummy) == unchanged(path_mapped, dummy)
        assert changed1(dummy, path) == unchanged(dummy, path)
        assert changed1(a=path, b=dummy) == unchanged(path_mapped, dummy)
        assert changed1(path, b=dummy) == unchanged(path_mapped, dummy)
        assert changed1(path, b=path, x=path) == \
            unchanged(path_mapped, path, x=path)

        @map_example_filename("b")
        def changed2(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        assert changed2(dummy, dummy) == unchanged(dummy, dummy)
        assert changed2(path, dummy) == unchanged(path, dummy)
        assert changed2(dummy, path) == unchanged(dummy, path_mapped)
        assert changed2(a=path, b=dummy) == unchanged(path, dummy)
        assert changed2(path, b=path) == unchanged(path, path_mapped)
        assert changed2(path, b=path, x=path) == \
            unchanged(path, path_mapped, x=path)

        @map_example_filename("x")
        def changed3(a, b="", **kwargs):
            return list(map(str, (a, b, kwargs)))
        assert changed3(dummy, dummy) == unchanged(dummy, dummy)
        assert changed3(path, dummy) == unchanged(path, dummy)
        assert changed3(dummy, path) == unchanged(dummy, path)
        assert changed3(a=path, b=dummy) == unchanged(path, dummy)
        assert changed3(path, b=dummy) == unchanged(path, dummy)
        assert changed3(path, b=path, x=path) == \
            unchanged(path, path, x=path_mapped)
