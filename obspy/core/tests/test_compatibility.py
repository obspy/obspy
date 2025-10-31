# -*- coding: utf-8 -*-

class TestCompatibility:
    def test_round(self):
        """
        Ensure round rounds correctly and returns expected data types.

        Maybe not needed after Py2 sunset? This usedd to be a test for
        compatibility function py3_round which was only an alias for round on
        Py3
        """
        # list of tuples; (input, ndigits, expected, excpected type)
        test_list = [
            (.222, 2, .22, float),
            (1516047903968282880, -3, 1516047903968283000, int),
            (1.499999999, None, 1, int),
            (.0222, None, 0, int),
            (12, -1, 10, int),
            (15, -1, 20, int),
            (15, -2, 0, int),
        ]
        for number, ndigits, expected, expected_type in test_list:
            out = round(number, ndigits)
            assert out == expected
            assert isinstance(out, expected_type)
