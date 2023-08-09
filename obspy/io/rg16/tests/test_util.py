"""
Tests for rg16 utilities.
"""
import unittest
from io import BytesIO

from obspy.io.rg16.util import _read, _read_bcd, _read_binary


class TestRG16Util(unittest.TestCase):

    def byte_io(self, byte_str):
        """
        write byte_str to BytesIO object, return.
        """
        return BytesIO(byte_str)

    def test_read_bcd(self):
        """
        Ensure bcd encoding returns expected values.
        """
        bcd = [(b'\x99', 1, True, 99), (b'\x99\x01', 2, True, 9901),
               (b'\x91', 0.5, True, 9), (b'\x91', 0.5, False, 1),
               (b'\x99\01', 1.5, True, 990), (b'\x99\01', 1.5, False, 901)]
        for byte, length, left_part, answer in bcd:
            out = _read_bcd(BytesIO(byte), length, left_part)
            self.assertEqual(out, answer)
        with self.assertRaises(ValueError) as e:
            _read_bcd(BytesIO(b'\xFF'), 1, True)
        self.assertIn('invalid bcd values', str(e.exception))

    def test_read_binary(self):
        """
        Ensure binary encoding return expected values.
        """
        binary = [(b'\x99', 1, True, 153), (b'\x99\x01', 2, True, 39169),
                  (b'\x91', 0.5, True, 9), (b'\x91', 0.5, False, 1),
                  (b'\x76\x23\x14', 3, True, 7742228),
                  (b'\x00\x10\x00\x10', 4, True, 1048592),
                  (b'\x10\xed\x7f\x01\x00\x00\x08\xc0', 8,
                  True, 1219770716358969536)]
        for byte, length, left_part, answer in binary:
            out = _read_binary(BytesIO(byte), length, left_part)
            self.assertEqual(out, answer)

    def test_read(self):
        """
        Ensure IEEE float are well returned in the function _read.
        """
        ieee = b'\x40\x48\xf5\xc3'
        out = _read(BytesIO(ieee), 0, 4, 'IEEE')
        self.assertAlmostEqual(out, 3.14, delta=1e-6)
