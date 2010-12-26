# -*- coding: utf-8 -*-
"""
The obspy.segy test suite.
"""

from __future__ import with_statement
from StringIO import StringIO
from obspy.segy import read
from obspy.segy.header import DATA_SAMPLE_FORMAT_PACK_FUNCTIONS, \
    DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS
from obspy.segy.segy import SEGYBinaryFileHeader, SEGYTraceHeader, SEGYFile
from obspy.segy.util import NamedTemporaryFile
import numpy as np
import os
import unittest


class SEGYTestCase(unittest.TestCase):
    """
    Test cases for SEG Y reading and writing..
    """
    def setUp(self):
        # directory where the test files are located
        self.dir = os.path.dirname(__file__)
        self.path = os.path.join(self.dir, 'data')
        # All the files and information about them. These files will be used in
        # most tests. data_sample_enc is the encoding of the data value and
        # sample_size the size in bytes of these samples.
        self.files = {'00001034.sgy_first_trace': {'endian': '<',
                           'data_sample_enc': 1, 'textual_header_enc': 'ASCII',
                           'sample_count': 2001, 'sample_size': 4,
                           'non_normalized_samples': [21, 52, 74, 89, 123,
                            126, 128, 132, 136, 155, 213, 221, 222, 223,
                            236, 244, 258, 266, 274, 281, 285, 286, 297, 298,
                            299, 300, 301, 302, 318, 335, 340, 343, 346, 353,
                            362, 382, 387, 391, 393, 396, 399, 432, 434, 465,
                            466, 470, 473, 474, 481, 491, 494, 495, 507, 513,
                            541, 542, 555, 556, 577, 615, 616, 622, 644, 652,
                            657, 668, 699, 710, 711, 717, 728, 729, 738, 750,
                            754, 768, 770, 771, 774, 775, 776, 780, 806, 830,
                            853, 857, 869, 878, 885, 890, 891, 892, 917, 962,
                            986, 997, 998, 1018, 1023, 1038, 1059, 1068, 1073,
                            1086, 1110, 1125, 1140, 1142, 1150, 1152, 1156,
                            1157, 1165, 1168, 1169, 1170, 1176, 1182, 1183,
                            1191, 1192, 1208, 1221, 1243, 1250, 1309, 1318,
                            1328, 1360, 1410, 1412, 1414, 1416, 1439, 1440,
                            1453, 1477, 1482, 1483, 1484, 1511, 1518, 1526,
                            1530, 1544, 1553, 1571, 1577, 1596, 1616, 1639,
                            1681, 1687, 1698, 1701, 1718, 1734, 1739, 1745,
                            1758, 1786, 1796, 1807, 1810, 1825, 1858, 1864,
                            1868, 1900, 1904, 1912, 1919, 1928, 1941, 1942,
                            1943, 1953, 1988]},
                      '1.sgy_first_trace': {'endian': '>',
                           'data_sample_enc': 2, 'textual_header_enc': 'ASCII',
                           'sample_count': 8000, 'sample_size': 4,
                           'non_normalized_samples': []},
                      'example.y_first_trace': {'endian': '>',
                            'data_sample_enc': 3,
                            'textual_header_enc': 'EBCDIC',
                            'sample_count': 500, 'sample_size': 2,
                            'non_normalized_samples': []},
                      'ld0042_file_00018.sgy_first_trace': {'endian': '>',
                           'data_sample_enc': 1,
                           'textual_header_enc': 'EBCDIC',
                           'sample_count': 2050, 'sample_size': 4,
                           'non_normalized_samples': []},
                      'planes.segy_first_trace': {'endian': '<',
                           'data_sample_enc': 1,
                           'textual_header_enc': 'EBCDIC',
                           'sample_count': 512, 'sample_size': 4,
                           'non_normalized_samples': []}}
        # The expected NumPy dtypes for the various sample encodings.
        self.dtypes = {1: 'float32',
                       2: 'int32',
                       3: 'int16',
                       5: 'float32'}

    def test_unpackSEGYData(self):
        """
        Tests the unpacking of various SEG Y files.
        """
        for file, attribs in self.files.iteritems():
            data_format = attribs['data_sample_enc']
            endian = attribs['endian']
            count = attribs['sample_count']
            file = os.path.join(self.path, file)
            # Use the with statement to make sure the file closes.
            with open(file, 'rb') as f:
                # Jump to the beginning of the data.
                f.seek(3840)
                # Unpack the data.
                data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[data_format](f,
                                count, endian)
            # Check the dtype of the data.
            self.assertEqual(data.dtype, self.dtypes[data_format])
            # Proven data values, read with Madagascar.
            correct_data = np.load(file + '.npy').ravel()
            # Compare both.
            np.testing.assert_array_equal(correct_data, data)

    def test_packSEGYData(self):
        """
        Tests the packing of various SEG Y files.
        """
        # Loop over all files.
        for file, attribs in self.files.iteritems():
            # Get some attributes.
            data_format = attribs['data_sample_enc']
            endian = attribs['endian']
            count = attribs['sample_count']
            size = attribs['sample_size']
            non_normalized_samples = attribs['non_normalized_samples']
            dtype = self.dtypes[data_format]
            file = os.path.join(self.path, file)
            # Load the data. This data has previously been unpacked by
            # Madagascar.
            data = np.load(file + '.npy').ravel()
            data = np.require(data, dtype)
            # Load the packed data.
            with open(file, 'rb') as f:
                # Jump to the beginning of the data.
                f.seek(3200 + 400 + 240)
                packed_data = f.read(count * size)
            # The pack functions all write to file objects.
            f = StringIO()
            # Pack the data.
            DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[data_format](f, data, endian)
            # Read again.0.
            f.seek(0, 0)
            new_packed_data = f.read()
            # Check the length.
            self.assertEqual(len(packed_data), len(new_packed_data))
            if len(non_normalized_samples) == 0:
                # The packed data should be totally identical.
                self.assertEqual(packed_data, new_packed_data)
            else:
                # Some test files contain non normalized IBM floating point
                # data. These cannot be reproduced exactly.
                # Just a sanity check to be sure it is only IBM floating point
                # data that does not work completely.
                self.assertEqual(data_format, 1)

                # Read the data as uint8 to be able to directly access the
                # different bytes.
                # Original data.
                packed_data = np.fromstring(packed_data, 'uint8')
                # Newly written.
                new_packed_data = np.fromstring(new_packed_data, 'uint8')

                # Figure out the non normalized fractions in the original data
                # because these cannot be compared directly.
                # Get the position of the first byte of the fraction depending
                # on the endianness.
                if endian == '>':
                    start = 1
                else:
                    start = 2
                # The first byte of the fraction.
                first_fraction_byte_old = packed_data[start::4]
                # First get all zeros in the original data because zeros have
                # to be treated differently.
                zeros = np.where(data == 0)[0]
                # Create a copy and set the zeros to a high number to be able
                # to find all non normalized numbers.
                fraction_copy = first_fraction_byte_old.copy()
                fraction_copy[zeros] = 255
                # Normalized numbers will have no zeros in the first 4 bit of
                # the fraction. This means that the most significant byte of
                # the fraction has to be at least 16 for it to be normalized.
                non_normalized = np.where(fraction_copy < 16)[0]

                # Sanity check if the file data and the calculated data are the
                # same.
                np.testing.assert_array_equal(non_normalized,
                                              np.array(non_normalized_samples))

                # Test all other parts of the packed data. Set dtype to int32
                # to get 4 byte numbers.
                packed_data_copy = packed_data.copy()
                new_packed_data_copy = new_packed_data.copy()
                packed_data_copy.dtype = 'int32'
                new_packed_data_copy.dtype = 'int32'
                # Equalize the non normalized parts.
                packed_data_copy[non_normalized] = \
                        new_packed_data_copy[non_normalized]
                np.testing.assert_array_equal(packed_data_copy,
                                              new_packed_data_copy)

                # Now check the non normalized parts if they are almost the
                # same.
                data = data[non_normalized]
                # Unpack the data again.
                new_packed_data.dtype = 'int32'
                new_packed_data = new_packed_data[non_normalized]
                length = len(new_packed_data)
                f = StringIO()
                f.write(new_packed_data.tostring())
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](f,
                                        length, endian)
                f.close()
                packed_data.dtype = 'int32'
                packed_data = packed_data[non_normalized]
                length = len(packed_data)
                f = StringIO()
                f.write(packed_data.tostring())
                f.seek(0, 0)
                old_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](f,
                                        length, endian)
                f.close()
                # This works because the normalized and the non normalized IBM
                # floating point numbers will be close enough for the internal
                # IEEE representation to be identical.
                np.testing.assert_array_equal(data, new_data)
                np.testing.assert_array_equal(data, old_data)

    def test_packAndUnpackIBMFloat(self):
        """
        Packing and unpacking IBM floating points might yield some inaccuracies
        due to floating point rounding errors.
        This test tests a large number of random floating point numbers.
        """
        # Some random seeds.
        seeds = [1234, 592, 459482, 6901, 0, 7083, 68349]
        endians = ['<', '>']
        # Loop over all combinations.
        for seed in seeds:
            # Generate 50000 random floats from -10000 to +10000.
            np.random.seed(seed)
            data = 200000.0 * np.random.ranf(50000) - 100000.0
            # Convert to float64 in case native floats are different to be
            # able to utilize double precision.
            data = np.require(data, 'float64')
            # Loop over little and big endian.
            for endian in endians:
                # Pack.
                f = StringIO()
                DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
                # Jump to beginning and read again.
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](f,
                                        len(data), endian)
                f.close()
                # A relative tolerance of 1E-6 is considered good enough.
                rms1 = rms(data, new_data)
                self.assertEqual(True, rms1 < 1E-6)

    def test_packAndUnpackVerySmallIBMFloats(self):
        """
        The same test as test_packAndUnpackIBMFloat just for small numbers
        because they might suffer more from the inaccuracies.
        """
        # Some random seeds.
        seeds = [123, 1592, 4482, 601, 1, 783, 6849]
        endians = ['<', '>']
        # Loop over all combinations.
        for seed in seeds:
            # Generate 50000 random floats from -10000 to +10000.
            np.random.seed(seed)
            data = 1E-5 * np.random.ranf(50000)
            # Convert to float64 in case native floats are different to be
            # able to utilize double precision.
            data = np.require(data, 'float64')
            # Loop over little and big endian.
            for endian in endians:
                # Pack.
                f = StringIO()
                DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
                # Jump to beginning and read again.
                f.seek(0, 0)
                new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](f,
                                        len(data), endian)
                f.close()
                # A relative tolerance of 1E-6 is considered good enough.
                rms1 = rms(data, new_data)
                self.assertEqual(True, rms1 < 1E-6)

    def test_packAndUnpackIBMSpecialCases(self):
        """
        Tests the packing and unpacking of several powers of 16 which are
        problematic because they need separate handling in the algorithm.
        """
        endians = ['>', '<']
        # Create the first 10 powers of 16.
        data = []
        for i in xrange(10):
            data.append(16 ** i)
            data.append(-16 ** i)
        data = np.array(data)
        # Convert to float64 in case native floats are different to be
        # able to utilize double precision.
        data = np.require(data, 'float64')
        # Loop over little and big endian.
        for endian in endians:
            # Pack.
            f = StringIO()
            DATA_SAMPLE_FORMAT_PACK_FUNCTIONS[1](f, data, endian)
            # Jump to beginning and read again.
            f.seek(0, 0)
            new_data = DATA_SAMPLE_FORMAT_UNPACK_FUNCTIONS[1](f,
                                    len(data), endian)
            f.close()
            # Test both.
            np.testing.assert_array_equal(new_data, data)

    def test_readAndWriteBinaryFileHeader(self):
        """
        Reading and writing should not change the binary file header.
        """
        for file, attribs in self.files.iteritems():
            endian = attribs['endian']
            file = os.path.join(self.path, file)
            # Read the file.
            with open(file, 'rb') as f:
                f.seek(3200)
                org_header = f.read(400)
            header = SEGYBinaryFileHeader(header=org_header, endian=endian)
            # The header writes to a file like object.
            new_header = StringIO()
            header.write(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            self.assertEqual(len(new_header), 400)
            # Assert the actual header.
            self.assertEqual(org_header, new_header)

    def test_readAndWriteTextualFileHeader(self):
        """
        Reading and writing should not change the textual file header.
        """
        for file, attribs in self.files.iteritems():
            endian = attribs['endian']
            header_enc = attribs['textual_header_enc']
            file = os.path.join(self.path, file)
            # Read the file.
            f = open(file, 'rb')
            org_header = f.read(3200)
            f.seek(0, 0)
            # Initialize an empty SEGY object and set certain attributes.
            segy = SEGYFile()
            segy.endian = endian
            segy.file = f
            segy.textual_header_encoding = None
            # Read the textual header.
            segy._readTextualHeader()
            # Assert the encoding and compare with known values.
            self.assertEqual(segy.textual_header_encoding, header_enc)
            # Close the file.
            f.close()
            # The header writes to a file like object.
            new_header = StringIO()
            segy._writeTextualHeader(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            self.assertEqual(len(new_header), 3200)
            # Assert the actual header.
            self.assertEqual(org_header, new_header)

    def test_readAndWriteTraceHeader(self):
        """
        Reading and writing should not change the trace header.
        """
        for file, attribs in self.files.iteritems():
            endian = attribs['endian']
            file = os.path.join(self.path, file)
            # Read the file.
            with open(file, 'rb') as f:
                f.seek(3600)
                org_header = f.read(240)
            header = SEGYTraceHeader(header=org_header, endian=endian)
            # The header writes to a file like object.
            new_header = StringIO()
            header.write(new_header)
            new_header.seek(0, 0)
            new_header = new_header.read()
            # Assert the correct length.
            self.assertEqual(len(new_header), 240)
            # Assert the actual header.
            self.assertEqual(org_header, new_header)

    def test_readAndWriteSEGY(self):
        """
        Reading and writing again should not change a file.
        """
        for file, attribs in self.files.iteritems():
            file = os.path.join(self.path, file)
            non_normalized_samples = attribs['non_normalized_samples']
            # Read the file.
            with open(file, 'rb') as f:
                org_data = f.read()
            segy_file = read(file)
            out_file = NamedTemporaryFile().name
            segy_file.write(out_file)
            # Read the new file again.
            with open(out_file, 'rb') as f:
                new_data = f.read()
            os.remove(out_file)
            # The two files should have the same length.
            self.assertEqual(len(org_data), len(new_data))
            # Replace the not normalized samples. The not normalized
            # samples are already tested in test_packSEGYData and therefore not
            # tested again here.
            if len(non_normalized_samples) != 0:
                # Convert to 4 byte integers. Any 4 byte numbers work.
                org_data = np.fromstring(org_data, 'int32')
                new_data = np.fromstring(new_data, 'int32')
                # Skip the header (4*960 bytes) and replace the non normalized
                # data samples.
                org_data[960:][non_normalized_samples] = \
                        new_data[960:][non_normalized_samples]
                # Create strings again.
                org_data = org_data.tostring()
                new_data = new_data.tostring()
            # Test the identity.
            self.assertEqual(org_data, new_data)


def rms(x, y):
    """
    Normalized RMS

    Taken from the mtspec library:
    http://svn.geophysik.uni-muenchen.de/trac/mtspecpy
    """
    return np.sqrt(((x - y) ** 2).mean() / (x ** 2).mean())


def suite():
    return unittest.makeSuite(SEGYTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
