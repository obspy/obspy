# -*- coding: utf-8 -*-
"""
The obspy.segy core test suite.
"""

from __future__ import with_statement
import numpy as np
from obspy.core.util import NamedTemporaryFile
from obspy.segy.core import isSEGY, readSEGY, writeSEGY, SEGYCoreWritingError
from obspy.segy.segy import SEGYError
from obspy.segy.segy import readSEGY as readSEGYInternal
from obspy.segy.tests.header import FILES, DTYPES
import os
import unittest


class SEGYCoreTestCase(unittest.TestCase):
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
        self.files = FILES
        self.dtypes = DTYPES

    def test_isSEGYFile(self):
        """
        Tests the isSEGY method.
        """
        # Test all files in the test directory.
        for file in self.files.keys():
            file = os.path.join(self.path, file)
            self.assertEqual(isSEGY(file), True)
        # Also check all the other files in the test directory and they should
        # not work. Just check certain files to ensure reproducibility.
        files = ['test_core.py', 'test_segy.py', '__init__.py']
        for file in files:
            file = os.path.join(self.path, file)
            self.assertEqual(isSEGY(file), False)

    def test_enforcingTextualHeaderEncodingWhileReading(self):
        """
        Tests whether or not the enforcing of the encoding of the textual file
        header actually works.
        """
        # File ld0042_file_00018.sgy_first_trace has an EBCDIC encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        # Read once with EBCDIC encoding and check if it is correct.
        st1 = readSEGY(file, textual_header_encoding='EBCDIC')
        self.assertTrue(st1[0].stats.segy.textual_file_header[3:21] \
                        == 'CLIENT: LITHOPROBE')
        # This should also be written the stats dictionary.
        self.assertEqual(st1[0].stats.segy.textual_file_header_encoding,
                         'EBCDIC')
        # Reading again with ASCII should yield bad results. Lowercase keyword
        # argument should also work.
        st2 = readSEGY(file, textual_header_encoding='ascii')
        self.assertFalse(st2[0].stats.segy.textual_file_header[3:21] \
                        == 'CLIENT: LITHOPROBE')
        self.assertEqual(st2[0].stats.segy.textual_file_header_encoding,
                         'ASCII')
        # Autodection should also write the textual file header encoding to the
        # stats dictionary.
        st3 = readSEGY(file)
        self.assertEqual(st3[0].stats.segy.textual_file_header_encoding,
                         'EBCDIC')

    def test_enforcingEndiannessWhileWriting(self):
        """
        Tests whether or not the enforcing of the endianness while writing
        works.
        """
        # File ld0042_file_00018.sgy_first_trace is in big endian.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st1 = readSEGY(file)
        # First write should be big endian.
        out_file = NamedTemporaryFile().name
        writeSEGY(st1, out_file)
        st2 = readSEGY(out_file)
        os.remove(out_file)
        self.assertEqual(st2[0].stats.segy.endian, '>')
        # Do once again to enforce big endian.
        writeSEGY(st1, out_file, byteorder='>')
        st3 = readSEGY(out_file)
        os.remove(out_file)
        self.assertEqual(st3[0].stats.segy.endian, '>')
        # Enforce little endian.
        writeSEGY(st1, out_file, byteorder='<')
        st4 = readSEGY(out_file)
        os.remove(out_file)
        self.assertEqual(st4[0].stats.segy.endian, '<')

    def test_settingDataEncodingWorks(self):
        """
        Test whether or not the enforcing the data encoding works.
        """
        # File ld0042_file_00018.sgy_first_trace uses IBM floating point
        # representation.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = readSEGY(file)
        # First test if it even works.
        out_file = NamedTemporaryFile().name
        writeSEGY(st, out_file)
        with open(out_file, 'rb') as f:
            data1 = f.read()
        os.remove(out_file)
        # Write again and enforce encoding one which should yield the same
        # result.
        writeSEGY(st, out_file, data_encoding=1)
        with open(out_file, 'rb') as f:
            data2 = f.read()
        os.remove(out_file)
        self.assertTrue(data1 == data2)
        # Writing IEEE floats which should not require any dtype changes.
        writeSEGY(st, out_file, data_encoding=5)
        with open(out_file, 'rb') as f:
            data3 = f.read()
        os.remove(out_file)
        self.assertFalse(data1 == data3)

    def test_readingAndWritingDifferentDataEncodings(self):
        """
        Writes and reads different data encodings and checks if the data
        remains the same.
        """
        # The file uses IBM data encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = readSEGY(file)
        data = st[0].data
        # All working encodings with corresponding dtypes.
        encodings = {1: 'float32',
                     2: 'int32',
                     3: 'int16',
                     5: 'float32'}
        out_file = NamedTemporaryFile().name
        # Loop over all encodings.
        for data_encoding, dtype in encodings.iteritems():
            this_data = np.require(data.copy(), dtype)
            st[0].data = this_data
            writeSEGY(st, out_file, data_encoding=data_encoding)
            # Read again and compare data.
            this_stream = readSEGY(out_file)
            os.remove(out_file)
            # Both should now be equal. Usually converting from IBM to IEEE
            # floating point numbers might result in small rouning errors but
            # in this case it seems to work. Might be different on different
            # computers.
            np.testing.assert_array_equal(this_data, this_stream[0].data)

    def test_notMatchingDataEncodingAndDtypeRaises(self):
        """
        obspy.segy does not automatically convert to the corresponding dtype.
        """
        encodings = [1, 2, 3, 5]
        # The file uses IBM data encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = readSEGY(file)
        # Use float64 as the wrong encoding in every case.
        st[0].data = np.require(st[0].data, 'float64')
        out_file = NamedTemporaryFile().name
        # Loop over all encodings.
        for data_encoding in encodings:
            self.assertRaises(SEGYCoreWritingError, writeSEGY, st, out_file,
                              data_encoding=data_encoding)
        os.remove(out_file)

    def test_invalidDataEncodingRaises(self):
        """
        Using an invalid data encoding raises an error.
        """
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st = readSEGY(file)
        out_file = NamedTemporaryFile().name
        self.assertRaises(SEGYCoreWritingError, writeSEGY, st, out_file,
                          data_encoding=0)
        self.assertRaises(SEGYCoreWritingError, writeSEGY, st, out_file,
                          data_encoding='')
        os.remove(out_file)

    def test_enforcingTextualHeaderEncodingWhileWriting(self):
        """
        Tests whether or not the enforcing of the endianness while writing
        works.
        """
        # File ld0042_file_00018.sgy_first_trace has an EBCDIC encoding.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        st1 = readSEGY(file)
        # Save the header to compare it later on.
        with open(file, 'rb') as f:
            header = f.read(3200)
        # First write should remain EBCDIC.
        out_file = NamedTemporaryFile().name
        writeSEGY(st1, out_file)
        st2 = readSEGY(out_file)
        # Compare header.
        with open(out_file, 'rb') as f:
            new_header = f.read(3200)
        self.assertTrue(header == new_header)
        os.remove(out_file)
        self.assertEqual(st2[0].stats.segy.textual_file_header_encoding,
                         'EBCDIC')
        # Do once again to enforce EBCDIC.
        writeSEGY(st1, out_file, textual_header_encoding='EBCDIC')
        st3 = readSEGY(out_file)
        # Compare header.
        with open(out_file, 'rb') as f:
            new_header = f.read(3200)
        self.assertTrue(header == new_header)
        os.remove(out_file)
        self.assertEqual(st3[0].stats.segy.textual_file_header_encoding,
                         'EBCDIC')
        # Enforce ASCII
        writeSEGY(st1, out_file, textual_header_encoding='ASCII')
        st4 = readSEGY(out_file)
        # Compare header. Should not be equal this time.
        with open(out_file, 'rb') as f:
            new_header = f.read(3200)
        self.assertFalse(header == new_header)
        os.remove(out_file)
        self.assertEqual(st4[0].stats.segy.textual_file_header_encoding,
                         'ASCII')

    def test_enforcingEndiannessWhileReading(self):
        """
        Tests whether or not enforcing the endianness while reading a file
        works. It will actually just deactivate the autodetection in case it
        produced a wrong result. Using a wrong endianness while reading a file
        will still produce an error because the data format will most likely be
        wrong and therefore obspy.segy cannot unpack the data.
        """
        # File ld0042_file_00018.sgy_first_trace is in big endian.
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        # This should work and write big endian to the stats dictionary.
        st1 = readSEGY(file)
        self.assertEqual(st1[0].stats.segy.endian, '>')
        # Doing the same with the right endianness should still work.
        st2 = readSEGY(file, byteorder='>')
        self.assertEqual(st2[0].stats.segy.endian, '>')
        # The wrong endianness should yield an key error because the routine to
        # unpack the wrong data format code cannot be found.
        self.assertRaises(KeyError, readSEGY, file, byteorder='<')

    def test_readingUsingCore(self):
        """
        This tests checks whether or not all necessary information is read
        during reading with core. It actually just assumes the internal SEGYFile
        object, which is thoroughly tested in obspy.segy.tests.test_segy, is
        correct and compared all values to it. This seems to be the easiest way
        to test everything.
        """
        for file, _ in self.files.iteritems():
            file = os.path.join(self.path, file)
            # Read the file with the internal SEGY representation.
            segy_file = readSEGYInternal(file)
            # Read again using core.
            st = readSEGY(file)
            # They all should have length one because all additional traces
            # have been removed.
            self.assertEqual(len(st), 1)
            # Assert the data is the same.
            np.testing.assert_array_equal(segy_file.traces[0].data, st[0].data)
            # Textual header.
            self.assertEqual(segy_file.textual_file_header,
                             st[0].stats.segy.textual_file_header)
            # Textual_header_encoding.
            self.assertEqual(segy_file.textual_header_encoding,
                             st[0].stats.segy.textual_file_header_encoding)
            # Endianness.
            self.assertEqual(segy_file.endian, st[0].stats.segy.endian)
            # Data encoding.
            self.assertEqual(segy_file.data_encoding,
                             st[0].stats.segy.data_encoding)
            # Test the file and trace binary headers.
            for key, value in \
                    segy_file.binary_file_header.__dict__.iteritems():
                self.assertEqual(getattr(st[0].stats.segy.binary_file_header,
                                 key), value)
            for key, value in \
                    segy_file.traces[0].header.__dict__.iteritems():
                self.assertEqual(getattr(st[0].stats.segy.trace_header, key),
                                 value)

    def test_writingUsingCore(self):
        """
        Tests the writing of SEGY rev1 files using obspy.core. It just compares
        the output of writing using obspy.core with the output of writing the
        files using the internal SEGY object which is thoroughly tested in
        obspy.segy.tests.test_segy.
        """
        for file, _ in self.files.iteritems():
            file = os.path.join(self.path, file)
            # Read the file with the internal SEGY representation.
            segy_file = readSEGYInternal(file)
            # Read again using core.
            st = readSEGY(file)
            # Create two temporary files to write to.
            out_file1 = NamedTemporaryFile().name
            out_file2 = NamedTemporaryFile().name
            # Write twice.
            segy_file.write(out_file1)
            writeSEGY(st, out_file2)
            # Read and delete files.
            with open(out_file1, 'rb') as f1:
                data1 = f1.read()
            with open(out_file2, 'rb') as f2:
                data2 = f2.read()
            os.remove(out_file1)
            os.remove(out_file2)
            # Test if they are equal.
            self.assertEqual(data1[3200:3600], data2[3200:3600])

    def test_invalidValuesForTextualHeaderEncoding(self):
        """
        Invalid keyword arguments should be caught gracefully.
        """
        file = os.path.join(self.path, 'ld0042_file_00018.sgy_first_trace')
        self.assertRaises(SEGYError, readSEGY, file,
                          textual_header_encoding='BLUB')


def suite():
    return unittest.makeSuite(SEGYCoreTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
