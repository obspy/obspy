# -*- coding: utf-8 -*-
"""
Created on Wed 2015-03-04 00:00 UTC

@author: Mike Turnbull

Kelunji Classic bindings to ObsPy core module.
"""

from obspy import Trace, UTCDateTime, Stream  # Not used yet
import numpy as np                            # Not used yet
import struct


def IsKelunjiClassic(filename):
    """
    Checks whether a file is a readable Kelunji Classic Seismogram (KCS).

    :type filename: string
    :param filename: Potential KC file to be checked.
    :rtype: integer
    :return: True if it is a readable KCS file.
    """

    testStatus = True  # Assume a successful outcome.

    # These are the conanical descriptors embedded in a valid Kelunji
    # Classic seismogram file - as specified by the manufacturer.
    headerVersion = 4
    recordType = 'S'          # Indicates that the file is a Seismogram
    descKA1 = '4E3(12N)'      # Indicates data is in KA1 real seismogram format
    descKA1C = '4E3(12N)16C'  # Indicates data is in KA1 calibration format
    descKA2 = '4E3(12N)'      # Indicates data is in KA2 real seismogram format
    descKA2C = '4E3(12N)16C'  # Indicates data is in KA2 calibration format

    try:
        fileIn = open(filename, "rb")

        # Interpret the first 2 bytes as an unsigned char and a character.
        bytes = fileIn.read(2)
        data = struct.unpack('Bc', bytes)

        # Compare the data with the conanical values.
        if data[0] <> headerVersion or data[1] <> recordType:
            # Fails header and version ID test.
            testStatus = False

        # Interpret the 8 bytes starting at file position 0x22 as characters.
        fileIn.seek(0x22)
        bytes = fileIn.read(20)
        data = struct.unpack('20c', bytes)
        strData = ("".join(data)).strip()
        # Compare the data with the conanical values.
        if not strData.startswith((descKA1, descKA1C)):
            # Fails KA1 format test.
            if not strData.startswith((descKA2, descKA2C)):
                # Fails KA2 format test.
                testStatus = False

        '''=========================================================
        TODO: Need to find out what the canonical KA2 descriptes are.
        =========================================================='''

        fileIn.close()
    except:
        # File input failed for some (as yet) unknown reason.
        testStatus = False

    return testStatus


def readKelunjiClassic(filename, **kwargs):  # @UnusedVariable
    """
    Reads a Kelunji Classic seismogram file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: Kelunji Classic seismogram file to be read.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """

    # TODO:


def test(filename):
    """
    Test the code with good and bad data files.

    :type filename: string
    :param filename: File to be tested.
    """
    result = IsKelunjiClassic(filename)

    if result:
        print '"' + filename + '" is a valid Kelunji Classic seismogram.'
    else:
        print '"' + filename + '" is NOT a valid Kelunji Classic seismogram.'

    return result

"""
Now test that code.
"""
goodFile = './test/2015-02-26 020706- FS03'
badFile = './test/FS03 0518, 2015-02-26 bulk'

print
goodTest = test(goodFile)
print
badTest = test(badFile)
print
if goodTest and not badTest:
    print 'Test is successful.'
else:
    print 'Test has FAILED.'
