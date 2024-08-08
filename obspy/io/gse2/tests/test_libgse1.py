#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The libgse1 test suite.
"""
from obspy.io.gse2 import libgse1
from obspy.io.gse2.libgse2 import ChksumError
import pytest


class TestLibGSE1():
    """
    Test cases for libgse1.
    """
    def test_verify_checksums(self, testdata):
        """
        Tests verifying checksums for CM6 encoded GSE1 files.
        """
        # 1
        fh = open(testdata['acc.gse'], 'rb')
        libgse1.read(fh, verify_chksum=True)
        fh.close()
        # 2
        fh = open(testdata['y2000.gse'], 'rb')
        libgse1.read(fh, verify_chksum=True)
        fh.close()
        # 3
        fh = open(testdata['loc_STAU20031119011659.z'], 'rb')
        libgse1.read(fh, verify_chksum=True)
        fh.close()
        # 4 - second checksum is wrong
        fh = open(testdata['GRF_031102_0225.GSE.wrong_chksum'], 'rb')
        libgse1.read(fh, verify_chksum=True)  # correct
        with pytest.raises(ChksumError):
            libgse1.read(fh, verify_chksum=True)
        fh.close()
