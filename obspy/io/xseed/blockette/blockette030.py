# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Integer, Loop, VariableString


class Blockette030(Blockette):
    """
    Blockette 030: Data Format Dictionary Blockette.

    All volumes, with the exception of MiniSEED data records, must have a Data
    Format Dictionary Blockette [30]. Each Channel Identifier Blockette [52]
    has a reference (field 16) back to a Data Format Dictionary Blockette
    [30], so that SEED reading programs will know how to decode data for the
    channels. Because every kind of data format requires an entry in the Data
    Format Dictionary Blockette [30], each recording network needs to list
    entries for each data format, if a heterogeneous mix of data formats are
    included in a volume. This data format dictionary is used to decompress
    the data correctly.

    Sample:
    0300086CDSN Gain-Ranged Format~000200104M0~W2 D0-13 A-8191~D1415~
    P0:#0,1:#2,2:#4,3:#7~
    """

    id = 30
    name = "Data Format Dictionary"
    fields = [
        VariableString(3, "Short descriptive name", 1, 50, 'UNLPS'),
        Integer(4, "Data format identifier code", 4),
        Integer(5, "Data family type", 3),
        Integer(6, "Number of decoder keys", 2),
        # REPEAT field 7 for the Number of decoder keys:
        Loop("Decoder keys", "Number of decoder keys", [
            VariableString(7, "Decoder keys", flags='UNLPS')], omit_tag=True),
    ]
