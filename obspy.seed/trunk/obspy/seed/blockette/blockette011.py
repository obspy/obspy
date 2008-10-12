# -*- coding: utf-8 -*-

from obspy.seed.blockette import Blockette
from obspy.seed.fields import Integer, FixedString, MultipleLoop


class Blockette011(Blockette):
    """Blockette 011: Volume Station Header Index Blockette.
    
    This is the index to the Station Identifier Blockettes [50] that appear 
    later in the volume. This blockette refers to each station described in 
    the station header section.
    
    Sample:
    0110054004AAK  000003ANMO 000007ANTO 000010BJI  000012
    """
    
    id = 11
    name = "Volume Station Header Index Blockette"
    fields = [
        Integer(3, "Number of stations", 3),
        # REPEAT fields 4 — 5 for the Number of stations:
        MultipleLoop("Stations", "Number of stations", [
            FixedString(4, "Station identifier code", 5),
            Integer(5, "Sequence number of station header", 6)
        ])
    ]
    
    def verify(self, parser):
        pass