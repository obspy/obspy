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
    
    def verifyData(self, volume):
        # there should be only one blockette 011
        if len(volume.blockettes.get(11))!=1:
            msg = "INVALID: Volume Station Header Index Blockette [11] " + \
                  "should be defined only once." 
            print(msg)
        # check if all blockette 50 are indexed in blockette 11
        for b50 in volume.blockettes.get(50):
            if b50.record_id in self.sequence_number_of_station_header:
                continue
            msg = "INVALID: All Station Identifier Blockettes [50] " + \
                  "must be indexed in the Volume Station Header Index " + \
                  "Blockette [11]. Index for station %s missing in [11]." 
            print(msg % b50.station_call_letters)
        # check if all indexed stations have actually defined a blockette 50
        for record_id in self.sequence_number_of_station_header:
            record_ids = [b50.record_id for b50 in volume.blockettes.get(50)]
            if record_id in record_ids:
                continue
            msg = "INVALID: All Station indexed in the Volume " + \
                  "Station Header Index must be defined in a own " + \
                  "Station Identifier Blockette [50]. Blockette [50] at " + \
                  "sequence number %s is not indexed." 
            print(msg % record_id)