#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
readRESP.py
script to read RESP files into inventory.Response objects
'''
from __future__ import print_function
import re
from pprint import pprint

def read_RESP(filename):
    respfile = open(filename)
    blockettefieldlist = list()
    for line in respfile:
        print(line,end='')
        m = re.match(r"^B(\d+)F(\d+)(?:-(\d+))?(.*)", line)
        if m:
            g = m.groups()
            blockette_number = g[0]
            if not g[2]:
                #single field per line
                value = re.search(r":\s*(\S*)", g[3]).groups()[0]
                print( (blockette_number, g[1], value) )
                blockettefieldlist.append( (blockette_number, g[1], value))
                
            else:
                #multiple fields per line
                first_field = int(g[1])
                last_field = int(g[2])
                fields = g[3].split()
                values = fields[-(last_field - first_field + 1 ):] 
                for i, value in enumerate(values):
                    print( (blockette_number, first_field + i, value) )
                    blockettefieldlist.append( (blockette_number, first_field + i, value) )
                
            print()
    return blockettefieldlist
                
def make_xseed(blockettefieldlist):
    from obspy.io.xseed import (parser, blockette)
    seedparser = parser.Parser()
    seedparser.temp = {'volume': [], 'abbreviations': [], 'stations': []}
    record_type = 'S'
    seedparser.temp['stations'].append([])
    root_attribute = seedparser.temp['stations'][-1]
 
    old_blockette_id = None
    for blockette_id, field, value in blockettefieldlist:
        if old_blockette_id != blockette_id:
            class_name = 'Blockette%03d' % int(blockette_id)
            blockette_class = getattr(blockette, class_name)
            blockette_obj = blockette_class(debug=True,
                                            strict=False,
                                            compact=False,
                                            record_type=record_type)
            blockette_fields = blockette_obj.default_fields + blockette_obj.get_fields()
            old_blockette_id = blockette_id
        
            
                                            
    
    
    

if __name__ == '__main__':
    blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/PDCC/2015/NRL-download-directfromIRIS/IRIS/dataloggers/quanterra/RESP.XX.NQ004..BHZ.Q330.SR.1.40.all')
    #blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/PDCC/2015/NRL-download-directfromIRIS/IRIS/sensors/streckeisen/RESP.XX.NS085..BHZ.STS2_gen3.120.1500')    
    #blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/test_data/RESP.GR.FUR..BHE_with_blkt60')    
    #blockettefieldlist = read_RESP(filename='/Volumes/liquid/work/TESTDATA/XE.2015/rdseed.out.XE')    
    pprint(blockettefieldlist)
    make_xseed(blockettefieldlist)
    