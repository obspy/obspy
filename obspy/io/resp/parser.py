#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
readRESP.py
script to read RESP files into inventory.Response objects
Lloyd Carothers
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import re
import io
from pprint import pprint

DEBUG = True

def read_RESP(filename):
    respfile = open(filename)
    # List of blockettes which is a list of fields
    blockettelist = list()
    # List of fields
    blockettefieldlist = list()
    last_blockette_id = None
    for line in respfile:
        print(line, end='')
        m = re.match(r"^B(\d+)F(\d+)(?:-(\d+))?(.*)", line)
        if m:
            g = m.groups()
            blockette_number = g[0]
            if blockette_number != last_blockette_id:
                # A new blockette starting
                if len(blockettefieldlist) > 0:
                    print("new blockette")
                    blockettelist.append(blockettefieldlist)
                    blockettefieldlist = list()
                last_blockette_id = blockette_number
            if not g[2]:
                #single field per line
                value = re.search(r":\s*(\S*)", g[3]).groups()[0]
                #print( (blockette_number, g[1], value) )
                blockettefieldlist.append( (blockette_number, g[1], value))
                
            else:
                #multiple fields per line
                first_field = int(g[1])
                last_field = int(g[2])
                fields = g[3].split()
                values = fields[-(last_field - first_field + 1 ):] 
                for i, value in enumerate(values):
                    #print( (blockette_number, first_field + i, value) )
                    blockettefieldlist.append( (blockette_number, first_field + i, value) )
        elif re.match(r"^#.*\+", line):
            # Comment line with a + in it means blockette is finished start a new one
            if len(blockettefieldlist) > 0:
                print("new blockette")
                blockettelist.append(blockettefieldlist)
                blockettefieldlist = list()
                
            #print()
    return blockettelist
                
def make_xseed(RESPblockettelist):
    from obspy.io.xseed import (parser, blockette, fields)
    seedparser = parser.Parser()
    seedparser.temp = {'volume': [], 'abbreviations': [], 'stations': []}
    # Make an empty blockette10
    seedparser.temp['volume'].append(  blockette.Blockette010(debug=DEBUG,
                                 strict=False,
                                 compact=False,
                                 record_type='V') )
                                 
    # Make unit lookup blockette34
    b34s = ('034  44  4M/S~velocity in meters per second~',
            '034  25  5V~emf in volts~',
            '034  32  7COUNTS~digital counts~')
    abbv_lookup = {'M/S': '4',
                  'V': '5',
                  'COUNTS': '7'}
    for b34 in b34s:
        data = io.BytesIO(b34.encode('utf-8'))
        b34_obj = blockette.Blockette034(debug=DEBUG,
                                         record_type='A')
        b34_obj.parse_SEED(data, expected_length=len(b34))
        seedparser.temp['abbreviations'].append(b34_obj)
    record_type = 'S'
    seedparser.temp['stations'].append([])
    root_attribute = seedparser.temp['stations'][-1]
 
    old_RESPblockette_id = None
    
    for RESPblockettefieldlist in RESPblockettelist:
        # Creat a new blockette using the first field
        RESPblockette_id, RESPfield, RESPvalue = RESPblockettefieldlist[0]
        class_name = 'Blockette%03d' % int(RESPblockette_id)
        blockette_class = getattr(blockette, class_name)
        blockette_obj = blockette_class(debug=DEBUG,
                                        strict=False,
                                        compact=False,
                                        record_type=record_type)
        blockette_fields = blockette_obj.default_fields + blockette_obj.get_fields()
        unrolled_blockette_fields = list()
        for bf in blockette_fields:
            if isinstance(bf, fields.Loop):
                for df in bf.data_fields:
                    unrolled_blockette_fields.append(df)
            else:
                unrolled_blockette_fields.append(bf)
        blockette_fields = unrolled_blockette_fields
        # List of fields with fields used removed so unused can be set to default after
        unused_fields = blockette_fields[:]
        
        for RESPblockette_id, RESPfield, RESPvalue in RESPblockettefieldlist:
            for bfield in blockette_fields:
                # bfield maybe a loop and have no id
                if bfield.id == int(RESPfield):
                    if isinstance(bfield, fields.VariableString):
                        # Variable string needs terminator '~'
                        RESPvalue += '~'
                    print(RESPvalue)
                    #lookup if abbv
                    RESPvalue = abbv_lookup.get(RESPvalue, RESPvalue)
                    print(RESPvalue)
                    dataRESPvalue = io.BytesIO(RESPvalue.encode('utf-8'))
                    if blockette_obj.id == 57 and bfield.id == 4:
                        # Oddity with B057F03
                        bfield.length = 12
                    bfield.parse_SEED(blockette_obj, dataRESPvalue)
                    if bfield in unused_fields:
                        # Looping fields can't be removed twice.
                        unused_fields.remove(bfield)
                    break
        for bfield in unused_fields:
            # Set unused fields to default
            bfield.parse_SEED(blockette_obj, None)
        
        root_attribute.append(blockette_obj)
        seedparser.blockettes.setdefault(blockette_obj.id,
                                         []).append(blockette_obj)
        
    seedparser._update_internal_SEED_structure()
    seedparser.get_SEED()
    pprint(seedparser.__dict__)        
    
if __name__ == '__main__':
    blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/PDCC/2015/NRL-download-directfromIRIS/IRIS/dataloggers/quanterra/RESP.XX.NQ004..BHZ.Q330.SR.1.40.all')
    #blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/PDCC/2015/NRL-download-directfromIRIS/IRIS/sensors/streckeisen/RESP.XX.NS085..BHZ.STS2_gen3.120.1500')    
    #blockettefieldlist = read_RESP(filename='/Users/lloyd/work/workMOONBASE/test_data/RESP.GR.FUR..BHE_with_blkt60')    
    #blockettefieldlist = read_RESP(filename='/Volumes/liquid/work/TESTDATA/XE.2015/rdseed.out.XE')    
    pprint(blockettefieldlist)
    make_xseed(blockettefieldlist)
    