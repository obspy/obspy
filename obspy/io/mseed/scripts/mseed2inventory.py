#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scans for mseed files and builds and inventory object from mseed headers
This contains code that duplicates functionallity in scan.py and mseed.read.

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from collections import defaultdict, namedtuple
from calendar import timegm
from itertools import groupby
from operator import attrgetter
from os import walk
from os.path import join, isfile, getsize
from struct import Struct
from sys import argv
from time import strftime, strptime, gmtime
from obspy.core import inventory
from obspy.core.inventory.util import Site
from obspy.core.utcdatetime import UTCDateTime


class SeisTime:
    '''A Class for handling generic time operations common to seismic
    data.
    '''
    FMT = '%Y:%j:%H:%M:%S'
    STRFMT = "%0.4d:%0.3d:%0.2d:%0.2d:%0.2d.%0.4d"

    @staticmethod
    def seis2struct_time(year, jday, hour=0, minute=0, second=0):
        timestring = ':'.join(map(str, (year, jday, hour, minute, second)))
        return strptime(timestring, SeisTime.FMT)

    @staticmethod
    def struct_time2epoch(structtime, microsecond=0):
        return timegm(structtime) + microsecond * 0.0001

    @staticmethod
    def seis2epoch(*args, **kwargs):
        ''' takes args: year, jday, hour=0, minute=0, second=0, microsecond=0
        returns an epochal time
        '''
        return SeisTime.struct_time2epoch(
            SeisTime.seis2struct_time(*args), **kwargs)

    @staticmethod
    def seis2string(*args, **kwargs):
        ''' takes args: year, jday, hour=0, minute=0, second=0, microsecond=0
        returns a printable usable string YYYY:DDD:HH:MM:SS.MMMM
        '''
        return SeisTime.STRFMT % (args + (kwargs['microsecond'],))

    @staticmethod
    def epoch2string(epoch):
        subsec = ('%0.4f' % (epoch % 1.0))[1:]
        timestring = strftime(SeisTime.FMT,  gmtime(epoch)) + subsec
        return timestring


class MSeed:
    '''A miniSEED file, usually, but truely a collection of data records
    '''
    # Fixed Header
    FixedHeaderFMT = '6scc5s2s3s2sHHBBBBHHhhBBBBlHH'
    FixedHeader = namedtuple('FixedHeader',
                             '''Sequence_number
                             Data_quality
                             Reserved
                             Station
                             Location
                             Channel
                             Network
                             Year
                             Day
                             Hour
                             Minute
                             Second
                             Unused
                             Microsecond
                             Number_of_samples
                             Sample_rate_factor
                             Sample_rate_multiplier
                             Activity_flags
                             IO_and_clock_flags
                             Data_quality_flags
                             Number_of_blockettes
                             Time_correction
                             Beginning_data
                             First_blockette''')
    # Blockette
    BlocketteCOMFMT = 'HH'
    BlocketteCOM = namedtuple('BlocketteCOM', 'Type Next_blockette')
    # Blockette1000
    Blockette1000FMT = 'HHbBBB'
    Blockette1000 = namedtuple('Blockette1000',
                               ''' Type
                               Next_blockette
                               Encoding_Format
                               Word_Order
                               Data_Record_Length
                               Reserved''')
    # Blockette100
    Blockette100FMT = 'HHfb3B'
    Blockette100 = namedtuple('Blockette100',
                              ''' Type
                               Next_blockette
                               Actual_sample_rate
                               Flags
                               Reserved1
                               Reserved2
                               Reserved3 ''')

    SNCL = namedtuple('SNCL', ('network, station, location, channel,'
                               'sr, encoding, endian'))
    # Time stored in epochal
    TimeSpan = namedtuple('TimeSpan', 'start, end')

    def __init__(self, filename):
        self.filename = filename
        self.filesize = getsize(filename)
        self.bigendian = None
        self.manifest = Manifest()
        self.multiplexed = None
        self.block_size = None
        # Times should be epoch seconds since 1970
        self.starttime = None
        self.endtime = None
        try:
            self.filehandle = open(filename, 'rb')
        except Exception, e:
            print "Could not open file %s" % self.filename
            print e
            return
        # Read the first datarecord to get the basic info
        # Average/largeish block size to start)
        chunk1 = self.filehandle.read(4096)
        datarecord = self.decodeDataRecord(chunk1)
        if not self.IsMseed:
            return
        self.fixedheader, self.blockettes = datarecord
        # Set the start time
        self.starttime = self.getEpoch(self.fixedheader)
        self.snclFirst = self.getSNCL(self.fixedheader, self.blockettes)

        # Is file multiple of blocksize
        if self.filesize % self.block_size != 0:
            print "%s NOT EVEN block size!!!!!!!!!!!!!" % self.filename
            return
        # Read last datarecord
        self.filehandle.seek(-self.block_size, 2)
        chunkLast = self.filehandle.read(self.block_size)
        if len(chunkLast) != self.block_size:
            print("%s Reading last block read smaller than block"
                  % self.filename)
            return
        datarecord = self.decodeDataRecord(chunkLast)
        if not datarecord:
            print("%s Couldn't read datarecord. Does blocksize change in file?"
                  % self.filename)
            return
        fixedheader, blockettes = datarecord
        # Set endtime
        self.endtime = self.getLastSampleTime(fixedheader)
        snclLast = self.getSNCL(fixedheader, blockettes)
        # Do fist and last block match? if not then multiplexed
        if self.snclFirst != snclLast:
            print "%s Multiplexed!!!!!!!!!!!!!!!!!" % self.filename
            self.multiplexed = True
            return
        self.ReadAllDataRecords()
        self.manifest = Manifest()
        self.manifest[self.snclFirst].append(
            self.TimeSpan(self.starttime, self.endtime))

    def __str__(self):
        return '%s %s %s %s ' % (self.filename,
                                 self.snclFirst,
                                 self.fixedheader,
                                 self.blockettes)

    @property
    def IsMseed(self):
        if self.bigendian is not None:
            return True
        return False

    def getSNCL(self, fxhd, blockettes):
        encoding, word_order = (
            (blk.Encoding_Format, blk.Word_Order) for blk in blockettes
            if blk.Type == 1000).next()
        return self.SNCL(fxhd.Network, fxhd.Station, fxhd.Location,
                         fxhd.Channel, self.getSampleRate(fxhd),
                         encoding, word_order)

    def getEpoch(self, fxhd):
        return SeisTime.seis2epoch(fxhd.Year, fxhd.Day, fxhd.Hour,
                                   fxhd.Minute, fxhd.Second,
                                   microsecond=fxhd.Microsecond)

    def getLastSampleTime(self, fxhd):
        '''returns epochal time of last sample in block
        '''
        rate = self.getSampleRate(fxhd)
        blockstart = self.getEpoch(fxhd)
        # LOG channels
        if rate == 0:
            return blockstart
        return blockstart + (fxhd.Number_of_samples - 1) / rate

    def getSampleRate(self, fxhd):
        sampFact = float(fxhd.Sample_rate_factor)
        sampMult = float(fxhd.Sample_rate_multiplier)
        if sampFact > 0 and sampMult > 0:
            rate = sampFact * sampMult
        elif sampFact > 0 and sampMult < 0:
            rate = -1.0 * (sampFact/sampMult)
        elif sampFact < 0 and sampMult > 0:
            rate = -1.0 * (sampMult/sampFact)
        elif sampFact < 0 and sampMult < 0:
            rate = 1.0/(sampFact * sampMult)
        else:
            rate = sampFact
        return rate

    def setEndian(self):
        '''once we know endian we can setup faster reads for future
        this assumes the entire file is the same endian and block size
        '''
        self.decodeFixedHeader = self.decodeFixedHeaderEndianEstablished
        self.decodeDataRecord = self.decodeDataRecordEndianEstablished
        if self.bigendian is True:
            endkey = '>'
        else:
            endkey = '<'
        self.blocketteCOMStruct = Struct(endkey + self.BlocketteCOMFMT)
        self.blockette1000Struct = Struct(endkey + self.Blockette1000FMT)
        self.blockette100Struct = Struct(endkey + self.Blockette100FMT)

    def ReadAllDataRecords(self):
        self.filehandle.seek(0)
        chunk = self.filehandle.read(self.block_size)
        while len(chunk) == self.block_size:
            # dr = self.decodeDataRecord(chunk)
            chunk = self.filehandle.read(self.block_size)

    def decodeDataRecord(self, chunk):
        '''Decodes a datarecord or block
        returns ( fixedheader, [ blockette,...] )
        first time run this then EndianEstablished version
        '''
        fixedheader = self.decodeFixedHeader(chunk)
        if not fixedheader:
            return None
        self.setEndian()
        blk_start = fixedheader.First_blockette
        blockettes = []
        while blk_start:
            blockette = self.decodeBlockette(chunk[blk_start:])
            blk_start = blockette.Next_blockette
            blockettes.append(blockette)
        return (fixedheader, blockettes)

    def decodeDataRecordEndianEstablished(self, chunk):
        '''After Endian established use this function to decode dr
        '''
        fixedheader = self.decodeFixedHeader(chunk)
        blk_start = fixedheader.First_blockette
        blockettes = []
        while blk_start:
            blockette = self.decodeBlockette(chunk[blk_start:])
            blk_start = blockette.Next_blockette
            blockettes.append(blockette)
        return (fixedheader, blockettes)

    def decodeFixedHeader(self, chunk):
        '''fist time run this then: determine endian, set self.compiled
        struct, selfendian, and rebind this method to decode...big or little
        '''
        if len(chunk) < 48:
            return False
        # Try big
        self.fixedheaderStruct = Struct('>' + self.FixedHeaderFMT)
        fixheadtry = self.decodeFixedHeaderEndianEstablished(chunk)
        if self.isValidFixedhdr(fixheadtry):
            self.bigendian = True
            return fixheadtry
        # Try Little
        self.fixedheaderStruct = Struct('<' + self.FixedHeaderFMT)
        fixheadtry = self.decodeFixedHeaderEndianEstablished(chunk)
        if self.isValidFixedhdr(fixheadtry):
            self.bigendian = False
            return fixheadtry
        else:
            # Not MSEED clean up
            return None

    def decodeFixedHeaderEndianEstablished(self, chunk):
        return self.FixedHeader._make(
            self.fixedheaderStruct.unpack(chunk[:48]))

    def isValidFixedhdr(self, fixedheader):
        if (0 <= fixedheader.Hour < 24 and
                fixedheader.Data_quality in ['D', 'R', 'Q', 'M'] and
                fixedheader.Sequence_number.isdigit() and
                1950 <= fixedheader.Year <= 2050):
            return True
        return False

    def decodeBlockette(self, chunk):
        baseblockette = self.BlocketteCOM._make(
            self.blocketteCOMStruct.unpack(chunk[:4]))
        if baseblockette.Type == 1000:
            blk1000 = self.Blockette1000._make(
                self.blockette1000Struct.unpack(chunk[:8]))
            self.block_size = 2**blk1000.Data_Record_Length
            return blk1000
        if baseblockette.Type == 100:
            return self.Blockette100._make(
                self.blockette100Struct.unpack(chunk[:12]))
        else:
            return baseblockette


class Manifest(defaultdict):
    '''Mainifest is a container for SNCLs with start and end times that are
    mergeable and appendable.
    A sublclass of defaultdict  {SNCL : list of timespantuples}.
    '''
    @staticmethod
    def SNCL2string(sncl):
        s = ('{s.network}:{s.station}:{s.location}:{s.channel}:'
             '{s.sr:>6}:{s.encoding}:{s.endian}')
        return s.format(s=sncl)

    @staticmethod
    def TimeSpan2string(span):
        return "%s -- %s" % (SeisTime.epoch2string(span.start),
                             SeisTime.epoch2string(span.end))

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(list, *args)
        # is a sort needed
        self.inorder = True

    def __setitem__(*args):
        ''' Overide this to set the inorder mark to false
        '''
        self = args[0]
        self.inorder = False
        super(self.__class__, self).__setitem__(*args[1:])

    def __str__(self):
        self.sort()
        s = ''
        for sncl, spans in ((key, self[key]) for key in sorted(self.keys())):
            s += self.SNCL2string(sncl) + '\n'
            for span in spans:
                s += '\t' + self.TimeSpan2string(span) + '\n'
            s += '\n'
        return s

    def merge(self, other):
        for sncl, spans in other.items():
            self[sncl].extend(spans)
        self.inorder = False

    def reduce21span(self):
        '''
        Reduces to a summary of 1 timespan per sncl: first and last sample
        '''
        self.sort()
        for sncl, spans in self.iteritems():
            self[sncl] = [MSeed.TimeSpan(spans[0].start, spans[-1].end)]
        self.inorder = True

    def getSummary(self):
        pass

    def sort(self):
        '''sorts the lists of time spans
        '''
        if self.inorder is True:
            return
        for spans in self.values():
            spans.sort()
        self.inorder = True


def GetManifestOfDIR(dir):
    '''
    Driver that searches depths of current dir looking for mseed files
    '''
    try:
        manifest = Manifest()
        # global numseen
        numseen = 0
        # global numseed
        numseed = 0
        for root, dirs, files in walk(dir):
            for basename in files:
                fullname = join(root, basename)
                print fullname
                numseen += 1
                if not isfile(fullname):
                    continue
                mseed = MSeed(fullname)
                if mseed.IsMseed:
                    numseed += 1
                    manifest.merge(mseed.manifest)
                else:
                    del mseed
    except KeyboardInterrupt:
        print "Caught Ctrl-C..."
    manifest.reduce21span()
    print manifest
    print "Saw: ", numseen
    print "MSEED:", numseed
    return manifest


def make_inventory(manifest):
    # Standard orientations
    AZ = 'az'
    DIP = 'dip'
    ORIENT = defaultdict(lambda: {AZ: 0.0, DIP: 0.0},
                         {'Z': {AZ: 0.0, DIP: -90.0},
                          'N': {AZ: 0.0, DIP: 0.0},
                          'E': {AZ: 90.0, DIP: 0.0}
                          })
    # Dummy values for now
    lat = inventory.util.Latitude(0.0)
    lon = inventory.util.Longitude(0.0)
    depth = 0.0
    elev = 0
    net_list = list()
    for network, net_group in groupby(sorted(manifest.keys()),
                                      key=attrgetter('network')):
        print network
        net_earliest = UTCDateTime()
        net_latest = UTCDateTime(0.0)
        station_list = list()
        for station, sta_group in groupby(net_group,
                                          key=attrgetter('station')):
            print station
            sta_earliest = UTCDateTime()
            sta_latest = UTCDateTime(0.0)
            chan_list = list()
            for location, loc_group in groupby(sta_group,
                                               key=attrgetter('location')):
                print location
                for sncl in loc_group:
                    print sncl
                    if sncl.channel in ['LOG', 'VM1', 'VM2', 'VM3']:
                        print 'skipping...'
                        continue
                    start = UTCDateTime(manifest[sncl][0].start)
                    sta_earliest = min(sta_earliest, start)
                    end = UTCDateTime(manifest[sncl][0].end)
                    sta_latest = max(sta_latest, end)
                    chan_list.append(inventory.Channel(
                        code=sncl.channel,
                        location_code=location,
                        latitude=lat,
                        longitude=lon,
                        elevation=elev,
                        depth=depth,
                        azimuth=ORIENT[sncl.channel[2]][AZ],
                        dip=ORIENT[sncl.channel[2]][DIP],
                        sample_rate=sncl.sr,
                        start_date=start,
                        end_date=end))
            station_list.append(inventory.Station(code=station,
                                                  latitude=lat,
                                                  longitude=lon,
                                                  elevation=elev,
                                                  channels=chan_list,
                                                  site=Site(
                                                      name='Site Name'),
                                                  creation_date=sta_earliest,
                                                  start_date=sta_earliest,
                                                  end_date=sta_latest))
            net_earliest = min(net_earliest, sta_earliest)
            net_latest = max(net_latest, sta_latest)
        net_list.append(inventory.Network(code=network,
                                          stations=station_list,
                                          start_date=net_earliest,
                                          end_date=net_latest))
    inv = inventory.Inventory(net_list, source='PIC')
    return inv


def inventory_from_mseed(directory, sensRESP, dlRESP):
    manifest = GetManifestOfDIR(directory)
    inv = make_inventory(manifest)
    # Attach response to inventory
    inv_resp = inventory.response.response_from_resp(sensRESP, dlRESP)
    inv_of_wf = inv.select(channel='CH*')
    for net in inv_of_wf.networks:
        for sta in net.stations:
            for chan in sta.channels:
                chan.response = inv_resp
    return inv

if __name__ == '__main__':
    sensRESP = ('/Users/lloyd/work/workMOONBASE/PDCC/2015/FULL PDCC '
                'tool/PDCC-3.8/NRL/edu.iris/sensors/guralp'
                '/RESP.XX.NS007..BHZ.CMG3T.120.1500')
    dlRESP = ('/Users/lloyd/work/workMOONBASE/PDCC/2015/FULL PDCC '
              'tool/PDCC-3.8/NRL/edu.iris/dataloggers/reftek'
              '/RESP.XX.NR012..HHZ.130.1.500')
    inv = inventory_from_mseed(argv[1], sensRESP, dlRESP)
    print(inv)
    inv.write('XE.station.xml', format='STATIONXML')
