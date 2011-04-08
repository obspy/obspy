#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: client.py
#  Purpose: Python client for the Data Handling Interface (DHI/Fissures)
#   Author: Moritz Beyreuther, Robert Barsch
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Moritz Beyreuther, Robert Barsch
#---------------------------------------------------------------------
"""
Data Handling Interface (DHI)/Fissures client.

Python function for accessing data from DHI/Fissures.
The method is based on omniORB CORBA requests.

:copyright: The ObsPy Development Team (devs@obspy.org)
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from omniORB import CORBA
from CosNaming import NameComponent, NamingContext
from idl import Fissures
from obspy.core import Trace, UTCDateTime, Stream, AttribDict
from obspy.core.util import deprecated_keywords
from obspy.mseed.libmseed import LibMSEED
from obspy.fissures.util import FissuresException, FissuresWarning, \
        poleZeroFilter2PAZ, utcdatetime2Fissures, use_first_and_raise_or_warn
import numpy as np
import sys
import warnings
from copy import deepcopy


DEPRECATED_KEYWORDS = {'network_id':'network', 'station_id':'station',
                       'location_id':'location', 'channel_id':'channel',
                       'start_datetime':'starttime', 'end_datetime':'endtime'}

MAG_TYPES = {'lg': "edu.iris.Fissures/MagType/LG",
             'mb': "edu.iris.Fissures/MagType/mb",
             'mbmle': "edu.iris.Fissures/MagType/mbmle",
             'ml': "edu.iris.Fissures/MagType/ML",
             'mo': "edu.iris.Fissures/MagType/MO",
             'ms': "edu.iris.Fissures/MagType/Ms",
             'msmle': "edu.iris.Fissures/MagType/msmle",
             'mw': "edu.iris.Fissures/MagType/MW"}
#MAG_TYPES = {'lg': "Fissures.LG_MAG_TYPE",
#             'mb': "Fissures.MB_MAG_TYPE",
#             'mbmle': "Fissures.MBMLE_MAG_TYPE",
#             'ml': "Fissures.ML_MAG_TYPE",
#             'mo': "Fissures.MO_MAG_TYPE",
#             'ms': "Fissures.MS_MAG_TYPE",
#             'msmle': "Fissures.MSMLE_MAG_TYPE",
#             'mw': "Fissures.MW_MAG_TYPE"}


class Client(object):
    """
    DHI/Fissures client class. For more informations see the
    :func:`~obspy.fissures.client.Client.__init__`
    method and all public methods of the client class.

    The Data Handling Interface (DHI) is a CORBA data access framework
    allowing users to access seismic data and metadata from IRIS DMC
    and other participating institutions directly from a DHI-supporting
    client program. The effect is to eliminate the extra steps of
    running separate query interfaces and downloading of data before
    visualization and processing can occur. The information is loaded
    directly into the application for immediate use.
    http://www.iris.edu/dhi/

    Detailed information on network_dc, seismogram_dc servers and CORBA:

    * http://www.seis.sc.edu/wily
    * http://www.iris.edu/dhi/servers.htm
    * http://www.seis.sc.edu/software/fissuresImpl/objectLocation.html

    Check availability of stations via SeismiQuery:

    * http://www.iris.edu/SeismiQuery/timeseries.htm

    .. note::
        Ports 6371 and 17508 must be open (IRIS Data and Name Services).
    """
    #
    # We recommend the port ranges 6371-6382, 17505-17508 to be open (this
    # is how it is configured in our institute).
    #
    def __init__(self, network_dc=("/edu/iris/dmc", "IRIS_NetworkDC"),
                 seismogram_dc=("/edu/iris/dmc", "IRIS_DataCenter"),
                 event_dc=("/edu/iris/dmc", "IRIS_EventDC"),
                 # XXX
                 # 2011-02-01: default iris dmc not working
                 # see http://www.iris.washington.edu/pipermail/dhi-servers/2011-January/001927.html
                 # replace with following line again if its working...
                 # XXX
                 #name_service="dmc.iris.washington.edu:6371/NameService",
                 name_service="dhiserv.iris.washington.edu:6371/NameService",
                 debug=False):
        """
        Initialize Fissures/DHI client. 
        
        :param network_dc: Tuple containing dns and NetworkDC name.
        :param seismogram_dc: Tuple containing dns and DataCenter name.
        :param event_dc: Tuple containing dns and EventDC name.
        :param name_service: String containing the name service.
        :param debug:  Enables verbose output of the connection handling
                (default is False).
        """
        # Some object wide variables
        if sys.byteorder == 'little':
            self.byteorder = True
        else:
            self.byteorder = False
        #
        self.mseed = LibMSEED()
        #
        # Initialize CORBA object, see pdf in trunk/obspy.fissures/doc or
        # http://omniorb.sourceforge.net/omnipy3/omniORBpy/omniORBpy004.html
        # for available options
        args = ["-ORBgiopMaxMsgSize", "2097152",
                "-ORBInitRef",
                "NameService=corbaloc:iiop:" + name_service]
        if debug:
            args = ["-ORBtraceLevel", "40"] + args
        orb = CORBA.ORB_init(args, CORBA.ORB_ID)
        self.obj = orb.resolve_initial_references("NameService")
        #
        # Resolve naming service
        try:
            self.rootContext = self.obj._narrow(NamingContext)
        except:
            msg = "Could not connect to " + name_service
            raise FissuresException(msg)
        #
        # network and seismogram cosnaming
        self.net_name = self._composeName(network_dc, 'NetworkDC')
        self.seis_name = self._composeName(seismogram_dc, 'DataCenter')
        self.ev_name = self._composeName(event_dc, 'EventDC')
        # resolve network finder
        try:
            netDC = self.rootContext.resolve(self.net_name)
            netDC = netDC._narrow(Fissures.IfNetwork.NetworkDC)
            netFind = netDC._get_a_finder()
            self.netFind = netFind._narrow(Fissures.IfNetwork.NetworkFinder)
        except:
            msg = "Initialization of NetworkFinder failed."
            warnings.warn(msg, FissuresWarning)
        # resolve event finder
        try:
            evDC = self.rootContext.resolve(self.ev_name)
            evDC = evDC._narrow(Fissures.IfEvent.EventDC)
            evFind = evDC._get_a_finder()
            self.evFind = evFind._narrow(Fissures.IfEvent.EventFinder)
        except:
            msg = "Initialization of EventFinder failed."
            warnings.warn(msg, FissuresWarning)
        # resolve seismogram DataCenter
        try:
            seisDC = self.rootContext.resolve(self.seis_name)
            self.seisDC = seisDC._narrow(Fissures.IfSeismogramDC.DataCenter)
        except:
            msg = "Initialization of seismogram DataCenter failed."
            warnings.warn(msg, FissuresWarning)
        # if both failed, client instance is useless, so raise
        if not self.netFind and not self.seisDC and not self.evFind:
            msg = "Neither NetworkFinder nor DataCenter nor EventFinder could be initialized."
            raise FissuresException(msg)

    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def getWaveform(self, network, station, location, channel, starttime,
                    endtime, getPAZ=False, getCoordinates=False):
        """
        Get Waveform in an ObsPy stream object from Fissures / DHI.

        >>> from obspy.core import UTCDateTime
        >>> from obspy.fissures import Client
        >>> client = Client()
        >>> t = UTCDateTime(2003,06,20,06,00,00)
        >>> st = client.getWaveform("GE", "APE", "", "SHZ", t, t+600)
        >>> print(st)
        1 Trace(s) in Stream:
        GE.APE..SHZ | 2003-06-20T06:00:00.001000Z - 2003-06-20T06:10:00.001000Z | 50.0 Hz, 30001 samples
        >>> st = client.getWaveform("GE", "APE", "", "SH*", t, t+600)
        >>> print(st)
        3 Trace(s) in Stream:
        GE.APE..SHZ | 2003-06-20T06:00:00.001000Z - 2003-06-20T06:10:00.001000Z | 50.0 Hz, 30001 samples
        GE.APE..SHN | 2003-06-20T06:00:00.001000Z - 2003-06-20T06:10:00.001000Z | 50.0 Hz, 30001 samples
        GE.APE..SHE | 2003-06-20T06:00:00.001000Z - 2003-06-20T06:10:00.001000Z | 50.0 Hz, 30001 samples

        :param network: Network id, 2 char; e.g. "GE"
        :param station: Station id, 5 char; e.g. "APE"
        :param location: Location id, 2 char; e.g. "  "
        :type channel: String, 3 char
        :param channel: Channel id, e.g. "SHZ". "*" as third letter is
                supported and requests "Z", "N", "E" components.
        :param starttime: UTCDateTime object of starttime
        :param endtime: UTCDateTime object of endtime
        :type getPAZ: Boolean
        :param getPAZ: Fetch PAZ information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.
        :type getCoordinates: Boolean
        :param getCoordinates: Fetch coordinate information and append to
            :class:`~obspy.core.trace.Stats` of all fetched traces. This
            considerably slows down the request.
        :return: Stream object
        """
        # NOTHING goes ABOVE this line!
        # append all args to kwargs, thus having everything in one dictionary
        # no **kwargs in method definition, so we need a get with default here
        kwargs = locals().get('kwargs', {})
        for key, value in locals().iteritems():
            if key not in ["self", "kwargs"]:
                kwargs[key] = value

        # intercept 3 letter channels with component wildcard
        # recursive call, quick&dirty and slow, but OK for the moment
        if len(channel) == 3 and channel[2] in ["*", "?"]:
            st = Stream()
            for cha in (channel[:2] + comp for comp in ["Z", "N", "E"]):
                # replace channel XXX a bit ugly:
                if 'channel' in kwargs:
                    kwargs.pop('channel')
                st += self.getWaveform(channel=cha, **kwargs)
            return st

        # get channel object
        channels = self._getChannelObj(network, station, location,
                channel)
        # get seismogram object
        seis = self._getSeisObj(channels, starttime, endtime)
        #
        # build up ObsPy stream object
        st = Stream()
        for sei in seis:
            # remove keep alive blockettes R
            if sei.num_points == 0:
                continue
            tr = Trace()
            tr.stats.starttime = UTCDateTime(sei.begin_time.date_time)
            tr.stats.npts = sei.num_points
            # calculate sampling rate
            unit = str(sei.sampling_info.interval.the_units.the_unit_base)
            if unit != 'SECOND':
                raise FissuresException("Wrong unit!")
            value = sei.sampling_info.interval.value
            power = sei.sampling_info.interval.the_units.power
            multi_factor = sei.sampling_info.interval.the_units.multi_factor
            exponent = sei.sampling_info.interval.the_units.exponent
            # sampling rate is given in Hertz within ObsPy!
            delta = pow(value * pow(10, power) * multi_factor, exponent)
            sr = sei.num_points / float(delta)
            tr.stats.sampling_rate = sr
            # set all kind of stats
            tr.stats.station = sei.channel_id.station_code
            tr.stats.network = sei.channel_id.network_id.network_code
            tr.stats.channel = sei.channel_id.channel_code
            tr.stats.location = sei.channel_id.site_code.strip()
            # loop over data chunks
            data = []
            for chunk in sei.data.encoded_values:
                # swap byte order in decompression routine if necessary 
                # src/IfTimeSeries.idl:52: FALSE = big endian format -
                swapflag = (self.byteorder != chunk.byte_order)
                compression = chunk.compression
                # src/IfTimeSeries.idl:44: const EncodingFormat STEIM2=11;
                if compression == 11:
                    data.append(self.mseed.unpack_steim2(chunk.values,
                                                         chunk.num_points,
                                                         swapflag=swapflag))
                # src/IfTimeSeries.idl:43: const EncodingFormat STEIM1=10;
                elif compression == 10:
                    data.append(self.mseed.unpack_steim1(chunk.values,
                                                         chunk.num_points,
                                                         swapflag=swapflag))
                else:
                    msg = "Compression %d not implemented" % compression
                    raise NotImplementedError(msg)
            # merge data chunks
            tr.data = np.concatenate(data)
            tr.verify()
            st.append(tr)
            # XXX: merging?
        st.trim(starttime, endtime)
        if getPAZ:
            for tr in st:
                cha = tr.stats.channel
                # XXX should add a check like metadata_check in seishub.client
                data = self.getPAZ(network=network, station=station,
                                   channel=cha, datetime=starttime)
                tr.stats['paz'] = deepcopy(data)
        if getCoordinates:
            # XXX should add a check like metadata_check in seishub.client
            data = self.getCoordinates(network=network,
                                       station=station,
                                       datetime=starttime)
            for tr in st:
                tr.stats['coordinates'] = deepcopy(data)
        return st


    def getNetworkIds(self):
        """
        Return all available networks as list.

        :note: This takes a very long time.
        """
        # Retrieve all available networks
        net_list = []
        networks = self.netFind.retrieve_all()
        for network in networks:
            network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
            attributes = network.get_attributes()
            net_list.append(attributes.id.network_code)
        return net_list


    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def getStationIds(self, network=None):
        """
        Return all available stations as list.

        If no network is specified this may take a long time

        :param network: Limit stations to network
        """
        # Retrieve network informations
        if network == None:
            networks = self.netFind.retrieve_all()
        else:
            networks = self.netFind.retrieve_by_code(network)
        station_list = []
        for network in networks:
            network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
            stations = network.retrieve_stations()
            for station in stations:
                station_list.append(station.id.station_code)
        return station_list

    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def getCoordinates(self, network, station, datetime):
        """
        Get Coordinates of a station.
        Still lacks a correct selection of metadata in time!

        >>> from obspy.fissures import Client
        >>> client = Client()
        >>> client.getCoordinates(network="GR", station="GRA1",
        ...                       datetime="2010-08-01")
        AttribDict({'latitude': 49.691886901855469, 'elevation': 499.5, 'longitude': 11.221719741821289})
        """
        sta = self._getStationObj(network=network, station=station,
                                  datetime=datetime)
        coords = AttribDict()
        loc = sta.my_location
        coords['elevation'] = loc.elevation.value
        unit = loc.elevation.the_units.name
        if unit != "METER":
            warnings.warn("Elevation not meter but %s." % unit)
        type = loc.type
        if str(type) != "GEOGRAPHIC":
            msg = "Location types != \"GEOGRAPHIC\" are not yet " + \
                  "implemented (type: \"%s\").\n" % type + \
                  "Please report the code that resulted in this error!"
            raise NotImplementedError(msg)
        coords['latitude'] = loc.latitude
        coords['longitude'] = loc.longitude
        return coords

    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def getPAZ(self, network, station, channel, datetime):
        """
        Get Poles&Zeros, gain and sensitivity of instrument for given ids and
        datetime.

        >>> from obspy.fissures import Client
        >>> client = Client()
        >>> client.getPAZ("GE", "APE", "BHZ", "2010-08-01") #doctest: +NORMALIZE_WHITESPACE
        AttribDict({'zeros': [0j, 0j], 'sensitivity': 588000000.0,
                    'poles': [(-0.037004001438617706+0.037016000598669052j),
                              (-0.037004001438617706-0.037016000598669052j),
                              (-251.33000183105469+0j),
                              (-131.03999328613281-467.29000854492188j),
                              (-131.03999328613281+467.29000854492188j)],
                    'gain': 60077000.0})
        
        Useful links:
        http://www.seis.sc.edu/software/simple/
        http://www.seis.sc.edu/downloads/simple/simple-1.0.tar.gz
        http://www.seis.sc.edu/viewvc/seis/branches/IDL2.0/fissuresUtil/src/edu/sc/seis/fissuresUtil2/sac/SacPoleZero.java?revision=16507&view=markup&sortby=log&sortdir=down&pathrev=16568
        http://www.seis.sc.edu/viewvc/seis/branches/IDL2.0/fissuresImpl/src/edu/iris/Fissures2/network/ResponseImpl.java?view=markup&sortby=date&sortdir=down&pathrev=16174

        :param network: Network id, 2 char; e.g. "GE"
        :param station: Station id, 5 char; e.g. "APE"
        :type channel: String, 3 char
        :param channel: Channel id, e.g. "SHZ", no wildcards.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime` or
                compatible String
        :param datetime: datetime of response information
        :return: :class:`~obspy.core.util.AttribDict`
        """
        if "*" in channel:
            msg = "Wildcards not allowed in channel"
            raise FissuresException(msg)
        net = self.netFind.retrieve_by_code(network)
        net = use_first_and_raise_or_warn(net, "network")
        datetime = UTCDateTime(datetime).formatFissures()
        sta = [sta for sta in net.retrieve_stations() \
               if sta.id.station_code == station \
               and datetime > sta.effective_time.start_time.date_time \
               and datetime < sta.effective_time.end_time.date_time]
        sta = use_first_and_raise_or_warn(sta, "station")
        cha = [cha for cha in net.retrieve_for_station(sta.id) \
               if cha.id.channel_code == channel]
        cha = use_first_and_raise_or_warn(cha, "channel")
        datetime = utcdatetime2Fissures(datetime)
        inst = net.retrieve_instrumentation(cha.id, datetime)
        resp = inst.the_response
        stage = use_first_and_raise_or_warn(resp.stages, "response stage")
        filters = [filter._v for filter in stage.filters \
                   if str(filter._d) == "POLEZERO"]
        filter = use_first_and_raise_or_warn(filters, "polezerofilter")
        paz = poleZeroFilter2PAZ(filter)
        norm = use_first_and_raise_or_warn(stage.the_normalization,
                                           "normalization")
        norm_fac = norm.ao_normalization_factor
        paz['gain'] = norm_fac
        paz['sensitivity'] = resp.the_sensitivity.sensitivity_factor
        return paz

    def getEvents(self, area_type, min_depth, max_depth, min_datetime,
                  max_datetime, min_magnitude, max_magnitude,
                  magnitude_types=[], catalogs=[], contributors=[],
                  max_results=500, **kwargs):
        """
        NOTE: THIS METHOD IS NOT WORKING AT THE MOMENT.
        
        :type area_type: String
        :param area_type: One of "global", "box" or "circle". Additional kwargs
                need to be specified for "box" ('min_latitude', 'max_latitude',
                'min_longitude', 'max_longitude') and for "circle" ('latitude',
                'longitude', 'min_distance', 'max_distance').
        :param min_depth: Minimum depth of events in kilometers
        :param max_depth: Maximum depth of events in kilometers
        :type min_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime` or
                UTCDateTime-compatible String
        :param min_datetime: Minimum origin time of events
        :type max_datetime: :class:`~obspy.core.utcdatetime.UTCDateTime` or
                UTCDateTime-compatible String
        :param max_datetime: Maximum origin time of events
        :param min_magnitude: Minimum magnitude of events
        :param max_magnitude: Minimum magnitude of events
        :type magnitude_types: List of Strings
        :param magnitude_types: Magnitude types to retrieve (defaults to all).
                Valid values are:
                "ml", "mb", "mo", "ms", "mw", "mbmle", "msmle", "lg"
        :type catalogs: List of Strings
        :param catalogs: Catalogs to retrieve events from
        :type contributors: List of Strings
        :param contributors: Contributors to retrieve events from
        """
        raise NotImplementedError()
        # construct Fissures area object
        if area_type == "global":
            area = Fissures.GlobalArea()
        elif area_type == "box":
            try:
                area = Fissures.BoxArea(min_latitude=kwargs['min_latitude'],
                                        max_latitude=kwargs['max_latitude'],
                                        min_longitude=kwargs['min_longitude'],
                                        max_longitude=kwargs['max_longitude'])
            except KeyError, e:
                raise FissuresException(str(e))
        elif area_type == "circle":
            try:
                area = Fissures.PointDistanceArea(latitude=kwargs['latitude'],
                        longitude=kwargs['longitude'],
                        min_distance=kwargs['min_distance'],
                        max_distance=kwargs['max_distance'])
            except KeyError, e:
                raise FissuresException(str(e))
        # construct depth range
        min_depth = Fissures.Quantity(min_depth * 1000, Fissures.METER)
        max_depth = Fissures.Quantity(max_depth * 1000, Fissures.METER)
        # construct time range
        min_datetime = utcdatetime2Fissures(UTCDateTime(min_datetime))
        max_datetime = utcdatetime2Fissures(UTCDateTime(max_datetime))
        time_range = Fissures.TimeRange(min_datetime, max_datetime)
        # map given magnitude types
        magnitude_types = [MAG_TYPES[mt.lower()] for mt in magnitude_types]
        # ensure floats for magnitudes
        #min_magnitude = float(min_magnitude)
        #max_magnitude = float(max_magnitude)
        # query EventFinder
        catalogs = self.evFind.known_catalogs()
        contributors = self.evFind.known_contributors()
        #import ipdb;ipdb.set_trace()
        # XXX strange: the idl definition seems to take 11 arguments, the last
        # XXX one being a EventSeqIterHolder but the python translation only
        # XXX takes 10 arguments and then raises an idl bad type error.
        # XXX Actually this seems to be ok, in Python the last argument ends
        # XXX up on the return side as an additional return variable.
        # XXX Unfortunately there seems to be no further information which
        # XXX argument has a wrong type, even when stepping through it in ipd.
        (events, event_iter) = self.evFind.query_events(area, min_depth, max_depth,
                time_range, magnitude_types, min_magnitude, max_magnitude,
                catalogs, contributors, max_results)
        return events, event_iter


    def _composeName(self, dc, interface):
        """
        Compose Fissures name in CosNaming.NameComponent manner. Set the
        dns, interfaces and objects together.
        
        >>> from obspy.fissures import Client
        >>> client = Client()
        >>> client._composeName(("/edu/iris/dmc", "IRIS_NetworkDC"),
        ...                     "NetworkDC") #doctest: +NORMALIZE_WHITESPACE
        [CosNaming.NameComponent(id='Fissures', kind='dns'),
         CosNaming.NameComponent(id='edu', kind='dns'),
         CosNaming.NameComponent(id='iris', kind='dns'),
         CosNaming.NameComponent(id='dmc', kind='dns'),
         CosNaming.NameComponent(id='NetworkDC', kind='interface'),
         CosNaming.NameComponent(id='IRIS_NetworkDC', kind='object_FVer1.0')]


        :param dc: Tuple containing dns and service as string
        :param interface: String describing kind of DC, one of EventDC,
            NetworkDC or DataCenter
        """
        # put network name together
        dns = [NameComponent(id='Fissures', kind='dns')]
        for id in dc[0].split('/'):
            if id != '':
                dns.append(NameComponent(id=id, kind='dns'))
        dns.extend([NameComponent(id=interface, kind='interface'),
                    NameComponent(id=dc[1], kind='object_FVer1.0')])
        return dns


    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def _getChannelObj(self, network, station, location, channel):
        """
        Return Fissures channel object.
        
        Fissures channel object is requested from the clients network_dc.
        
        :param network: Network id, 2 char; e.g. "GE"
        :param station: Station id, 5 char; e.g. "APE"
        :param location: Location id, 2 char; e.g. "  "
        :param channel: Channel id, 3 char; e.g. "SHZ"
        :return: Fissures channel object
        """
        # retrieve a network
        net = self.netFind.retrieve_by_code(network)
        net = use_first_and_raise_or_warn(net, "network")
        net = net._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
        # retrieve channels from network
        if location.strip() == "":
            # must be two empty spaces
            location = "  "
        # Retrieve Channel object
        # XXX: wildcards not yet implemented
        return net.retrieve_channels_by_code(station, location,
                                             channel)

    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def _getSeisObj(self, channel_obj, starttime, endtime):
        """
        Return Fissures seismogram object.
        
        Fissures seismogram object is requested from the clients
        network_dc. This actually contains the data.
        
        :param channel_obj: Fissures channel object
        :param starttime: UTCDateTime object of starttime
        :param endtime: UTCDateTime object of endtime
        :return: Fissures seismogram object
        """
        # Transform datetime into correct format
        t1 = utcdatetime2Fissures(starttime)
        t2 = utcdatetime2Fissures(endtime)
        # Form request for all channels
        request = [Fissures.IfSeismogramDC.RequestFilter(c.id, t1, t2) \
                   for c in channel_obj]
        # Retrieve Seismogram object
        return self.seisDC.retrieve_seismograms(request)

    @deprecated_keywords(DEPRECATED_KEYWORDS)
    def _getStationObj(self, network, station, datetime):
        """
        Return Fissures station object.
        
        Fissures station object is requested from the clients network_dc.
        
        :param network: Network id, 2 char; e.g. "GE"
        :param station: Station id, 5 char; e.g. "APE"
        :type datetime: String (understood by
                :class:`~obspy.core.datetime.DateTime`)
        :param datetime: Datetime to select station
        :return: Fissures channel object
        """
        net = self.netFind.retrieve_by_code(network)
        net = use_first_and_raise_or_warn(net, "network")
        # filter by station and by datetime (comparing datetime strings)
        datetime = UTCDateTime(datetime).formatFissures()
        stations = [sta for sta in net.retrieve_stations() \
                    if station == sta.id.station_code \
                    and datetime > sta.effective_time.start_time.date_time \
                    and datetime < sta.effective_time.end_time.date_time]
        return use_first_and_raise_or_warn(stations, "station")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
