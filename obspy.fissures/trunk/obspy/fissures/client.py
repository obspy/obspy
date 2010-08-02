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
from obspy.core import Trace, UTCDateTime, Stream
from obspy.mseed.libmseed import LibMSEED
import numpy as np
import sys


class Client(object):
    """
    DHI/Fissures client class. For more informations see the __init__
    method and all public methods of the client class.

    The Data Handling Interface (DHI) is a CORBA data access framework
    allowing users to access seismic data and metadata from IRIS DMC
    and other participating institutions directly from a DHI-supporting
    client program. The effect is to eliminate the extra steps of
    running separate query interfaces and downloading of data before
    visualization and processing can occur. The information is loaded
    directly into the application for immediate use.
    http://www.iris.edu/dhi/

    Detailed information on network_dc and seismogram_dc servers:

    * http://www.seis.sc.edu/wily
    * http://www.iris.edu/dhi/servers.htm

    .. note::
        Ports 6371 and 17508 must be open (IRIS Data and Name Services).
    """
    # We recommend the port ranges (6371-6382,17505-17508) to be open.
    def __init__(self, network_dc=("/edu/iris/dmc", "IRIS_NetworkDC"),
                 seismogram_dc=("/edu/iris/dmc", "IRIS_DataCenter"),
                 name_service="dmc.iris.washington.edu:6371/NameService",
                 debug=False):
        """
        Initialize Fissures/DHI client. 
        
        :param network_dc: Tuple containing dns and NetworkDC name.
        :param seismogram_dc: Tuple containing dns and DataCenter name.
        :param name_service: String containing the name service.
        :param debug:  Enables verbose output of the connection handling
                (default is False).
        """
        #
        # Some object wide variables
        if sys.byteorder == 'little':
            self.byteorder = True
        else:
            self.byteorder = False
        #
        self.mseed = LibMSEED()
        #
        # Initialize CORBA object
        args = ["-ORBgiopMaxMsgSize", "2097152",
                "-ORBInitRef",
                "NameService=corbaloc:iiop:" + name_service,
        ]
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
            raise Exception(msg)
        #
        # network cosnaming
        self.net_name = self._composeName(network_dc, 'NetworkDC')
        #
        # seismogram cosnaming
        self.seis_name = self._composeName(seismogram_dc, 'DataCenter')

    def getWaveform(self, network_id, station_id, location_id, channel_id,
            start_datetime, end_datetime):
        """
        Get Waveform in an ObsPy stream object from Fissures / DHI.

        >>> client = Client()
        >>> t = UTCDateTime(2003,06,20,06,00,00)
        >>> st = client.getWaveform("GE", "APE", "", "SHZ", t, t+600)

        :param network_id: Network id, 2 char; e.g. "GE"
        :param station_id: Station id, 5 char; e.g. "APE"
        :param location_id: Location id, 2 char; e.g. "  "
        :param channel_id: Channel id, 3 char; e.g. "SHZ"
        :param start_datetime: UTCDateTime object of starttime
        :param end_datetime: UTCDateTime object of endtime
        :return: Stream object
        """
        # get channel object
        channels = self._getChannelObj(network_id, station_id, location_id,
                channel_id)
        # get seismogram object
        seis = self._getSeisObj(channels, start_datetime, end_datetime)
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
                raise Exception("Wrong unit!")
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
        return st

    def getNetworkIds(self):
        """
        Return all available network_ids as list.

        :note: This takes a very long time.
        """
        netDC = self.rootContext.resolve(self.net_name)
        netDC = netDC._narrow(Fissures.IfNetwork.NetworkDC)
        netFind = netDC._get_a_finder()
        netFind = netFind._narrow(Fissures.IfNetwork.NetworkFinder)
        # Retrieve all available networks
        net_list = []
        networks = netFind.retrieve_all()
        for network in networks:
            network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
            attributes = network.get_attributes()
            net_list.append(attributes.id.network_code)
        return net_list

    def getStationIds(self, network_id=None):
        """
        Return all available stations as list.

        If no network_id is specified this may take a long time

        :param network_id: Limit stations to network_id
        """
        netDC = self.rootContext.resolve(self.net_name)
        netDC = netDC._narrow(Fissures.IfNetwork.NetworkDC)
        netFind = netDC._get_a_finder()
        netFind = netFind._narrow(Fissures.IfNetwork.NetworkFinder)
        # Retrieve network informations
        if network_id == None:
            networks = netFind.retrieve_all()
        else:
            networks = netFind.retrieve_by_code(network_id)
        station_list = []
        for network in networks:
            network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
            stations = network.retrieve_stations()
            for station in stations:
                station_list.append(station.id.station_code)
        return station_list

    def _composeName(self, dc, interface):
        """
        Compose Fissures name in CosNaming.NameComponent manner. Set the
        dns, interfaces and objects together.

        >>> self._composeName(("/edu/iris/dmc", "IRIS_NetworkDC"),
                              "NetworkDC")

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

    def _dateTime2Fissures(self, utc_datetime):
        """
        Convert datetime instance to fissures time object
        
        :param utc_datetime: UTCDateTime instance
        :return: Fissures time object
        """
        t = str(utc_datetime)[:-3] + 'Z'
        return Fissures.Time(t, -1)

    def _getChannelObj(self, network_id, station_id, location_id, channel_id):
        """
        Return Fissures channel object.
        
        Fissures channel object is requested from the clients network_dc.
        
        :param network_id: Network id, 2 char; e.g. "GE"
        :param station_id: Station id, 5 char; e.g. "APE"
        :param location_id: Location id, 2 char; e.g. "  "
        :param channel_id: Channel id, 3 char; e.g. "SHZ"
        :return: Fissures channel object
        """
        # resolve network finder
        netDC = self.rootContext.resolve(self.net_name)
        netDC = netDC._narrow(Fissures.IfNetwork.NetworkDC)
        netFind = netDC._get_a_finder()
        netFind = netFind._narrow(Fissures.IfNetwork.NetworkFinder)
        # retrieve a network
        network = netFind.retrieve_by_code(network_id)[0]
        network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
        # retrieve channels from network
        if location_id.strip() == "":
            # must be two empty spaces
            location_id = "  "
        # Retrieve Channel object
        # XXX: wildcards not yet implemented
        return network.retrieve_channels_by_code(station_id, location_id,
                                                 channel_id)

    def _getSeisObj(self, channel_obj, start_datetime, end_datetime):
        """
        Return Fissures seismogram object.
        
        Fissures seismogram object is requested from the clients
        network_dc. This actually contains the data.
        
        :param channel_obj: Fissures channel object
        :param start_datetime: UTCDateTime object of starttime
        :param end_datetime: UTCDateTime object of endtime
        :return: Fissures seismogram object
        """
        seisDC = self.rootContext.resolve(self.seis_name)
        seisDC = seisDC._narrow(Fissures.IfSeismogramDC.DataCenter)
        #
        # Transform datetime into correct format
        t1 = self._dateTime2Fissures(start_datetime)
        t2 = self._dateTime2Fissures(end_datetime)
        #
        # Form request for all channels
        request = [Fissures.IfSeismogramDC.RequestFilter(c.id, t1, t2) \
                for c in channel_obj]
        #
        # Retrieve Seismogram object
        return seisDC.retrieve_seismograms(request)
