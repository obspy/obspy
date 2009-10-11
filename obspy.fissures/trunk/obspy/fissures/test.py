#import sys
from omniORB import CORBA
#, PortableServer
#from idl import IfSeismogramDC_idl
#from idl.Fissures_idl import Time
#from idl.IfNetwork_idl import Channel, NetworkAccess, NetworkFinder, Station, NetworkDC
#from idl.IfSeismogramDC_idl import DataCenter, LocalSeismogram, RequestFilter

#orb = CORBA.ORB_init()
#o = orb.string_to_object("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService")
import pdb;pdb.set_trace()
#o = o._narrow(Fortune.CookieServer)
#print o.get_cookie()


#        // Pick a name server to get FISSURES servers.
#        FissuresNamingService namingService = new FissuresNamingService(orb);
#        namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");



#import edu.iris.Fissures.IfNetwork.NetworkDCOperations;
#import edu.iris.Fissures.model.AllVTFactory;
#import edu.sc.seis.fissuresUtil.namingService.FissuresNamingService;
#
#public class SeismogramClient {
#
#        // used.
#        org.omg.CORBA_2_3.ORB orb = (org.omg.CORBA_2_3.ORB)org.omg.CORBA.ORB.init(new String[] {},
#                                                                                  new Properties());
#        // Registers the FISSURES classes with the ORB
#        new AllVTFactory().register(orb);
#        // Pick a name server to get FISSURES servers.
#        FissuresNamingService namingService = new FissuresNamingService(orb);
#        namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");
#        NetworkDCOperations netDC = namingService.getNetworkDC("edu/iris/dmc",
#                                                               "IRIS_NetworkDC");
#        NetworkFinder netFinder = netDC.a_finder();
#        NetworkAccess net = netFinder.retrieve_by_code("IU")[0];
#        Station[] stations = net.retrieve_stations();
#        Channel[] channels = net.retrieve_for_station(stations[0].get_id());
#        DataCenter seisDC = namingService.getSeismogramDC("edu/iris/dmc",
#                                                          "IRIS_PondDataCenter");
#        RequestFilter[] seismogramRequest = new RequestFilter[1];
#        Time start = new Time("2003-06-20T06:23:25.0000Z", -1);
#        Time end = new Time("2003-06-20T06:43:25.0000Z", -1);
#        seismogramRequest[0] = new RequestFilter(channels[0].get_id(),
#                                                 start,
#                                                 end);
#        logger.info("querying for " + seismogramRequest[0].channel_id.network_id.network_code + "."
#                    + seismogramRequest[0].channel_id.station_code + "."
#                    + seismogramRequest[0].channel_id.site_code + "."
#                    + seismogramRequest[0].channel_id.channel_code);
#        LocalSeismogram[] seis = seisDC.retrieve_seismograms(seismogramRequest);
#        for(int i=0; i < seis.length; i + +) {
#            logger.info("seis[" + i + "] has " + seis[i].num_points
#                    + " points and starts at " + seis[i].begin_time.date_time);
#        }
