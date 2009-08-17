# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.seishub.client import Client
from obspy.signal import seisSim
from obspy.signal.util import xcorr
import os
import numpy as N
import ctypes as C


def xcorrEvents(starttime, endtime, network_id='*', station_id='*',
                location_id='', channel_id='EHZ', phase='P',
                time_window=(-1, 6), winlen=10.0, method='manual'):
    """
    @param method: 'manual' or 'auto' or None.
    """
    # get all events between start and end time
    client = Client("http://teide.geophysik.uni-muenchen.de:8080",
                    user="admin", password="admin")
    if method != None:
        event_list = client.event.getList(datetime=(starttime, endtime),
                                          localisation_method=method)
    else:
        event_list = client.event.getList(datetime=(starttime, endtime))

    print "Fetching events ..."
    networks = {}
    for event in event_list:
        id = event['resource_name']
        print "  EVENT:", str(event['datetime']), id
        # request event resource
        res = client.event.getXMLResource(id)
        # fetch all picks with given phase
        pick_list = res.xpath("/event/pick[phaseHint='%s']" % phase)
        # cycle through picks
        streams = []
        for pick in pick_list:
            temp = {}
            # XXX: ignoring location code for now
            try:
                dt = UTCDateTime(str(pick.time.value))
            except:
                continue
            sid = pick.waveform.attrib['stationCode']
            nid = pick.waveform.attrib['networkCode'] or 'BW'
            cid = pick.waveform.attrib['channelCode']
            lid = pick.waveform.attrib['locationCode']
            print "    PICK: %s.%s.%s.%s - %s - %s" % (nid, sid, lid, cid,
                                                       phase, dt)
            # generate station/network list
            networks.setdefault(nid, {})
            networks[nid].setdefault(sid, [])
            networks[nid][sid].append((event, dt))

    print
    print "Correlate events over each station ..."
    # cycle through all networks/stations/events
    for nid, stations in networks.iteritems():
        for sid, events in stations.iteritems():
            print "  %s.%s:" % (nid, sid)
            if len(events) < 2:
                print "    -> Skipping: Need at least 2 events per station"
                print
                continue
            streams = []
            for event in events:
                id = event[0]['resource_name']
                dt = event[1]
                # get station PAZ for this date time
                paz = client.station.getPAZ(nid, sid, dt, location_id,
                                            channel_id)
                if not paz:
                    print "!!! Missing PAZ for %s.%s for %s" % (nid, sid, dt)
                    continue
                # get waveforms
                try:
                    stream = client.waveform.getWaveform(nid, sid, location_id,
                                                         channel_id,
                                                         dt + time_window[0],
                                                         dt + time_window[1])
                except:
                    msg = "!!! Error fetching waveform for %s.%s.%s.%s for %s"
                    print msg % (nid, sid, location_id, channel_id, dt)
                    continue
                for trace in stream:
                    # calculate zero mean
                    trace.data = trace.data - trace.data.mean()
                    # instrument correction
                    trace.data = seisSim(trace.data, trace.stats.sampling_rate,
                                         paz, inst_sim=None, water_level=50.0)
                    print '    Got Trace:', trace
                # append
                streams.append((id, stream))
            # cross correlation over all prepared streams
            l = len(streams)
            if l < 2:
                print "    -> Skipping: Need at least 2 events per station"
                print
                fp.close()
                continue
            # output file
            filename = "%s.%s.txt" % (nid, sid)
            fp = open(filename, "w")
            #print "XCORR:"
            for i in range(0, l - 1):
                id1 = streams[i][0]
                tr1 = streams[i][1][0]
                for j in range(i + 1, l):
                    id2 = streams[j][0]
                    tr2 = streams[j][1][0]
                    #print '  ' , i, ' x ', j, ' = ',  
                    # check sampling rate for both traces
                    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
                        print
                        print "!!! Sampling rate are not equal!"
                        continue
                    if tr1.stats.npts != tr2.stats.npts:
                        print
                        print "!!! Number of samples are not equal!"
                        continue
                    # divide by 2.0 as in eventcluster.c line 604
                    winlen = int(winlen / float(tr1.stats.sampling_rate) / 2.0)
                    shift, coe = xcorr(tr1.data.astype('float32'),
                                       tr2.data.astype('float32'), winlen)
                    fp.write("%d %d % .3f %d %s %s\n" % (i + 1, j + 1, coe,
                                                         shift, id1, id2))
            print
            fp.close()


start = UTCDateTime(2009, 7, 1)
end = UTCDateTime(2009, 8, 1) - 1
xcorrEvents(start, end)
