# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from obspy.seishub.client import Client
from obspy.signal import seisSim, cornFreq2Paz, bandpassZPHSH
from obspy.signal.util import xcorr
import os, fnmatch
import numpy as N
import ctypes as C


def xcorrEvents(starttime, endtime, network_id='*', station_id='*',
                location_id='', channel_id='EHZ', phase='P',
                time_window=(-1, 6), method='manual', merge=True):
    """
    @param method: 'manual' or 'auto' or None.
    """
    wildcard = "%s.%s.%s.%s" % (network_id, station_id,
                                location_id, channel_id)
    # PAZ of instrument to simulate, 2.0Hz corner-frequency, 0.707 damping
    inst = cornFreq2Paz(2.0)
    # get all events between start and end time
    client = Client("http://teide.geophysik.uni-muenchen.de:8080",
                    user="admin", password="admin")
    event_list = client.event.getList(datetime=(starttime, endtime),
                                      localisation_method=method)

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
            try:
                dt = UTCDateTime(str(pick.time.value))
            except:
                continue
            sid = pick.waveform.attrib['stationCode']
            nid = pick.waveform.attrib['networkCode'] or 'BW'
            cid = pick.waveform.attrib['channelCode']
            lid = pick.waveform.attrib['locationCode']
            pid = '%s.%s.%s.%s' % (nid, sid, lid, cid)
            print "    PICK: %s - %s - %s" % (pid, phase, dt)
            if not fnmatch.filter([pid], wildcard):
                continue
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
                if merge:
                    stream.merge()
                for trace in stream:
                    # calculate zero mean
                    trace.data = trace.data - trace.data.mean()
                    # instrument correction
                    #trace.data = seisSim(trace.data, trace.stats.sampling_rate,
                    #                     paz, inst_sim=inst, water_level=50.0)
                    trace.data = bandpassZPHSH(trace.data,2.0,20.0,
                                               df=trace.stats.sampling_rate,
                                               corners=4)
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
            # xcorr
            for i in range(0, l - 1):
                id1 = streams[i][0]
                tr1 = streams[i][1][0]
                for j in range(i + 1, l):
                    id2 = streams[j][0]
                    tr2 = streams[j][1][0]
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
                    # remove last sample if npts is an odd number
                    delta = -1 * (tr1.stats.npts % 2)
                    winlen = int((tr1.stats.npts + delta) / 2.0)
                    shift, coe = xcorr(tr1.data[:delta].astype('float32'),
                                       tr2.data[:delta].astype('float32'),
                                       winlen)
                    fp.write("%d %d %.3f %d %s %s\n" % (i + 1, j + 1, coe,
                                                        shift, id1, id2))
            print
            fp.close()


start = UTCDateTime(2007, 9, 20)
end = UTCDateTime(2007, 11, 20) - 1
xcorrEvents(start, end, network_id='BW', station_id='RNON',
            time_window=(-0.5, 2.5))
