#!/usr/bin/env python
# REQUIRES PYTHON 2.5

import os, numpy as N, ctypes as C
from obspy.filter import seisSim
from obspy.seishub.client import Client
from obspy.core import UTCDateTime

client = Client("http://admin:admin@teide.geophysik.uni-muenchen.de:8080")



def xcorr(tr1, tr2, window_len):
    """
    Crosscorreltation of tr1 and tr2 in the time domain using window_len.
    
    >>> tr1 = N.random.randn(10000).astype('float32')
    >>> tr2 = tr1.copy()
    >>> a,b = xcorr(tr1, tr2, 1000)
    >>> a, int(1e6*b) # Rounding Errors
    (0, 1000000)
    
    @type tr1: numpy ndarray float32
    @param tr1: Trace 1
    @type tr2: numpy ndarray float32
    @param tr2: Trace 2 to correlate with trace 1
    @type window_len: Int
    @param window_len: Window length of cross correlation in samples
    """
    # 2009-07-10 Moritz
    lib = C.CDLL(os.path.join(os.path.dirname(__file__),'./xcorr.so'))
    #lib = C.CDLL(os.path.join(os.path.dirname(__file__),'xcorr.dll'))
    lib.X_corr.argtypes=[N.ctypeslib.ndpointer(dtype='float32', ndim=1,
                                           flags='C_CONTIGUOUS'),
                     N.ctypeslib.ndpointer(dtype='float32', ndim=1,
                                           flags='C_CONTIGUOUS'),
                     C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    lib.X_corr.restype=C.c_void_p
    
    shift = C.c_int()
    coe_p = C.c_double()
    
    lib.X_corr(tr1, tr2, window_len, len(tr1), len(tr2),
               C.byref(shift), C.byref(coe_p) )
    
    return shift.value, coe_p.value


def getStationPAZ(network_id, station_id, datetime):
    """
    """
    # request station information
    station_list = client.station.getList(network_id=network_id, 
                                          station_id=station_id,
                                          datetime=datetime)
    if len(station_list)!=1:
        return {}
    # request station resource
    res = client.station.getXMLResource(station_list[0]['resource_name'])
    # get poles
    node = res.station_control_header.response_poles_and_zeros
    poles_real = node.complex_pole.real_pole[:]    
    poles_imag = node.complex_pole.imaginary_pole[:]
    poles = zip(poles_real, poles_imag)
    poles = [p[0] + p[1]*1j for p in poles]
    # get zeros
    node = res.station_control_header.response_poles_and_zeros
    zeros_real = node.complex_zero.real_zero[:][:]    
    zeros_imag = node.complex_zero.imaginary_zero[:][:]
    zeros = zip(zeros_real, zeros_imag)
    zeros = [p[0] + p[1]*1j for p in zeros]
    # get gain
    # XXX: not sure about that!
    node = res.station_control_header
    gains = node.xpath('channel_sensitivity_gain/sensitivity_gain')
    gain = float(gains[-1])/1000
    return {'poles': poles, 'zeroes': zeros, 'gain': gain}


def xcorrEvents(starttime, endtime, network_id='*', station_id='*', 
                location_id='', channel_id='EHE', phase='P',
                time_window=(-1, 6), winlen=10.0):
    """
    """
    # get all events betwenn start and end time
    event_list = client.event.getList(datetime = (starttime, endtime))
    networks = {}
    print "Fetching events ..."
    for event in event_list:
        id = event['resource_name']
        print "EVENT:", event['datetime'], id
        # request event resource
        res = client.event.getXMLResource(id)
        # fetch all picks with given phase
        pick_list = res.xpath("/event/pick[phaseHint='%s']" % phase)
        # cycle through picks
        streams = []
        for pick in pick_list:
            temp = {}
            # XXX: ignoring location code for now
            dt = UTCDateTime(str(pick.time.value))
            sid = pick.waveform.attrib['stationCode']
            nid = pick.waveform.attrib['networkCode'] or 'BW'
            #print '   Got Pick:', nid, sid, dt
            # generate station/network list
            networks.setdefault(nid, {})
            networks[nid].setdefault(sid, [])
            networks[nid][sid].append((event, dt))

    print 
    print "Correlate events over each stations ..."
    # cycle through all networks/stations/events
    for nid, stations in networks.iteritems():
        for sid, events in stations.iteritems():
            print
            print "%s.%s:" % (nid, sid)
            if len(events)<2:
                print "  -> Skipping: Need at least 2 events per station"
                continue
            # output file
            filename = "%s.%s.csv" % (nid, sid)
            fp = open(filename, "w")
            # body
            streams = []
            for event in events:
                id = event[0]['resource_name']
                dt = event[1]
                # get station PAZ for this date time
                paz = getStationPAZ(nid, sid, dt)
                if not paz:
                    print "!!! Missing PAZ for %s.%s for %s" % (nid, sid, dt)
                    continue
                # get waveforms
                try:
                    stream = client.waveform.getWaveform(nid, sid, location_id, 
                                                         channel_id, 
                                                         dt+time_window[0],
                                                         dt+time_window[1])
                except:
                    msg = "!!! Error fetching waveform for %s.%s.%s.%s for %s"
                    print msg % (nid, sid, location_id, channel_id, dt)
                    continue
                # trim to time window
                stream.trim(dt+time_window[0], dt+time_window[1])
                for trace in stream:
                    # calculate zero mean
                    trace.data = trace.data - trace.data.mean()
                    # instrument correction
                    trace.data = seisSim(trace.data, trace.stats.sampling_rate, 
                                         paz, inst_sim=None, water_level=50.0)
                    print '  Got Trace:', trace           
                # append
                streams.append(stream)
                if event == events[0]:
                    fp.write('# ' + id)
                else:
                    fp.write(',' + id)
            fp.write("\n")
            # cross correlation over all prepared streams
            l = len(streams)
            if l<2:
                print "Need at least 2 events per station"
                continue
            #print "XCORR:"
            for i in range(0,l):
                fp.write("\n")
                tr1 = streams[i][0]
                for j in range(0,l):
                    tr2 = streams[j][0]
                    #print '  ' , i, ' x ', j, ' = ',  
                    # check samling rate for both traces
                    if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
                        print
                        print "!!! Sampling rate are not equal!"
                        continue
                    if tr1.stats.npts != tr2.stats.npts:
                        print
                        print "!!! Number of samples in time window are not equal!"
                        continue
                    # devide by 2.0 as in eventcluster line 604: param = windowlen/2
                    winlen = int(winlen / float(tr1.stats.sampling_rate) / 2.0)
                    shift, coe = xcorr(tr1.data.astype('float32'),  
                                       tr2.data.astype('float32'), winlen)
                    #print "%.4lf" % coe
                    if j == 0:
                        fp.write("%.3f" % coe)
                    else:
                        fp.write(",%.3f" % coe)
            print
            print  
            fp.close()




start = UTCDateTime(2009, 7, 1)
end = UTCDateTime(2009, 8, 1)
xcorrEvents(start, end)

#if __name__ == '__main__':
#    import doctest
#    doctest.testmod(exclude_empty=True)

