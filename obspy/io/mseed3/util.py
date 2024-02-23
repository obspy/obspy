

def mseed3_to_obspy_header(ms3):
    stats = {}
    h=ms3.header
    stats['npts'] = h.numSamples
    stats['sampling_rate'] = h.sampleRate
    sid = FDSNSourceId(ms3.identifier)
    nslc = sid.asNslc()
    stats['network'] = nslc.networkCode
    stats['station'] = nslc.stationCode
    stats['location'] = nslc.locationCode
    stats['channel'] = nslc.channelCode

    # store _all_ provided SAC header values
    if 'bag' in ms3.eh:
        stats['bag'] = ms3.eh['bag']
    if 'FDSN' in ms3eh:
        stats['FDSN'] = ms3eh['FDSN']

    stats['starttime'] = UTCDateTime(ms3.starttime)

    return Stats(stats)
