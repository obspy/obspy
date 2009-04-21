# -*- coding: utf-8 -*-

class Stats(dict):
    """
    A stats class which behaves like a dictionary.
    
    You may the following syntax to change or access data in this class:
      >>> stats = Stats()
      >>> stats.network = 'BW'
      >>> stats['station'] = 'ROTZ'
      >>> stats.get('network')
      'BW'
      >>> stats['network']
      'BW'
      >>> stats.station
      'ROTZ'
      >>> stats.keys()
      ['network', 'station']
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def getParser():
    """
    Collects all obspy parser classes.
    """
    temp = []
    try:
        from obspy.mseed.core import MSEEDTrace
        temp.append(MSEEDTrace)
    except:
        pass
    try:
        from obspy.gse2.core import GSE2Trace
        temp.append(GSE2Trace)
    except:
        pass
    try:
        from obspy.sac.core import SACTrace
        temp.append(SACTrace)
    except:
        pass
    try:
        from obspy.wav.core import WAVTrace
        temp.append(WAVTrace)
    except:
        pass
    return tuple(temp)
