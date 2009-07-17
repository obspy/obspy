from obspy.core import UTCDateTime, Stats


class Trace(object):
    """
    ObsPy Trace class.
    
    This class contains information about a single trace.
    
    @type data: Numpy ndarray 
    @ivar data: Data samples 
    """
    def __init__(self, header=None, data=None):
        self.stats = Stats()
        self.data = None
        if header != None:
            for _i in header.keys():
                if type(header[_i]) == dict:
                    self.stats[_i] = Stats(dummy=False)
                    for _j in header[_i].keys():
                        self.stats[_i][_j] = header[_i][_j]
                else:
                    self.stats[_i] = header[_i]
        if data != None:
            self.data = data

    def __str__(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s | " + \
              "%(starttime)s - %(endtime)s | " + \
              "%(sampling_rate).1f Hz, %(npts)d samples"
        return out % (self.stats)

    def __len__(self):
        """
        Returns the number of data samples of a L{Trace} object.
        """
        return len(self.data)

    def getId(self):
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out % (self.stats)

    def plot(self, **kwargs):
        """
        Creates a graph of this L{Trace} object.
        """
        try:
            from obspy.imaging import waveform
        except:
            msg = "Please install module obspy.imaging to be able to " + \
                  "plot ObsPy Trace objects."
            print msg
            raise
        waveform.plotWaveform(self, **kwargs)

    def ltrim(self, starttime):
        """
        Cuts L{Trace} object to given start time.
        """
        if isinstance(starttime, float) or isinstance(starttime, int):
            starttime = UTCDateTime(self.stats.starttime) + starttime
        elif not isinstance(starttime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if starttime <= self.stats.starttime or \
           starttime >= self.stats.endtime:
            return
        # cut from left
        delta = (starttime - self.stats.starttime)
        samples = int(round(delta * self.stats.sampling_rate))
        self.data = self.data[samples:]
        self.stats.npts = len(self.data)
        self.stats.starttime = starttime

    def rtrim(self, endtime):
        """
        Cuts L{Trace} object to given end time.
        """
        if isinstance(endtime, float) or isinstance(endtime, int):
            endtime = UTCDateTime(self.stats.endtime) - endtime
        elif not isinstance(endtime, UTCDateTime):
            raise TypeError
        # check if in boundary
        if endtime >= self.stats.endtime or endtime < self.stats.starttime:
            return
        # cut from right
        delta = (self.stats.endtime - endtime)
        samples = int(round(delta * self.stats.sampling_rate))
        total = len(self.data) - samples
        if endtime == self.stats.starttime:
            total = 1
        self.data = self.data[0:total]
        self.stats.npts = len(self.data)
        self.stats.endtime = endtime

    def trim(self, starttime, endtime):
        """
        Cuts L{Trace} object to given start and end time.
        """
        # check time order and switch eventually
        if starttime > endtime:
            endtime, starttime = starttime, endtime
        # cut it
        self.ltrim(starttime)
        self.rtrim(endtime)
