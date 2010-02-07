#!/usr/bin/env python
# -*- coding: utf-8 -*-

from obspy.core import read, Stream

class WaveformHandler(object):
    """
    Class that handles all waveform reading and transforming operations.
    """

    def __init__(self, env):
        """
        Not much to initialize in here.
        """
        self.env = env
        # Maximum sampling_rate allowed. This corresponds to one sample every
        # minute.
        self.MAX_FREQ = 1.0/60.0
        # Empty stream object that will store all traces being used.
        self.stream = Stream()

    def readFile(self, filename):
        """
        Reads the file and appends it to self.stream.
        """
        self.temp_stream = read(filename)
        try:
            self._processTempStream()
        except CaughtException:
            return
        self.stream += self.temp_stream

    def _processTempStream(self):
        """
        Takes the stream object in self.temp_stream, makes some sanity checks
        and creates a logarithmic kind of scale.
        """
        # Raise an exception if the sampling rate is higher than self.MAX_FREQ.
        for trace in self.temp_stream():
            if trace.stats.sampling_rate > self.MAX_FREQ:
                msg = 'The sampling rate of the index is to large. Please '+\
                      'use a smaller sampling_rate to avoid performance '+\
                      'issues.'
                self.env.errorHandler.addError(msg)
                raise CaughtException
        # The index file must only contain one Trace. All merging stuff should
        # be handled by the indexer.
        if len(self.temp_stream) > 1:
            msg = 'The index file contains more than one Trace.'
            self.env.errorHandler.addError(msg)
