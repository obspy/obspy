#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import numpy as np
import pickle
import os
from waveform_plot import WaveformPlot

class Utils(GUIElement):
    """
    Some helpful methods. Has its own class to be able to inherit the
    environment and windows.
    """
    def __init__(self, *args, **kwargs):
        """
        Standart init call.
        """
        super(Utils, self).__init__(self, **kwargs)
        # Additional seconds that will be requested from SeisHub as a buffer.
        self.buffer = 5

    def add_plot(self, network, station, location, channel):
        """
        Will check whether the requested waveform is already available in the cache
        directory and otherwise fetch it from a SeisHub Server.
        
        The cache directory has the following structure:

        cache
        L-- network
            L-- station
                L-- channel[location]--starttime_timestamp--endtime_timestamp.cache
        """
        network = str(network)
        station = str(station)
        location = str(location)
        channel = str(channel)
        id = network + '.' + station + '.' + location + '.' + channel
        # XXX: Wildcards not implemented yet.
        if '*' in id:
            return
        # Check if it already exists.
        for waveform in self.win.waveforms:
            if waveform.header  == id:
                return
        # Go through directory structure and create all necessary folders if
        # necessary.
        network_path = os.path.join(self.env.cache_dir, network)
        if not os.path.exists(network_path):
            os.mkdir(network_path)
        station_path = os.path.join(network_path, station)
        if not os.path.exists(station_path):
            os.mkdir(station_path)
        files = os.listdir(station_path)
        # Remove all unwanted files.
        files = [file for file in files if file[-7:] == '--cache' and file.split('--')[0] == '%s[%s]' % (channel, location)]
        # If not file exists move on.
        if len(files) == 0:
            stream = self.getWaveform(network, station, location, channel, station_path)
        else:
            # Otherwise figure out if the requested time span is already cached.
            times = [(float(file.split('--')[1]), (file.split('--')[2]),
                      os.path.join(station_path, file)) for file in files]
            starttime = self.win.starttime.timestamp
            endtime = self.win.endtime.timestamp
            # Times should be sorted anyway so explicit sorting is not necessary.
            # Additionally by design there should be no overlaps.
            missing_time_frames = []
            times = [time for time in times if time[0] <= endtime and time[1] >=
                     starttime]
            if starttime < times[0][0]:
                missing_time_frames.append((starttime, times[0][0] + self.buffer))
            for _i in xrange(len(times) - 1):
                missing_time_frames.append((times[_i][1] - self.buffer,
                                times[_i + 1][0] + self.buffer))
            if endtime > times[-1][1]:
                missing_time_frames.append((times[-1][1] - self.buffer, endtime))
            # Load all cached files.
            stream = self.loadFiles(times)
            # Get the gaps.
            stream += self.loadGaps(missing_time_frames)
            # Merge everything and pickle once again.
            stream.merge(method = 1, interpolation_samples = -1)
            # Pickle the stream object for future reference. Do not pickle it if it
            # is smaller than 200 samples. Just not worth the hassle.
            if stream[0].stats.npts > 200:
                filename = os.path.join(station_path, '%s[%s]--%s--%s--cache' % \
                            (channel, location, str(stream[0].stats.starttime.timestamp),
                             str(stream[0].stats.endtime.timestamp)))
                file = open(filename, 'wb')
                pickle.dump(stream, file, 2)
                file.close()
        if len(stream):
            WaveformPlot(parent = self.win, group = 2, stream = stream)

    def loadGaps(frames, network, station, location, channel):
        """
        Returns a stream object that will contain all time spans from the
        provided list.
        """
        streams = []
        for frame in missing_time_frames:
            temp = self.win.seishub.getPreview(network, station, location,
                                        channel, frames[0], frames[1])
            # Convert to float32.
            temp[0].data = np.require(temp[0].data)
            streams.append(temp)
        stream = streams[0]
        if len(streams) > 1:
            for _i in streams[1:]:
                stream += _i
        return stream


    def loadFiles(self, times):
        """
        Loads all necessary cached files.
        """
        streams = []
        for _t in times:
            file = open(_t[2], 'rb')
            streams.append(pickle.load(file))
            file.close()
        stream = streams[0]
        if len(streams) > 1:
            for _i in streams[1:]:
                stream += _i
        return stream

    def getWaveform(self, network, station, location, channel, station_path,
                starttime = None, endtime = None):
        """
        Actually get the file.
        """
        stream = self.win.seishub.getPreview(network, station, location,
                 channel, starttime, endtime)
        # It will always return exactly one Trace. Make sure the data is in
        # float32.
        stream[0].data = np.require(stream[0].data, 'float32')
        # Pickle the stream object for future reference. Do not pickle it if it
        # is smaller than 200 samples. Just not worth the hassle.
        if stream[0].stats.npts > 200:
            filename = os.path.join(station_path, '%s[%s]--%s--%s--cache' % \
                        (channel, location, str(stream[0].stats.starttime.timestamp),
                         str(stream[0].stats.endtime.timestamp)))
            file = open(filename, 'wb')
            pickle.dump(stream, file, 2)
            file.close()
        return stream


        
            




