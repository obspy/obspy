# -*- coding: utf-8 -*-

import os
import numpy as np
from obspy.core import UTCDateTime, Stream, Trace
from obspy.core.preview import mergePreviews, resamplePreview
from numpy.ma import is_masked
import pickle
from copy import deepcopy


class WaveformHandler(object):
    """
    Class that handles the caching or retrieving of the data for all waveforms.
    """
    def __init__(self, env, *args, **kwargs):
        self.env = env
        # Empty dict to handle the waveforms.
        self.waveforms = {}

    def getItem(self, network, station, location, channel):
        """
        Will check whether the requested waveform is already available in the cache
        directory and otherwise fetch it from a SeisHub Server. It will always
        return one stream object for one channel with the as many items as
        self.env.details.
        
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
        id = '%s.%s.%s.%s' % (network, station, location, channel)
        if id in self.waveforms:
            stream = self.waveforms[id]['org_stream']
        # Otherwise get the waveform.
        stream = self.getWaveform(network, station, location, channel, id)
        self.waveforms[id] = {}
        self.waveforms[id]['org_stream'] = stream
        if not stream:
            self.waveforms[id]['empty'] = True
            data = np.empty(self.env.detail)
            data[:] = -1
            trace = Trace(data=data)
            self.waveforms[id]['minmax_stream'] = Stream(traces=[trace])
            return self.waveforms[id]
        self.waveforms[id]['empty'] = False
        # Process the stream_object.
        self.waveforms[id]['minmax_stream'] = \
                self.processStream(self.waveforms[id]['org_stream'])
        return self.waveforms[id]

    def processStream(self, stream):
        """
        Returns a min_max_list.
        """
        pixel = self.env.detail
        stream = deepcopy(stream)
        # Trim to times and pad with masked elements.
        stream.trim(self.env.starttime, self.env.endtime, pad=True)
        # Set masked arrays to -1.
        if is_masked(stream[0].data):
            stream[0].data.fill_value = -1.0
            stream[0].data = stream[0].data.filled()
        length = len(stream[0].data)
        # For debugging purposes. This should never happen!
        if len(stream) != 1:
            print stream
            raise
        resamplePreview(stream[0], pixel)
        return stream

    def getWaveform(self, network, station, location, channel, id):
        """
        Gets the waveform. Loads it from the cache or requests it from SeisHub.
        """
        if self.env.debug and not self.env.seishub.online:
            msg = 'No connection to SeisHub server. Only locally cached ' + \
                  'information is available.'
            print msg
        # Go through directory structure and create all necessary
        # folders if necessary.
        network_path = os.path.join(self.env.cache_dir, network)
        if not os.path.exists(network_path):
            os.mkdir(network_path)
        station_path = os.path.join(network_path, station)
        if not os.path.exists(station_path):
            os.mkdir(station_path)
        files = os.listdir(station_path)
        # Remove all unwanted files.
        files = [file for file in files if file[-7:] == '--cache' and
                 file.split('--')[0] == '%s[%s]' % (channel, location)]
        # If no file exists get it from SeisHub. It will also get cached for
        # future access.
        if len(files) == 0 and self.env.seishub.online:
            if self.env.debug:
                print ' * No cached file found for %s.%s.%s.%s' \
                    % (network, station, location, channel)
            stream = self.getPreview(network, station, location, channel,
                                      station_path)
            return stream
        else:
            # Otherwise figure out if the requested time span is already cached.
            times = [(float(file.split('--')[1]), float(file.split('--')[2]),
                      os.path.join(station_path, file)) for file in files]
            starttime = self.env.starttime.timestamp
            endtime = self.env.endtime.timestamp
            # Times should be sorted anyway so explicit sorting is not necessary.
            # Additionally by design there should be no overlaps.
            missing_time_frames = []
            times = [time for time in times if time[0] <= endtime and time[1] >=
                     starttime]
            if len(times):
                if starttime < times[0][0]:
                    missing_time_frames.append((starttime, times[0][0] +
                                                self.env.buffer))
                for _i in xrange(len(times) - 1):
                    missing_time_frames.append((times[_i][1] - self.env.buffer,
                                    times[_i + 1][0] + self.env.buffer))
                if endtime > times[-1][1]:
                    missing_time_frames.append((times[-1][1] - self.env.buffer,
                                                endtime))
                # Load all cached files.
                stream = self.loadFiles(times)
            else:
                missing_time_frames.append((self.env.starttime -
                        self.env.buffer, self.env.endtime + self.env.buffer))
                stream = Stream()
            # Get the gaps.
            if missing_time_frames and self.env.seishub.online:
                if self.env.debug:
                    print ' * Only partially cached file found for %s.%s.%s.%s.' \
                          % (network, station, location, channel) + \
                          ' Requesting the rest from SeisHub...'
                stream += self.loadGaps(missing_time_frames, network, station,
                                        location, channel)
                if not stream:
                    msg = 'No data available for %s.%s.%s.%s for the selected timeframes'\
                        % (network, station, location, channel)
                    self.win.status_bar.setError(msg)
                    return
            else:
                if self.env.debug:
                    print ' * Cached file found for %s.%s.%s.%s' \
                        % (network, station, location, channel)
            # XXX: Pretty ugly to ensure all data has the same dtype.
            for trace in stream:
                trace.data = np.require(trace.data, dtype='float32')
            # Merge everything and pickle once again.
            stream = mergePreviews(stream)
            # Pickle the stream object for future reference. Do not pickle it if it
            # is smaller than 200 samples. Just not worth the hassle.
            if stream[0].stats.npts > 200:
                # Delete all the old files.
                for _, _, file in times:
                    os.remove(file)
                filename = os.path.join(station_path, '%s[%s]--%s--%s--cache' % \
                            (channel, location, str(stream[0].stats.starttime.timestamp),
                             str(stream[0].stats.endtime.timestamp)))
                file = open(filename, 'wb')
                pickle.dump(stream, file, 2)
                file.close()
            return stream

    def loadGaps(self, frames, network, station, location, channel):
        """
        Returns a stream object that will contain all time spans from the
        provided list.
        """
        streams = []
        for frame in frames:
            temp = self.env.seishub.getPreview(network, station, location,
                        channel, UTCDateTime(frame[0]), UTCDateTime(frame[1]))
            # XXX: Not necessary in the future once SeisHub updates itsself.
            temp[0].stats.preview = True
            start = temp[0].stats.starttime
            temp[0].stats.starttime = UTCDateTime(start.year, start.month,
                                                  start.day, start.hour,
                                                  start.minute, start.second)
            # Convert to float32
            if len(temp):
                streams.append(temp)
        if len(streams):
            stream = streams[0]
            if len(streams) > 1:
                for _i in streams[1:]:
                    stream += _i
        else:
            stream = Stream()
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

    def getPreview(self, network, station, location, channel, station_path,
                starttime=None, endtime=None):
        """
        Actually get the file.
        """
        stream = self.env.seishub.getPreview(network, station, location,
                 channel, starttime, endtime)
        # None will be returned if some server error prevented the preview from
        # getting retrieved. Also do not show empty Stream object.
        if stream is None or not len(stream):
            return None
        # It will always return exactly one Trace. Make sure the data is in
        # float32.
        # XXX: Not necessary in the future once SeisHub updates itsself.
        stream[0].stats.preview = True
        start = stream[0].stats.starttime
        stream[0].stats.starttime = UTCDateTime(start.year, start.month,
                                                start.day, start.hour,
                                                start.minute, start.second)
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
