#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:author: Luca Trani
:contact: trani@knmi.nl


:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from obspy.core import UTCDateTime, read, Stream
from obspy.io.mseed.util import get_flags, get_start_and_end_time
import numpy as np
import json
from collections import defaultdict


class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPyEncoder, self).default(obj)


class MSEEDMetadata():
    """
    A container for MSEED specific metadata including QC
    """

    def __init__(self):
        self.__msmeta__ = defaultdict()
        self.__samples__ = []
        self.__num_records__ = 0
        self.__num_samples__ = 0
        self.__st__ = Stream()
        self.start_time = None
        self.end_time = None
        self.files = []

    def __read__(self, files, start_time, end_time):
        """
        Assumes that each file contains a
        single stream (scnl)
        Files should represent consecutive
        ordered time windows:
        e.g. daily files
        """
        if(len(files) > 1):
            first_file = files.pop(0)
            first_end_time = get_start_and_end_time(first_file)[1]
            last_file = files.pop()
            last_start_time = get_start_and_end_time(first_file)[0]
            if first_end_time > start_time:
                self.__st__ = read(first_file)
                self.files.append(first_file)
            for file in files:
                self.__st__ += read(file)
                self.files.append(file)
            if last_start_time < end_time:
                self.__st__ += read(last_file)
                self.files.append(last_file)
        else:
            for file in files:
                self.__st__ += read(file)
                self.files.append(file)
        self.start_time = start_time
        self.end_time = end_time
        if start_time is not None or end_time is not None:
            # cut the Stream to the requested time window
            self.__st__.trim(starttime=start_time, endtime=end_time)
        for t in self.__st__:
            self.__samples__.extend(t.data)
            self.__num_records__ += t.stats.mseed.number_of_records
            self.__num_samples__ += t.stats.npts

    def __extract_mseed_stream_metadata__(self):
        """
        Extracts metadata from the MSEED header
        and populates the msmeta dictionary
        """
        if(self.__st__.__nonzero__()):
            stats = self.__st__[0].stats
            self.__msmeta__['net'] = stats.network
            self.__msmeta__['sta'] = stats.station
            self.__msmeta__['cha'] = stats.channel
            self.__msmeta__['loc'] = stats.location
            self.__msmeta__['files'] = self.files
            # start time of the requested available stream
            self.__msmeta__['start_time'] = self.__st__[0].\
                stats['starttime'].isoformat()
            # end time of the requested available stream
            self.__msmeta__['end_time'] = self.__st__[-1].\
                stats['endtime'].isoformat()
            self.__msmeta__['sample_rate'] = stats.sampling_rate
            self.__msmeta__['record_len'] = stats.mseed.record_length
            self.__msmeta__['quality'] = stats.mseed.dataquality
            self.__msmeta__['encoding'] = stats.mseed.encoding
            self.__msmeta__['num_records'] = self.__num_records__
            self.__msmeta__['num_samples'] = len(self.__samples__)
            if len(self.files) > 1:
                data_quality_flags = [0, 0, 0, 0, 0, 0, 0, 0]
                activity_flags = [0, 0, 0, 0, 0, 0, 0]
                io_and_clock_flags = [0, 0, 0, 0, 0, 0]
                timing_quality = []
                for f in self.files:
                    flags = get_flags(f, starttime=self.start_time,
                                      endtime=self.end_time,
                                      t_quality=True)
                    data_quality_flags = [sum(i)
                                          for i in zip(
                        data_quality_flags, flags['data_quality_flags'])]
                    activity_flags = [sum(i)
                                      for i in zip(
                        activity_flags, flags['activity_flags'])]
                    io_and_clock_flags = [sum(i)
                                          for i in zip(
                        io_and_clock_flags, flags['io_and_clock_flags'])]
                    if 'timing_quality' in flags:
                        timing_quality.extend(flags['timing_quality'])
                count = len(timing_quality)
                timing_quality_mean = sum(timing_quality) / count \
                    if(count > 0) else None
                timing_quality_min = min(timing_quality) \
                    if(count > 0) else None
                timing_quality_max = max(timing_quality) \
                    if(count > 0) else None
            else:
                flags = get_flags(self.files[0],
                                  starttime=self.start_time,
                                  endtime=self.end_time)
                data_quality_flags = flags['data_quality_flags']
                activity_flags = flags['activity_flags']
                io_and_clock_flags = flags['io_and_clock_flags']
                timing_quality_mean = flags['timing_quality_average'] \
                    if ('timing_quality_average' in flags) else None
                timing_quality_min = flags['timing_quality_min'] \
                    if ('timing_quality_min' in flags) else None
                timing_quality_max = flags['timing_quality_max'] \
                    if ('timing_quality_max' in flags) else None
            self.__msmeta__['glitches'] = data_quality_flags[3]
            self.__msmeta__['amplifier_saturation'] = data_quality_flags[0]
            self.__msmeta__['digital_filter_charging'] = data_quality_flags[6]
            self.__msmeta__['digitizer_clipping'] = data_quality_flags[1]
            self.__msmeta__['missing_padded_data'] = data_quality_flags[4]
            self.__msmeta__['spikes'] = data_quality_flags[2]
            self.__msmeta__['suspect_time_tag'] = data_quality_flags[7]
            self.__msmeta__['telemetry_sync_error'] = data_quality_flags[5]
            self.__msmeta__['calibration_signal'] = activity_flags[0]
            self.__msmeta__['event_begin'] = activity_flags[2]
            self.__msmeta__['event_end'] = activity_flags[3]
            self.__msmeta__['event_in_progress'] = activity_flags[6]
            self.__msmeta__['timing_correction'] = activity_flags[1]
            self.__msmeta__['clock_locked'] = io_and_clock_flags[5]
            self.__msmeta__['timing_quality_mean'] = timing_quality_mean
            self.__msmeta__['timing_quality_min'] = timing_quality_min
            self.__msmeta__['timing_quality_max'] = timing_quality_max

    def __compute_sample_metrics__(self):
        """
        Computes metrics on samples contained in the specified time window
        """
        if(self.__st__.__nonzero__()):
            gaps = self.__st__.getGaps()
            self.__msmeta__['sample_rms'] = np.sqrt(
                sum([np.square(n) for n in self.__samples__]) /
                len(self.__samples__)
                )
            self.__msmeta__['sample_mean'] = np.mean(self.__samples__)
            self.__msmeta__['sample_min'] = np.min(self.__samples__)
            self.__msmeta__['sample_max'] = np.max(self.__samples__)
            self.__msmeta__['sample_stdev'] = np.std(self.__samples__)
            self.__msmeta__['num_gaps'] = len(
                [gap for gap in gaps if gap[6] > 0]
                ) + self.__get_head_and_trail_gaps()
            self.__msmeta__['num_overlaps'] = len(
                [gap for gap in gaps if gap[6] < 0]
                )
            self.__msmeta__['overlaps_len'] = abs(
                sum([gap[6] for gap in gaps if gap[6] < 0])
                )
            self.__msmeta__['gaps_len'] = sum(
                [gap[6] for gap in gaps if gap[6] > 0]) + \
                ((self.__st__[0].stats['starttime'] - self.start_time) +
                 (self.end_time - self.__st__[-1].stats['endtime']))
            start_time = self.start_time
            end_time = self.end_time
            # set the availability with respect to the
            # requested time interval
            self.__msmeta__['percent_availability'] = 100*(
                (end_time - start_time - self.__msmeta__['gaps_len']) /
                (end_time - start_time)
                )

    def __get_head_and_trail_gaps(self):
        extra_gaps = 0
        if(self.__st__[0].stats['starttime'] > self.start_time):
            extra_gaps += 1
        if(self.end_time > self.__st__[-1].stats['endtime']):
            extra_gaps += 1
        return extra_gaps

    def __compute_continuous_seg_sample_metrics__(self):
        """
        Computes metrics on the samples in the continuous segments
        """
        if(self.__st__.__nonzero__()):
            c_segments = []
            for t in self.__st__:
                seg = defaultdict()
                seg['start_time'] = t.stats.starttime.isoformat()
                seg['end_time'] = t.stats.endtime.isoformat()
                seg['sample_min'] = np.min(t.data)
                seg['sample_max'] = np.max(t.data)
                seg['sample_mean'] = np.mean(t.data)
                seg['sample_rms'] = np.sqrt(sum([np.square(n)
                                                 for n in t.data])/len(t.data))
                seg['sample_stdev'] = np.std(t.data)
                seg['num_samples'] = t.stats.npts
                seg['seg_len'] = t.stats.endtime - t.stats.starttime
                c_segments.append(seg)
            self.__msmeta__['c_segments'] = c_segments

    @property
    def msmeta(self):
        """
        Returns the msmeta dictionary previously populated
        :return: Dictionary with MSEED metadata
        """
        return self.__msmeta__

    def populate_metadata(self, files, starttime=None, endtime=None,
                          c_seg=True, **kwargs):
        """
        Reads the MSEED input, computes and extracts the
        metadata populating the msmeta dictionary
        :type str
        :param starttime
        :type str
        :param endtime
        :type list
        :param files: list containing the Mini-SEED files
        """
        stime = UTCDateTime(starttime) if starttime is not None\
            else start_time
        etime = UTCDateTime(endtime) if endtime is not None\
            else endtime
        self.__read__(files, stime, etime)
        self.__extract_mseed_stream_metadata__()
        self.__compute_sample_metrics__()
        if c_seg:
            self.__compute_continuous_seg_sample_metrics__()

    def get_json_meta(self):
        """
        Serializes the msmeta dictionary in json format
        :return: JSON containing the MSEED metadata
        """
        return json.dumps(self.__msmeta__, cls=NumPyEncoder)

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
=======

class MSEEDMetadata():
    """
    A container for MSEED specific metadata including QC

    """

    def __init__(self):
        self.msmeta=collections.OrderedDict()
        self.__samples__=[]

    def read(self,files,start_time,end_time):
        """
        Assumes that each file contains a single stream

        """

        for file in files:
            self.st += obspy.read(file)
        self.files=files
        self.start_time=start_time
        self.end_time=end_time
        #cut the Stream to the requested window
        self.st.trim(starttime=start_time,endtime=end_time)
        for t in self.st:
            self.__samples__.extend(t.data)

    def extractMSStreamMetadata(self):
        """
        Extracts metadata from the MSEED header and populates a dictionary

        """
        stats=self.st[0].stats
        self.msmeta['net']=stats.network
        self.msmeta['sta']=stats.station
        self.msmeta['cha']=stats.channel
        self.msmeta['loc']=stats.location
        self.msmeta['sample_rate']=stats.sampling_rate
        self.msmeta['record_len']=stats.mseed.record_length
        self.msmeta['quality']=stats.mseed.dataquality
        self.msmeta['encoding']=stats.mseed.encoding
        self.msmeta['num_records']=stats.mseed.number_of_records
        # TODO the method get flags and the related get_record_information should have starttime and endtime parameters
        # only in this way the computation can be consistent with users' request
        flags=get_flags(self.st)
        data_quality_flags=flags['data_quality_flags']
        activity_flags=flags['activity_flags']
        io_and_clock_flags=flags['io_and_clock_flags']
        self.msmeta['glitches']=data_quality_flags[3]
        self.msmeta['amplifier_saturation']=data_quality_flags[0]
        self.msmeta['digital_filter_charging']=data_quality_flags[6]
        self.msmeta['digitizer_clipping']=data_quality_flags[1]
        self.msmeta['missing_padded_data']=data_quality_flags[4]
        self.msmeta['spikes']=data_quality_flags[2]
        self.msmeta['suspect_time_tag']=data_quality_flags[7]
        self.msmeta['telemetry_sync_error']=data_quality_flags[5]
        self.msmeta['calibration_signal']=activity_flags[0]
        self.msmeta['event_begin']=activity_flags[2]
        self.msmeta['event_end']=activity_flags[3]
        self.msmeta['event_in_progress']=activity_flags[6]
        self.msmeta['timing_correction']=activity_flags[1]
        self.msmeta['clock_locked']=io_and_clock_flags[5]
        ##TODO
        #Daily average of the SEED timing quality stored in miniSEED blockette 1001.
        #self.msmeta['timing_quality']=
        self.msmeta['timing_quality_mean']=flags['timing_quality_average']
        self.msmeta['timing_quality_min']=flags['timing_quality_min']
        self.msmeta['timing_quality_max']=flags['timing_quality_max']
        #TODO
        #Flag indicating the existence of the actual sample rate in Blockette 100 being different from the sample rate as described by fields 10 and 11 in the Fixed Section of the Data Record Header
        #self.msmeta['sample_rate_dev']


    def computeSampleMetrics(self):
        """
        Computes metrics on the samples contained in the specified time window
        """
        gaps=self.st.getGaps()
        self.msmeta['sample_rms']=np.sqrt(sum([np.square(n) for n in self.__samples__])/len(self.__samples__))
        self.msmeta['sample_mean']=np.mean(self.__samples__)
        self.msmeta['sample_min']=np.min(self.__samples__)
        self.msmeta['sample_max']=np.max(self.__samples__)
        self.msmeta['sample_stdev']=np.std(self.__samples__)
        self.msmeta['num_gaps']=len([gap for gap in gaps if gap[7]>0])
        self.msmeta['num_overlaps']=len([gap for gap in gaps if gap[7]<0])
        self.msmeta['overlaps_len']=sum([gap[6] for gap in gaps if gap[6]<0])
        self.msmeta['gaps_len']= sum([gap[6] for gap in gaps if gap[6]>0])
        start_time=self.st[0].stats.starttime
        end_time=self.st[-1].stats.endtime
        self.msmeta['percent_availability']=100*((end_time-start_time-self.msmeta['gaps_len'])/(end_time-start_time)


>>>>>>> First commit mseed-qc branch
