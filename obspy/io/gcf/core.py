#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCF bindings to python
"""
from __future__ import print_function
import sys, os, math
import ctypes  # NOQA
import numpy as np
from obspy import UTCDateTime
from obspy.core import Stream, Trace, AttribDict
from obspy.core.util.libnames import _load_cdll

### supported sampling rates outside range 1-250 Hz
### key:value = samplingrate:fractional_time_denominator
_SPS_MAP = {
   0.1:1,
   0.125:1,
   0.2:1,
   0.25:1,
   0.5:1,
   400:8,
   500:2,
   625:5,
   800:16,
   1000:4,
   1250:5,
   2000:8,
   2500:10,
   4000:16,
   5000:20
}
### Unsupported sampling rates in range 1-250 Hz
_SPS_RESERVED = [157, 161, 162, 164, 167, 171, 174, 175, 176, 179, 181, 182, 191, 193, 194]
### Valid differential gains
_VALID_GAIN = [-1, 0, 1, 2, 4, 8, 16, 32, 64]


class _GcfSeg(ctypes.Structure):
   _fields_ = [
      ('streamID'     , ctypes.c_char * 7),
      ('systemID'     , ctypes.c_char * 7),
      ('start'        , ctypes.c_int32),
      ('t_numerator'  , ctypes.c_int),
      ('t_denominator', ctypes.c_int),
      ('t_leap'       , ctypes.c_int),
      ('gain'         , ctypes.c_int),
      ('sysType'      , ctypes.c_int),
      ('digi'         , ctypes.c_int),
      ('ttl'          , ctypes.c_int),
      ('blk'          , ctypes.c_int),
      ('err'          , ctypes.c_int),
      ('sps'          , ctypes.c_int),
      ('sps_denom'    , ctypes.c_int),
      ('compr'        , ctypes.c_int),
      ('FIC'          , ctypes.c_int32),
      ('RIC'          , ctypes.c_int32),
      ('n_data'       , ctypes.c_int32),
      ('n_alloc'      , ctypes.c_int32),
      ('data'         , ctypes.POINTER(ctypes.c_int32))
   ]


class _GcfFile(ctypes.Structure):
   _fields_ = [
      ('n_blk',     ctypes.c_int),
      ('n_seg',     ctypes.c_int),
      ('n_alloc',   ctypes.c_int),
      ('n_errHead', ctypes.c_int),
      ('n_errData', ctypes.c_int),
      ('seg',       ctypes.POINTER(_GcfSeg))      
   ]


def compatible_sps(sps):
   """
   Checks if a sampling rate is compatible with the GCF format
   
   :type sps: float, int
   :param sps: sampling rate to test
   :rtype: bool
   :returns: True if sampling rate is compatible else False
   """
   is_compat = True
   try:
      if sps >= 1 and sps <= 250:
         if int(sps)-sps != 0 or sps in _SPS_RESERVED:
            is_compat = False
      elif sps not in _SPS_MAP:
         is_compat = False
   except:
      is_compat = False
   return is_compat


def get_time_denominator(sps):
   """
   Returns the time fractional offset denominator, d, associated with an input sampling rate. 
   
   Any data written to a gcf file must have its first sample sampled at a fractional time: 
     n/d 
   where 0 <= n < d
   
   :type sps: float, int
   :param sps: sampling rate (samples per second)
   :rtype: int
   :returns: fractional offset denominator, if 0 input sampling rate is not compatible with
      implemented GCF format, if 1 first data sample must be sampled in integer time
   """
   denom = 0
   if compatible_sps(sps):
      if sps <= 250:
         denom = 1
      else:
         denom = _SPS_MAP[sps]
   return denom


def merge_gcf_stream(st):
    """
    Merges GCF stream (replacing Stream.merge(-1) for headonly=True)

    :type st: :class:`~obspy.core.stream.Stream`
    :param st: GCF Stream object with no data
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.
    """
    traces = []
    for tr in st:
        delta = tr.stats.delta
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
        for trace in traces:
            if tr.id == trace.id and delta == trace.stats.delta \
               and not starttime == trace.stats.starttime:
                if 0 < starttime - trace.stats.endtime <= delta:
                    trace.stats.npts += tr.stats.npts
                    break
                elif 0 < trace.stats.starttime - endtime <= delta:
                    trace.stats.starttime = UTCDateTime(starttime)
                    trace.stats.npts += tr.stats.npts
                    break
        else:
            traces.append(tr)
    return Stream(traces=traces) 


def _is_gcf(filename):
   """
   Checks whether a file is GCF or not.

   :type filename: str
   :param filename: path to GCF file to be checked.
   :rtype: bool
   :return: True if a object pointed to by path is a GCF file.
   """
   is_gcf = True
   if not os.path.isfile(filename) or os.path.getsize(filename)%1024:
      # File either does not point at a file object or file is not of proper size
      is_gcf = False
   else:
      # Load shared library
      try:
         gcf_io = _load_cdll("gcf")
      except Exception as e:
         print(str(e))
      else:
         # declare function argument and return types
         gcf_io.read_gcf.argtypes = [ctypes.c_char_p, ctypes.POINTER(_GcfFile), ctypes.c_int]
         gcf_io.read_gcf.restype = ctypes.c_int
         gcf_io.free_GcfFile.argtypes = [ctypes.POINTER(_GcfFile)]
         gcf_io.free_GcfFile.restype = None

         # Decode first block
         obj = _GcfFile()
         b_filename = filename.encode('utf-8')
         ret = gcf_io.read_gcf(b_filename, obj, 3)
         if ret or (obj.n_errHead and obj.seg[0].err not in (10, 11, 21)) or obj.n_errData: 
            is_gcf = False
            
         # release allocated memory
         gcf_io.free_GcfFile(obj)
   return is_gcf


def _read_gcf(filename, networkcode='', stationcode='', locationcode='', bandcode="H", instrumentcode="H", channel_prefix=None, 
              blockmerge=True, headonly=False, cleanoverlap=True, errorret=False, **kwargs):
   """
   Reads a GCF file and returns a :class:`~obspy.core.stream.Stream`
   
   Only GCF data records are supported. Function supports format as described by 
   GCF Reference `SWA-RFC-GCFR Issue F, December 2021 <https://www.guralp.com/apps/ok?doc=GCF_format>`
      
   .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.
   
   :type filename: str
   :param filename: path to GCF file to read
   :type networkcode: str, optional
   :param networkcode: network code to use 
   :type stationcode: str, optional
   :param stationcode: station code to use, if not specified unitID in gcf header will be used (first 4
   :   characters in streamID)
   :type locationcode: str, optional
   :param locationcode: location code to use
   :type bandcode: str, optional
   :param bandcode: 1-character band code to use, ignored if channel_prefix is input
   :type instrumentcode: str, optional
   :param instrumentcode: 1-character instrument code to use, ignored if channel_prefix is input
   :type channel_prefix: str, optional
   :param channel_prefix: 2-character channel prefix to use, if not input channel_prefix = bandcode+instrumentcode
   :type blockmerge: bool, optional
   :param blockmerge: if True merge blocks if aligned
   :type headonly: bool, optional
   :param headonly: if True only read block headers
   :type cleanoverlap: bool, optional
   :param cleanoverlap: ignored if blockmerge is False, if True, remove overlapping identical data prior to merging blocks, 
       if False overlapping blocks will not be merged even if data in overlap is identical
   :type errorret: bool, optional
   :param errorret: if True block and header issues will be set in trace.stats.gcf.stat for each :class:`~obspy.core.trace.Trace`
       object in returned :class:`~obspy.core.stream.Stream` object, else function will raise raise an IOError with an appropriate
       error message
   :rtype: :class:`~obspy.core.stream.Stream`
   :returns: Stream object
   
   ..rubric:: Exceptions
   
   Function will raise :class:IOError on problems to read file
   
   .. rubric:: GCF header data
   
   GCF specific meta data will be set for each :class:`~obspy.core.trace.Trace` object in the returned :class:`~obspy.core.stream.Stream`
    object under attribute ``stats.gcf``, more precisely the following attributes will be set:
    
     ``stats.gcf.systemID`` : str
         systemID set in block header
     ``stats.gcf.streamID`` : str
         6-character streamID set in block, typically consists of: (4-character unitID) + (1-character orientation code) + (int, tap)
     ``stats.gcf.sysType`` : int
         systemID type: ``0`` - regular, ``1`` - extended, ``2`` - double extended
     ``stats.gcf.t_leap`` : bool
         1 if block start at leap second, else 0
     ``stats.gcf.gain`` : int
         variable gain setting (if ``-1`` not used)
     ``stats.gcf.digi`` : int
         digitizer type, combine with ``stats.gcf.sysType`` to find the digitizer according to:
          =================  ==============  ===========================
          stats.gcf.sysType  stats.gcf.digi  digitizer
          =================  ==============  ===========================
          0                  0               unknown (probably DM24 Mk2)
          1                  0               DM24
          1                  1               CD24
          2                  0               Affinity
          2                  1               Minimus
          =================  ==============  ===========================
     ``stats.gcf.ttl`` : int
         tap-table-lookup to retrieve sequence of decimation filters used (see 
         `GCF reference<https://www.guralp.com/apps/ok?doc=GCF_format>` 
         for further info)
     ``stats.gcf.blk`` : int
         Lowest block number in gcf file of blocks included in trace (count start at 0)
     ``stats.gcf.FIC`` : int 
         Forward Integration Constant (i.e. first data value)
     ``stats.gcf.RIC`` : int 
         Reverse Integration Constant (ideally last data value)
     ``stats.gcf.stat`` : int
         Status code, available codes are:
         ==== ==========================================================================
         code description
         ==== ==========================================================================
         -1   not a data block
         0    no detected block issues
         1    stream ID use extended system ID format
         2    stream ID use double-extended system ID format
         3    unknown compression
         4    to few/many data samples indicated in block header
         5    start of first data sample is negative
         9    failure to decode header values
         10   failure to decode data values (last data != RIC, may be due to bad RIC!!!)
         11   first first difference != 0 (may not be critical!!!)
         21   status codes 10 + 11
         ==== ==========================================================================
   
   ..rubric:: Example

    >>> from obspy import read
    >>> st = read("/path/to/20160603_1955n.gcf", format="GCF")
   """
   if not os.path.exists(filename):
      raise IOError("file %s could not be located (erroneous path?)" % (filename))
   
   stream = None
   # Load shared library
   try:
      gcf_io = _load_cdll("gcf")
   except Exception as e:
      raise IOError(str(e))
   else:
      # declare function argument and return types
      gcf_io.read_gcf.argtypes = [ctypes.c_char_p, ctypes.POINTER(_GcfFile), ctypes.c_int]
      gcf_io.read_gcf.restype = ctypes.c_int
      gcf_io.free_GcfFile.argtypes = [ctypes.POINTER(_GcfFile)]
      gcf_io.free_GcfFile.restype = None
      # set reader mode
      if headonly:
         if not blockmerge:
            mode = -2
         else:
            mode = -1
      else:
         if not blockmerge:
            mode = 2
         elif not cleanoverlap:
            mode = 1
         else:
            mode = 0

      # read file
      obj = _GcfFile()
      b_filename = filename.encode('utf-8')
      ret = gcf_io.read_gcf(b_filename, obj, mode)
      if ret == -1:
         if os.path.isfile(filename):
            raise IOError("cannot open file %s in read mode (missing read permissions for user?)" % (filename))
         else:
            raise IOError("%s is not a file" % (filename))
      elif ret == 1:
         raise IOError("file %s is not a GCF data file" % (filename))
      elif ret:
         raise IOError("failed to read file %s (unknown error code %d returned)" % (filename,ret))
      else:
         err_msg = ""
         # So far so good, set up trace objects
         traces = []
         for i in range(int(obj.n_seg)):
            if obj.seg[i].err and not errorret:
               if obj.seg[i].err == -1:
                  err_msg = "data block %d is not a data block" % (i+1)
               elif obj.seg[i].err == 1:
                  err_msg = "stream ID in header of data block %d use extended system ID format" % (i+1)
               elif obj.seg[i].err == 2:
                  err_msg = "stream ID in header ofdata block %d use double-extended system ID format" % (i+1)
               elif obj.seg[i].err == 3:
                  err_msg = "unknown compression in data block %d" % (i+1)
               elif obj.seg[i].err == 4:
                  err_msg = "to few/many data samples indicated in block header of data block %d" % (i+1)
               elif obj.seg[i].err == 5:
                  err_msg = "start time of first data sample in data block %d is negative" % (i+1)
               elif obj.seg[i].err == 9:
                  err_msg = "failure to decode header values of data block %d" % (i+1)
               elif obj.seg[i].err == 10:
                  err_msg = "failure to decode data block %d (last data != RIC)" % (i+1)
               elif obj.seg[i].err == 11:
                  err_msg = "1'st first difference != 0 in data block %d" % (i+1)
               elif obj.seg[i].err == 21:
                  err_msg = "failure to decode data block %d (last data != RIC, 1'st first difference != 0)" % (i+1)
               else:
                  err_msg = "unknown error code (%d) set for data block %d" % (obj.seg[i].err,i+1)
               break
            channel =((channel_prefix if channel_prefix else bandcode+instrumentcode)+obj.seg[i].streamID.decode('utf-8')[-2]).upper()
            if headonly or obj.seg[i].n_data <= 0:
               data = None
            else:
               data = np.array(obj.seg[i].data[:obj.seg[i].n_data], dtype=np.int32)
            sps = obj.seg[i].sps*1./obj.seg[i].sps_denom
            dt = obj.seg[i].t_leap
            start = obj.seg[i].start+obj.seg[i].t_numerator*1./obj.seg[i].t_denominator-dt
            stats = {
               "network":networkcode,
               "station":stationcode if stationcode else obj.seg[i].streamID[:4].decode('utf-8'),
               "channel":channel,
               "sampling_rate":sps,
               "starttime":UTCDateTime(start),
               "npts":obj.seg[i].n_data,
               "gcf":AttribDict({
                  "systemID":obj.seg[i].systemID.decode('utf-8'),
                  "streamID":obj.seg[i].streamID.decode('utf-8'),
                  "sysType":obj.seg[i].sysType,
                  "t_leap":True if obj.seg[i].t_leap else False,
                  "gain":obj.seg[i].gain,
                  "digi":obj.seg[i].digi,
                  "ttl":obj.seg[i].ttl,
                  "blk":obj.seg[i].blk,
                  "FIC":obj.seg[i].FIC,
                  "RIC":obj.seg[i].RIC,
                  "stat":obj.seg[i].err
               })
            }
            if data is not None:
               traces.append(Trace(data=data,header=stats))
            else:
               traces.append(Trace(header=stats))

         # Set up the stream object
         if not err_msg:
            stream = Stream(traces=traces)

         # Free memory allocated by the C-function
         gcf_io.free_GcfFile(obj)
         
         if err_msg:
            raise IOError(err_msg)
   return stream



def _write_gcf(stream, filename, streamID=None, systemID=None, isLeap=False, gain=None, ttl=None, digi=None, sysType=None, missalign=0.1, **kwargs):
   """   
   Writes a :class:`~obspy.core.stream.Stream` or a :class:`~obspy.core.trace.Trace` to a GCF file
   
   Only GCF data records are supported. Function supports format as described by 
   GCF Reference `SWA-RFC-GCFR Issue F, December 2021 <https://www.guralp.com/apps/ok?doc=GCF_format>`
     
   .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.write` function, call this instead.
   
   :type stream: :class:`~obspy.core.stream.Stream` or :class:`~obspy.core.trace.Trace`
   :param stream: waveform to write to file, 
   :type filename: str
   :param filename: path to GCF file to read
   :type streamID: str
   :param streamID: 6-character stream ID, first 4 characters (the unit ID) should be in the range 0-9, A-Z
      fifth character should be the orientation code (e.g. Z, N, E) and the last character an integer yielding
      the tap the data were output from in the digitizer. If not specified streamID will be taken from
      ``stats.gcf.streamID`` for each :class:`~obspy.core.trace.Trace` object in the input :class:`~obspy.core.stream.Stream`
      object if present else streamID will be constructed as:
       ``stats.station.ljust(4,x)[:4]+stats.channel[-1]+'0'``
   :type systemID: str
   :param systemID: 4- to 6-character SysID (4 if ``sysType`` is 2, 5 if ``sysType`` is 1 and
      6 if ``sysType`` is 0). If not specified systemID will be taken from ``stats.gcf.systemID``
      for each :class:`~obspy.core.trace.Trace` object in the input :class:`~obspy.core.stream.Stream` 
      object and if present else systemID will be extracted from ``streamID`` starting from first character.
   :type isLeap: int
   :param isLeap: should be set to ``True`` if first sample is sampled at an integer second that is a 
      leap second. If not specified value will be taken from ``stats.gcf.t_leap`` for each 
      :class:`~obspy.core.trace.Trace` object in the input :class:`~obspy.core.stream.Stream` if 
      present else the default value is ``0``
   :type gain: int
   :param gain: digitizer gain (permitted values are 2^n; n <= 6, and -1 if combined with ``sysType`` = 0), 
      if not specified gain will be taken from ``stats.gcf.gain`` for each :class:`~obspy.core.trace.Trace` 
      object in the input  :class:`~obspy.core.stream.Stream` object if present else the default value is ``0``.      
   :type ttl: int
   :param ttl: tap-table-lookup reference. if not specified ttl will be taken from t``stats.gcf.gain`` for each 
      :class:`~obspy.core.trace.Trace` object in the input :class:`~obspy.core.stream.Stream` object if 
      present else the default value is ``0``
   :type digi: int
   :param digi: digitizer type (permitted values are 0 or 1), if not specified type will be taken from 
       ``stats.gcf.digi`` for each :class:`~obspy.core.trace.Trace` object in the input 
       :class:`~obspy.core.stream.Stream` object if present else the default value is ``0``
   :type sysType: int
   :param sysType: systemID type (permitted values are 0, 1, or 2), if not specified sysType will be taken 
      from ``stats.gcf.sysType`` for each :class:`~obspy.core.trace.Trace` object in the input 
      :class:`~obspy.core.stream.Stream` object if present else the default value is ``0``
   :type missalign: float
   :param missalign: fraction of a sampling interval (permitted range is 0-0.5) of tolerated missalignment of 
      starttime. If not specified default value is ``0.1``
   
   ..Note::
   
   Sampling rate is restricted, most sampling rates  between 1-250 samples per second
   (minus a few exceptions)are supported, for greater or lower sampling rates format 
   support can be checked  with function :func:compatible_sps. 
   
   First data sample in each trace may only be sampled at non-integer second if 
   sampling rate > 250. For sampling rates > 250 first data sample in each trace must 
   start at an integer nominator of the denominator associated with the ampling rate. 
   e.g. for a sampling rate of 1000 samples per second the associated denominator 
   is 4 hence first data sample must be sampled at either ss.00 (0/4), ss.25 (1/4), 
   ss.50 (2/4), or ss.75 (3/4). Use function :func:get_time_denominator to get the 
   associated denominator.

   The GCF format is only guaranteed to support 32-bit signed integer values. While data with values
   out of range may be properly stored in the GCF format (if first and last data sample can be represented
   as a 32-bit signed integer as well as all first difference values of the dat avector) the current 
   implementaion only permitts input data to be representable as a 32-bit signed integer. If input waveforms
   cannot be representable as 32-bit signed integers they will silently be clipped at -2,147,483,648 and 
   2,147,483,647
     
   ..rubric:: Exceptions
   
   :class:TypeError will be raised upon unsupported input
   :class:IOError will be raised upon failure to output file
   
   ..rubric:: Example
   
   >>> st.write('GCF-filename.gcf', format='GCF') #doctest: +SKIP
   
   """
   # Make sure we have a Stream or Trace object
   if not isinstance(stream,Stream):
      if not isinstance(stream,Trace):
         raise TypeError("write requires either a Stream or a Trace object")
      else:
         stream = Stream(traces=stream)
         
   # Check output path
   if os.path.exists(filename) and not os.path.isfile(filename):
      IOError("path %s exists but is not a file object" % (filename))
   d,f = os.path.split(filename)
   if d and not os.path.exists(d):
      try:
         os.makedirs(d)
      except:
         raise IOError("failed to create output directorie(s) %s" % (d))
         
   # check missalign parameter for supported range
   if missalign < 0 or missalign > 0.5:
      raise TypeError("argument missalign out of range (permitted range 0 - 0.5)")
   
   ret = 0
   # Load shared library
   try:
      gcf_io = _load_cdll("gcf")
   except Exception as e:
      raise IOError(str(e))
   else:
      # declare function argument and return types
      gcf_io.write_gcf.argtypes = [ctypes.c_char_p, ctypes.POINTER(_GcfFile)]
      gcf_io.write_gcf.restype = ctypes.c_int

      # prepare the GcfFile object
      obj         = _GcfFile()
      obj.n_seg   = len(stream)
      obj.seg     = (_GcfSeg * obj.n_seg)()
      obj.n_alloc = obj.n_seg
      ctypes.cast(obj.seg, ctypes.POINTER(_GcfSeg))

      for i,trace in enumerate(stream):
         gcf_stats = trace.stats.gcf if hasattr(trace.stats,'gcf') else None
         
         ### Input checks, all of these are checked in the C-writer function
         ###  but still duplicated here for clarity and ease of formatting
         ###  proper exception messages

         ### Check data type
         if not isinstance(trace.data[0],np.int32):
            data = []
            for d in trace.data:
                if d < -2147483648:
                    d = -2147483648
                elif d > 2147483647:
                    d = 2147483647
                data.append(d)
            trace.data = np.array(data, dtype=np.int32)
         trace.data = np.ascontiguousarray(trace.data)

         ### Check sampling rate
         sps_denom = 1
         if not compatible_sps(trace.stats.sampling_rate):
            sps = ".3f" % (trace.stats.sampling_rate) if trace.stats.sampling_rate < 1 else "%d" % (trace.stats.sampling_rate)
            raise TypeError("trace sampling rate, %s, in trace %d not supported in GCF format" % (sps,i+1))
         else:
            sps = trace.stats.sampling_rate
            if sps < 1:
               if sps < 0.12:
                  sps_denom = 10
               elif sps < 0.19:
                  sps_denom = 8
               elif sps < 0.24:
                  sps_denom = 5
               elif sps < 0.49:
                  sps_denom = 4
               else:
                  sps_denom = 2
               sps = 1
            sps = int(sps)

         ### check start time
         sec = trace.stats.starttime.timestamp
         t_numerator = 0 
         if sps <= 250:
            # fractional start time not supported, allow for rounding of up 10% of the sampling interval
            # hmm is this enough for sampling rates < 1 Hz
            start = int(round(sec))
            if abs(start-sec) > 1./sps*missalign:
               raise TypeError("fractional start time not supported for sampling rates <= 250 Hz (in trace %d)" % (i+1))
         else:
            # fractional start time supported but restricted, allow for 10% missalignment
            start = int(math.floor(sec))
            dt = sec-start
            t_denom = _SPS_MAP[sps]
            numerator = dt*t_denom
            t_numerator = int(round(numerator))
            if abs(numerator-t_numerator) > missalign/t_denom:
               raise TypeError("start time in trace %d not aligned with supported (fractional) start time" % (i+1))

         ### Check if isLeap is set else set
         use_isLeap = 0
         if isLeap is not None:
            use_isLeap = 1 if isLeap else 0
         elif hasattr(gcf_stats,"isLeap"):
            use_isLeap = 1 if gcf.isLeap else 0

         ### Check if gain is set else set
         use_gain = 0
         gain_in_stats = False
         if gain is not None:
            use_gain = gain
         elif hasattr(gcf_stats,"gain"):
            gain_in_stats = True
            use_gain = gcf_stats.gain
         if use_gain not in  _VALID_GAIN:
            if gain is None:
               raise TypeError("bad value on sys.stats.gain, %s, in stats.gcf in trace %d (permitted values: %s)" % (use_gain,i+1,', '.join(["%d" % g for g in _VALID_GAIN])))
            else:
               raise TypeError("bad value on argument gain, %s (permitted values: %s)" % (use_gain,', '.join(["%d" % g for g in _VALID_GAIN])))

         ### Check if ttl is set else set
         use_ttl = 0
         if ttl is not None:
            use_ttl = ttl
         elif hasattr(gcf_stats,"ttl"):
            use_ttl = gcf_stats.ttl            

         ### Check if type is set else set
         use_digi = 0
         if digi is not None:
            use_digi = digi
         elif hasattr(gcf_stats,"digi"):
            use_digi = gcf_stats.digi   
         if use_digi not in [0, 1]:
            if digi is None:
               raise TypeError("bad value on stats.gcf.digi, %s, in trace %d (permitted values: 0, 1)" % (use_digi,i+1))
            else:
               raise TypeError("bad value on argument digi, %s (permitted values: 0, 1)" % (use_digi))

         ### Check if sysType is set else set
         use_sysType = 0
         st_in_stats = False
         if sysType is not None:
            use_sysType = sysType
         elif hasattr(gcf_stats,"sysType"):
            st_in_stats = True
            use_sysType = gcf_stats.sysType    
         if use_sysType not in [0, 1, 2]:
            if digi is None:
               raise TypeError("bad value on stats.gcf.digi, %s, in trace %d (permitted values: 0, 1, 2)" % (use_sysType,i+1))
            else:
               raise TypeError("bad value on argument digi, %s (permitted values: 0, 1, 2)" % (use_sysType,i+1))
         elif use_sysType != 0 and use_gain == -1:
            if gain_in_stats and st_in_stats:
               raise TypeError("value -1 on stats.gcf.gain may only be combined with value 0 on stats.gcf.sysType (in trace %s)" % (i+1))
            elif gain_in_stats:
               raise TypeError("value -1 on stats.gcf.gain may only be combined with value 0 on argument sysType (in trace %s)" % (i+1))
            elif st_in_stats:
               raise TypeError("value -1 on argument gain may only be combined with value 0 on stats.gcf.sysType (in trace %s)" % (i+1))
            else:
               raise TypeError("value -1 on argument gain may only be combined with value 0 on argument sysType")

         ### Check if streamID is set else build
         if streamID is not None:
            use_streamID = streamID
         elif gcf_stats is not None and hasattr(gcf_stats,"streamID"):
            use_streamID = gcf_stats.streamID.upper()
         else:
            use_streamID = (trace.stats.station.ljust(4,"X")[:4]+(trace.stats.channel[-1] if trace.stats.channel else 'X')+'0').upper()
         if len(use_streamID) != 6:
            if streamID is None:
               raise TypeError("bad value on stats.gcf.streamID, %s, in trace %d (must be 6-character long)" % (use_streamID,i+1))
            else:
               raise TypeError("bad value on argument streamID, %s (must be 6-character long)" % (use_streamID))

         ### Check if systemID is set else set
         len_systemID = 6 if use_sysType == 0 else (5 if use_sysType == 1 else 4)
         use_systemID = use_streamID[:len_systemID]
         si_in_stats = False
         if systemID is not None:
            use_systemID = systemID.upper() 
         elif gcf_stats is not None and hasattr(gcf_stats,"systemID"):
            si_in_stats = True
            use_systemID = gcf_stats.systemID.upper()
         if len(use_systemID) > len_systemID:
            if st_in_stats and si_in_stats:
               raise TypeError("stats.gcf.systemID not compatible with stats.gcf.sysType in trace %d (max length %d characters )" % (i+1,len_systemID))
            elif st_in_stats:
               raise TypeError("argument systemID not compatible with stats.gcf.sysType in trace %d (max length %d characters )" % (i+1,len_systemID))
            elif si_in_stats:
               raise TypeError("stats.gcf.systemID not compatible with argument sysType in trace %d (max length %d characters )" % (i+1,len_systemID))
            else:
               raise TypeError("argument systemID not compatible with argument sysType (max length %d characters )" % (len_systemID))

         ### Populate segment
         tr = _GcfSeg()
         tr.streamID    = use_streamID.encode('utf-8')
         tr.systemID    = use_systemID.encode('utf-8')
         tr.start       = start
         tr.t_numerator = t_numerator
         tr.t_leap      = use_isLeap  
         tr.gain        = use_gain    
         tr.sysType     = use_sysType 
         tr.digi        = use_digi    
         tr.ttl         = use_ttl     
         tr.sps         = sps
         tr.sps_denom   = sps_denom 
         tr.n_data      = trace.data.size
         tr.n_alloc     = trace.data.size
         tr.data        = trace.data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

         ### Add segment to file object
         obj.seg[i] = tr

      ### Write to file
      ret = gcf_io.write_gcf(filename.encode('utf-8'), obj)
      if ret == 0:
         pass
      elif ret == -2:
         raise IOError("failed to write to disc")
      elif ret == -1:
         raise IOError("failed to open file to write to")
      elif ret == 1:
         raise TypeError("no data or inconsistent headers")
      elif ret == 2:
         raise TypeError("unsupported sampling rate")
      elif ret == 3:
         raise TypeError("bad fractional start time")
      elif ret == 4:
         raise TypeError("unsupported gain")
      elif ret == 5:
         raise TypeError("unsupported instrument type")
      elif ret == 6:
         raise TypeError("to long systemID")
      else:
         raise IOError("unknown error code %d" % (ret))
