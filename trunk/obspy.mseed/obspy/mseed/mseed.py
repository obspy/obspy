import ctypes as C
from math import log
import numpy as np

from obspy.core import Stream, Trace, UTCDateTime
from obspy.mseed.headers import clibmseed, SAMPLESIZES, HPTMODULUS


# XXX: Put all the definitions in the header file.
#####################################
# Define the C structures.
#####################################

# Container for a continuous trace segment, linkable
class MSTraceSeg(C.Structure):
    pass
MSTraceSeg._fields_ = [
    ('starttime', C.c_longlong),      # Time of first sample
    ('endtime', C.c_longlong),        # Time of last sample
    ('samprate', C.c_double),         # Nominal sample rate (Hz)
    ('samplecnt', C.c_int),           # Number of samples in trace coverage
    ('datasamples', C.c_void_p),      # Data samples, 'numsamples' of type 'sampletype'
    ('numsamples', C.c_int),          # Number of data samples in datasamples
    ('sampletype', C.c_char),         # Sample type code: a, i, f, d
    ('prvtptr', C.c_void_p),          # Private pointer for general use, unused by libmseed
    ('prev', C.POINTER(MSTraceSeg)),  # Pointer to previous segment
    ('next', C.POINTER(MSTraceSeg))   # Pointer to next segment
    ]

# Container for a trace ID, linkable
class MSTraceID(C.Structure):
    pass
MSTraceID._fields_ = [
    ('network', C.c_char * 11),       # Network designation, NULL terminated
    ('station', C.c_char * 11),       # Station designation, NULL terminated
    ('location', C.c_char * 11),      # Location designation, NULL terminated
    ('channel', C.c_char * 11),       # Channel designation, NULL terminated
    ('dataquality', C.c_char),        # Data quality indicator
    ('srcname', C.c_char * 45),       # Source name (Net_Sta_Loc_Chan_Qual), NULL terminated
    ('type', C.c_char),               # Trace type code
    ('earliest', C.c_longlong),       # Time of earliest sample
    ('latest', C.c_longlong),         # Time of latest sample
    ('prvtptr', C.c_void_p),          # Private pointer for general use, unused by libmseed
    ('numsegments', C.c_int),         # Number of segments for this ID
    ('first', C.POINTER(MSTraceSeg)), # Pointer to first of list of segments
    ('last', C.POINTER(MSTraceSeg)),  # Pointer to last of list of segments  
    ('next', C.POINTER(MSTraceID))    # Pointer to next trace
    ]

# Container for a continuous trace segment, linkable
class MSTraceList(C.Structure):
    pass
MSTraceList._fields_ = [
    ('numtraces', C.c_int),           # Number of traces in list
    ('traces', C.POINTER(MSTraceID)), # Pointer to list of traces
    ('last', C.POINTER(MSTraceID))    # Pointer to last used trace in list
    ]

# Data selection structure time window definition containers
class SelectTime(C.Structure):
    pass
SelectTime._fields_ = [
    ('starttime', C.c_longlong),      # Earliest data for matching channels
    ('endtime', C.c_longlong),        # Latest data for matching channels
    ('next', C.POINTER(SelectTime))
    ]

# Data selection structure definition containers
class Selections(C.Structure):
    pass
Selections._fields_ = [
    ('srcname', C.c_char * 100),      # Matching (globbing) source name: Net_Sta_Loc_Chan_Qual
    ('timewindows', C.POINTER(SelectTime)),
    ('next', C.POINTER(Selections))
    ]

#####################################
# Done with the C structures defintions.
#####################################

# Set the necessary arg- and restypes.
clibmseed.readMSEEDBuffer.argtypes = [
    C.POINTER(MSTraceList),
    np.ctypeslib.ndpointer(dtype='b', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int,
    C.POINTER(Selections),
    C.c_int,
    C.c_int,
    C.c_int
    ]

clibmseed.mstl_init.restype = C.POINTER(MSTraceList)
clibmseed.mstl_free.argtypes = [C.POINTER(C.POINTER(MSTraceList)), C.c_int]

HPTERROR = -2145916800000000L


def _ctypesArray2NumpyArray(buffer, buffer_elements, sampletype):
    """
    Takes a Ctypes array and its length and type and returns it as a
    NumPy array.
    
    :param buffer: Ctypes c_void_p pointer to buffer.
    :param buffer_elements: length of the whole buffer
    :param sampletype: type of sample, on of "a", "i", "f", "d"
    """
    # Allocate NumPy array to move memory to
    numpy_array = np.empty(buffer_elements, dtype=sampletype)
    datptr = numpy_array.ctypes.get_data()
    # Manually copy the contents of the C allocated memory area to
    # the address of the previously created NumPy array
    C.memmove(datptr, buffer, buffer_elements * SAMPLESIZES[sampletype])
    return numpy_array

def _convertMSTimeToDatetime(timestring):
    """
    Takes Mini-SEED timestamp and returns a obspy.util.UTCDateTime object.
    
    :param timestamp: Mini-SEED timestring (Epoch time string in ms).
    """
    return UTCDateTime(timestring / HPTMODULUS)

def _convertDatetimeToMSTime(dt):
    """
    Takes obspy.util.UTCDateTime object and returns an epoch time in ms.
    
    :param dt: obspy.util.UTCDateTime object.
    """
    return int(dt.timestamp * HPTMODULUS)


def readMSEED(mseed_object, selection=None, unpack_data=True, reclen=None):
    """
    Takes a file like object that contains binary MiniSEED data and returns an
    obspy.core.Stream object.

    :param mseed_object: Filename or open file like object that contains the
        binary MiniSEED data. Any object that provides a read() method will be
        considered to be a file like object.
    :param selection: If given only parts of the MiniSEED data will be used. It
        is a dictionary with the following structure:
                
            selection = {'starttime': UTCDateTime(...),
                         'endtime': UTCDateTime(...),
                         'sourcename': '*EHZ'}
        selection['sourcename'] has to have the structure
        'network.station.location.channel' and can contain globbing characters.
        Defaults to None
    :param unpack_data: Determines whether or not to unpack the data or just
        read the headers.
    :param reclen: If it is None, it will be automatically determined for every
        record. If it is known, just set it to the record length in bytes.

    Example usage
    =============
    The following example will read all EHZ channels from the BW network from
    the binary data in mseed_data. Only the first hour of 2010 will be read.

    >> from cStringIO import StringIO
    >> f = StringIO(mseed_data)
    >> selection = {'starttime': UTCDateTime(2010, 1, 1, 0, 0, 0),
                    'endtime': UTCDateTime(2010, 1, 1, 1, 0, 0),
                    'sourcename': 'BW.*.*.EHZ'}
    >> st = readMSEED(f, selection)
    """
    # Parse the unpack_data and reclen flags.
    if unpack_data:
        unpack_data = 1
    else:
        unpack_data = 0
    if not reclen:
        reclen = -1
    else:
        reclen = int(log(reclen, 2))

    # If its a filename just read it.
    if type(mseed_object) is str:
        # Read to numpy array which is used as a buffer.
        buffer = np.fromfile(mseed_object, dtype='b')
    elif hasattr(mseed_object, 'read'):
        buffer = np.fromstring(mseed_object.read(), dtype='b')

    buflen = len(buffer)

    # Create the MSTraceList structure and read the buffer to the structure.
    mstl = clibmseed.mstl_init(None)
    # If no selection is given pass None to the C function.
    if not selection:
        selections = None
    else:
        select_time = SelectTime()
        selections = Selections()
        selections.timewindows.contents = select_time
        if 'starttime' in selection and \
                type(selection['starttime']) == UTCDateTime:
            selections.timewindows.contents.starttime = \
                    _convertDatetimeToMSTime(selection['starttime'])
        else: 
            selections.timewindows.contents.starttime = HPTERROR
        if 'endtime' in selection and \
                type(selection['endtime']) == UTCDateTime:
            selections.timewindows.contents.endtime = \
                    _convertDatetimeToMSTime(selection['endtime'])
        else: 
            selections.timewindows.contents.endtime = HPTERROR
        if 'sourcename' in selection:
            # libmseed uses underscores as seperators and allows filtering
            # after the dataquality which is disabled here to not confuse
            # users.
            selections.srcname = selection['sourcename'].replace('.', '_') + '_D'
        else:
            selections.srcname = '*'

    clibmseed.readMSEEDBuffer(mstl, buffer, buflen, selections, unpack_data,
                              reclen, 0)

    traces = []
    # Loop over all traces.
    trace_count = mstl.contents.numtraces
    # Return stream if not traces are found.
    if not trace_count:
        return Stream()
    this_trace = mstl.contents.traces.contents
    for _i in xrange(trace_count):
        # Init header with the essential information once.
        header = {'network': this_trace.network,
                  'station': this_trace.station,
                  'location': this_trace.location,
                  'channel': this_trace.channel,
                  'mseed': {'dataquality': this_trace.dataquality}}
        # Loop over all segments. Every segment will be a new trace.
        this_segment = this_trace.first.contents
        for _j in xrange(this_trace.numsegments):
            header['sampling_rate'] = this_segment.samprate
            header['starttime'] = _convertMSTimeToDatetime(this_segment.starttime)
            data = _ctypesArray2NumpyArray(this_segment.datasamples,
                                           this_segment.numsamples,
                                           this_segment.sampletype)
            traces.append(Trace(header=header, data=data))
            # A Null pointer access results in a ValueError
            try:
                this_segment = this_segment.next.contents
            except ValueError:
                break
        try:
            this_trace = this_trace.next.contents
        except ValueError:
            break
    clibmseed.mstl_free(C.pointer(mstl), C.c_int(1))
    return Stream(traces=traces)
