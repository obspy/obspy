import ctypes as C
from math import log
import numpy as np

from obspy.core import Stream, Trace, UTCDateTime
from obspy.mseed.headers import clibmseed, SAMPLESIZES, HPTMODULUS


DATATYPES = {"a": C.c_char, "i": C.c_int32, "f": C.c_float, "d": C.c_double}
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


# Container for a continuous linked list of records.
class ContinuousSegment(C.Structure):
    pass
ContinuousSegment._fields_ = [
    ('starttime', C.c_longlong),
    ('endtime', C.c_longlong),
    ('samprate', C.c_double),
    ('sampletype', C.c_char),
    ('hpdelta', C.c_longlong),
    ('samplecnt', C.c_int),
    ('datasamples', C.c_void_p),      # Data samples, 'numsamples' of type 'sampletype'
    ('firstRecord', C.c_void_p),
    ('lastRecord', C.c_void_p),
    ('next', C.POINTER(ContinuousSegment)),
    ('previous', C.POINTER(ContinuousSegment))
    ]





# A container for continuous segments with the same id
class LinkedIDList(C.Structure):
    pass
LinkedIDList._fields_ = [
    ('network', C.c_char * 11),       # Network designation, NULL terminated
    ('station', C.c_char * 11),       # Station designation, NULL terminated
    ('location', C.c_char * 11),      # Location designation, NULL terminated
    ('channel', C.c_char * 11),       # Channel designation, NULL terminated
    ('dataquality', C.c_char),        # Data quality indicator
    ('firstSegment',  C.POINTER(ContinuousSegment)), # Pointer to first of list of segments
    ('lastSegment',  C.POINTER(ContinuousSegment)),  # Pointer to last of list of segments
    ('next',  C.POINTER(LinkedIDList)),              # Pointer to next id
    ('previous',  C.POINTER(LinkedIDList)),          # Pointer to previous id
    ('last',  C.POINTER(LinkedIDList))              # Pointer to the last id
    ]


#####################################
# Done with the C structures defintions.
#####################################

# Set the necessary arg- and restypes.
clibmseed.readMSEEDBuffer.argtypes = [
    np.ctypeslib.ndpointer(dtype='b', ndim=1, flags='C_CONTIGUOUS'),
    C.c_int,
    C.POINTER(Selections),
    C.c_int,
    C.c_int,
    C.c_int,
    C.CFUNCTYPE(C.c_long, C.c_int, C.c_char)
    ]

clibmseed.readMSEEDBuffer.restype = C.POINTER(LinkedIDList)

clibmseed.mstl_init.restype = C.POINTER(MSTraceList)
clibmseed.mstl_free.argtypes = [C.POINTER(C.POINTER(MSTraceList)), C.c_int]

clibmseed.lil_init.restype = C.POINTER(LinkedIDList)

clibmseed.lil_free.argtypes = [C.POINTER(LinkedIDList)]

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

    all_data = []
    # Use a callback function to allocate the memory and keep track of the
    # data.
    def allocate_data(samplecount, sampletype):
        data = np.empty(samplecount, dtype=DATATYPES[sampletype])
        all_data.append(data)
        return data.ctypes.data
    # Define Python callback function for use in C function. Return a long so
    # it hopefully works on 32 and 64 bit systems.
    allocData = C.CFUNCTYPE(C.c_long, C.c_int, C.c_char)(allocate_data)

    lil = clibmseed.readMSEEDBuffer(buffer, buflen, selections, unpack_data,
                              reclen, 0, allocData)

    # XXX: Check if the freeing works.
    del selections

    traces = []
    try:
        currentID = lil.contents
    # Return stream if not traces are found.
    except ValueError:
        return Stream()

    while True:
        # Init header with the essential information once.
        header = {'network': currentID.network,
                  'station': currentID.station,
                  'location': currentID.location,
                  'channel': currentID.channel,
                  'mseed': {'dataquality': currentID.dataquality}}
        # Loop over segments.
        currentSegment = currentID.firstSegment.contents
        while True:
            header['sampling_rate'] = currentSegment.samprate
            header['starttime'] = _convertMSTimeToDatetime(currentSegment.starttime)
            # The data always will be in sequential order.
            data = all_data.pop(0)
            traces.append(Trace(header=header, data=data))
            # A Null pointer access results in a ValueError
            try:
                currentSegment = currentSegment.next.contents
            except ValueError:
                break
        try:
            currentID = currentID.next.contents
        except ValueError:
            break

    clibmseed.lil_free(lil)
    del lil
    return Stream(traces=traces)
