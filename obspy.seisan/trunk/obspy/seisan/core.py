# -*- coding: utf-8 -*-

from StringIO import StringIO
import array, struct
import os


def isSEISAN(filename):
    """
    Checks whether a file is SEISAN or not. Returns True or False.
    
    From the SEISAN documentation:
    When Fortran writes a files opened with "form=unformatted", additional 
    data is added to the file to serve as record separators which have to be 
    taken into account if the file is read from a C-program or if read binary 
    from a Fortran program. Unfortunately, the number of and meaning of these 
    additional characters are compiler dependent. On Sun, Linux, MaxOSX and PC 
    from version 7.0 (using Digital Fortran), every write is preceded and 
    terminated with 4 additional bytes giving the number of bytes in the write.
    On the PC, Seisan version 6.0 and earlier using Microsoft Fortran, the 
    first 2 bytes in the file are the ASCII character "KP". Every write is 
    preceded and terminated with one byte giving the number of bytes in the 
    write. If the write contains more than 128 bytes, it is blocked in records 
    of 128 bytes, each with the start and end byte which in this case is the 
    number 128. Each record is thus 130 bytes long. All of these additional 
    bytes are transparent to the user if the file is read as an unformatted 
    file. However, since the structure is different on Sun, Linux, MacOSX and 
    PC, a file written as unformatted on Sun, Linux or MacOSX cannot be read 
    as unformatted on PC or vice versa.
    The files are very easy to write and read on the same computer but 
    difficult to read if written on a different computer. To further 
    complicate matters, the byte order is different on Sun and PC. With 64 bit 
    systems, 8 bytes is used to define number of bytes written. This type of 
    file can also be read with SEISAN, but so far only data written on Linux 
    have been tested for reading on all systems.
    From version 7.0,the Linux and PC file structures are exactly the same. 
    On Sun the structure is the same except that the bytes are swapped. This 
    is used by SEISAN to find out where the file was written. Since there is 
    always 80 characters in the first write, character one in the Linux and PC 
    file will be the character P (which is represented by 80) while on Sun 
    character 4 is P.
    
    
    @param filename: SEISAN file to be read.
    """
    try:
        f = open(filename, 'rb')
    except:
        return False
    # read some data - contains at least 12 lines a 80 characters 
    data = f.read(12 * 80)
    f.close()
    if _getVersion(data):
        return True
    return False


def _getVersion(data):
    """
    Extracts SEISAN version from given data chunk.
    
    @type data: String.
    @param data: Data chunk. 
    @rtype: String or None.
    @return: SEISAN version.
    """
    # check size of data chunk
    if len(data) < 12 * 80:
        return False
    if data[0:2] == 'KP'and data[82] == 'P':
        return ("PC", 32, 6)
    elif data[0:8] == '\x00\x00\x00\x00\x00\x00\x00P' and \
        data[88:96] == '\x00\x00\x00\x00\x00\x00\x00P':
        return ("SUN", 64, 7)
    elif data[0:8] == 'P\x00\x00\x00\x00\x00\x00\x00' and \
        data[88:96] == '\x00\x00\x00\x00\x00\x00\x00P':
        return ("PC", 64, 7)
    elif data[0:4] == '\x00\x00\x00P' and data[84:88] == '\x00\x00\x00P':
        return ("SUN", 32, 7)
    elif data[0:4] == 'P\x00\x00\x00' and data[84:88] == 'P\x00\x00\x00':
        return ("PC", 32, 7)
    return None


def readSEISAN(filename, headonly=False, **kwargs):
    """
    Reads a SEISAN file and returns an L{obspy.Stream} object.
    
    @param filename: SEISAN file to be read.
    @rtype: L{obspy.Stream}.
    @return: A ObsPy Stream object.
    """
    # read data chunk from given file
    fh = open(filename, 'rb')
    data = fh.read(80 * 12)
    # get version info from file
    (platform, arch, version) = _getVersion(data)
    # fetch lines
    fh.seek(0)
    seisan = {}
    header = {'seisan': seisan}
    # start with event file header
    # line 1
    data = _readline(fh)
    seisan['network_name'] = data[1:30]
    seisan['number_of_channels'] = data[30:33]
    seisan['year'] = data[33:36]
    seisan['day'] = data[37:40]
    seisan['month'] = data[41:43]
    seisan['hr'] = data[47:49]
    seisan['min'] = data[50:52]
    seisan['sec'] = data[53:59]
    seisan['total_time_window'] = data[60:69]
    # line 2
    data = _readline(fh)
    # line 3
    # calculate number of lines with channels
    noc = int(seisan['number_of_channels'])
    nol = noc // 3 + (noc % 3 and 1)
    if nol < 10:
        nol = 10
    seisan['channels'] = {}
    for _i in xrange(0, nol):
        data = _readline(fh)
        temp = _parseChannel(data[0:28])
        if temp['station_code']:
            seisan['channels'][_i * 3] = temp
        temp = _parseChannel(data[28:52])
        if temp['station_code']:
            seisan['channels'][_i * 3 + 1] = temp
        temp = _parseChannel(data[52:78])
        if temp['station_code']:
            seisan['channels'][_i * 3 + 2] = temp
    # now parse each event file channel header + data
    for _i in xrange(noc):
        data = _readline(fh, 1040)
    import pprint
    pprint.pprint(seisan)

def _parseChannel(data):
    temp = {}
    temp['station_code'] = data[1:5].strip()
    temp['first_two_components'] = data[5:7]
    temp['last_component_code'] = data[8]
    temp['station_code_last_character'] = data[9]
    temp['start_time_relative_to_event_file_time'] = data[10:17].strip()
    temp['station_data_interval_length'] = data[18:26].strip()
    return temp

def _readline(fh, length=80):
    data = fh.read(length + 8)
    end = length + 4
    start = 4
    return data[start:end]

def writeSEISAN(stream_object, filename, **kwargs):
    """
    Writes SEISAN file.
    
    @type stream_object: L{obspy.Stream}.
    @param stream_object: A ObsPy Stream object.
    @param filename: SEISAB file to be written.
    """
    raise NotImplementedError


class SEISANFile(object):
    """
    """
    def __init__(self, fh):
        """
        """
        self.fh = fh
