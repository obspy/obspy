# -*- coding: utf-8 -*-
"""
Evt (Kinemetrics) support for ObsPy.
Base classes (cannot be directly called)

:copyright:
    Royal Observatory of Belgium, 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime


class EvtBaseError(Exception):
    """
    Base Class for all Evt specific errors.
    """
    pass


class EvtBadDataError(EvtBaseError):
    """
    Raised if bad data is encountered while reading an Evt file.
    """
    pass


class EvtBadHeaderError(EvtBaseError):
    """
    Raised if an error occured while parsing an Evt header.
    """
    pass


class EvtEOFError(EvtBaseError):
    """
    Raised if an unexpected EOF is encountered.
    """
    pass


class EvtVirtual(object):
    """
    class for parameters reading.
    The dictionary has this structure :
       {"name":[header_place,["function","param"],value], ...}
            name is the attribute (key) (in lower cases)
            header_place : offset to find value in file
            function : function to call to set value (option)
            param : parameters to send to function
            value : value of name (can be omitted)
    """
    def __init__(self):
        self.diconame = ""

    def __getattr__(self, item):
        """
        __getattr__ is called only if no class attribute is found
        :type item: str
        :param item: name of the attribute (key)
        :rtype: any
        :return: the value in the dictionary
        """
        key = item.lower()
        if key in self.HEADER:
            return self.HEADER[key][2]

    def unset_dict(self):
        """
        remove all values from dictionary
        """
        for key in self.HEADER:
            try:
                self.HEADER[key].pop(2)
            except IndexError:
                pass

    def set_dict(self, val, offset=0):
        """
        fill the dictionary with values found in the input 'val' list
            the nth value in val is placed in the dictionary if a key
            of type 'name':[nth, ''] exist
            the offset is used to include the 'val' list further
            in the dictionary
        :type val: list
        :param val : a list of values
        :type offset: int
        :param offset : offset in the dictionary
        """
        if not isinstance(val, tuple):
            raise TypeError("set_dict() expects a tuple")
        for key in self.HEADER:
            index = self.HEADER[key][0] - offset
            if 0 <= index < len(val):
                if self.HEADER[key][1] != "":
                    fct = self.HEADER[key][1][0]
                    param = self.HEADER[key][1][1]
                    value = getattr(self, fct)(val[index], param, val, offset)
                else:
                    value = val[index]
                try:
                    self.HEADER[key][2] = value
                except IndexError:
                    self.HEADER[key].append(value)

    def __str__(self):
        """
        create a string with all dictionary values
        :rtype:  str
        :return: string representation of dictionary
        """
        chaine = ""
        for vname in sorted(self.HEADER):
            chaine += vname + "\t is \t" + str(getattr(self, vname)) + "\n"
        return chaine

    def _time(self, blocktime, param, val, offset):
        """
        change a Evt time format to
                :class:`~obspy.core.utcdatetime.UTCDateTime` format
        :param blocktime : time in sec after 1980/1/1
        :param param: parameter with milliseconds values (in val)
        :param val: list of value
        :param offset: Not used
        """
        frame_time = blocktime
        if param > 0:
            frame_milli = val[param - offset]
        else:
            frame_milli = 0
        frame_time += 315532800  # diff between 1970/1/1 and 1980/1/1
        time = UTCDateTime(frame_time) + frame_milli / 1000.0
        return UTCDateTime(ns=time.ns)

    def _strnull(self, strn,
                 unused_param=None, unused_val=None, unused_offset=None):
        """
        Change a C string (null terminated to Python string)

        :type strn: str
        :param strn: string to convert
        :param param: not used
        :param val: not used
        :param offset: not used
        :rtype: str
        """
        return strn.split(b"\0", 1)[0].decode()

    def _array(self, unused_firstval, param, val, offset):
        """
        extract a list of 'size_array' values from val
           each value is separate in val by a distance of 'size_structure',
        :param firstval: first value to extract (unused)
        :param param: a list with the size of the list ('size_array'),
                      the dimension of the structure ('size_structure'),
                      and the first value to read ('index0')
        :param val: list of values
        :param offset: used
        :rtype: list
        :return: a list of values
        """
        ret = []
        size_array = param[0]
        size_structure = param[1]
        index0 = param[2]
        for i in range(size_array):
            ret.append(val[index0 - offset + (i * size_structure)])
        return ret

    def _arraynull(self, unused_firstval, param, val, offset):
        """
        extract a list of 'size_array' values from val
            and change C string to python str
        :param firstval: first value to extract (unused)
        :param param: a list with the size of the list, the dimension of the
                 structure, and the first value to read
        :param val: list of value
        :param offset:
        :rtype: list
        :return: a list of values
        """

        ret = []
        size_array = param[0]
        size_structure = param[1]
        index0 = param[2]
        for i in range(size_array):
            mystr = self._strnull(val[index0 - offset + (i * size_structure)])
            ret.append(mystr)
        return ret

    def _instrument(self, code, unused_param, unused_val, unused_offset):
        """
        change instrument type code to name
        :param code: code to convert
        :param param: not used
        :param val: not used
        :param offset: not used
        :rtype: str
        """
        dico = {0: 'QDR', 9: 'K2', 10: 'Makalu', 20: 'New Etna',
                30: 'Rock', 40: 'SSA2EVT'}
        if code in dico:
            return dico[code]
        else:
            raise EvtBadHeaderError("Bad Instrument Code")
