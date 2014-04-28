# -*- coding: utf-8 -*-
"""
EVT (Kinemetrics) support for ObsPy.
Base classes (Can not be directly called)

:copyright:
Royal Observatory of Belgium, 2013
:license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
"""


from obspy import UTCDateTime


### -------------   Definition of Evt Errors               -----------------###
#-----------------------------------------------------------------------------#

class EVTBaseError(Exception):
    """
    Base Class for all EVT specific errors.
    """
    pass


class EVTBadDataError(EVTBaseError):
    """
    Raised if a Bad Data is readed in EVT File
    """
    pass


class EVTBadHeaderError(EVTBaseError):
    """
    Raised if an error occured in EVT headers
    """
    pass


class EVTEOFError(EVTBaseError):
    """
    Raised if an EOF occured (formely not an error)
    """
    pass


class EVTNotImplementedError(EVTBaseError):
    """
    Raised if a function is called with not implemented options
    """
    pass

#---------- Class EVT_Virtual ------------------------------------------------#
#-----------------------------------------------------------------------------#


class EVT_Virtual(object):
    """
    class for parameters reading.
    The dictionnary have this structure :
       {"name":[header_place,["function","param"],value], ...}
            name is the attribute (key) (in lower cases)
            header_place : offset to find value in file
            function : function to call to set value (option)
            param : parameters to send to function
            value : value of name (can be omited)
    """

    def __init__(self):
        self.diconame = ""  # set the dictionnary name to be used

    def __getattr__(self, item):
        """
        __getattr__ is called only if no class attribute is found
        :type item: string
        :param item: name of the attribute (key)
        :rtype: any
        :return: the value in the dictionnary
        """
        key = item.lower()
        if key in eval(self.diconame):
            try:
                return eval(self.diconame)[key][2]
            except IndexError:
                return("No value for " + key)

    def __setattr__(self, item, value):
        """
        __setattr__ is called for every attribute assignment
        :type item: string
        :param item: name of the attribute (key)
        :type value: any
        :param value: value is the value to be assigned to item
        """
        key = item.lower()
        try:
            if getattr(self, diconame) != "":
                if key in eval(self.diconame):
                    eval(self.diconame)[key][2] = value
                else:
                    object.__setattr__(self, item, value)
            else:
                object.__setattr__(self, item, value)
        except NameError:
            object.__setattr__(self, item, value)

    def unsetdico(self):
        """
        remove value from dictionary
        """
        for key in eval(self.diconame):
            try:
                eval(self.diconame)[key].pop(2)
            except IndexError:
                pass

    def setdico(self, val, offset=0):
        """
        fill dico with values found in val
        :type val: list
        :param val : a list of values
        :type offset: integer
        :param offset : offset in the dictonary (header place)
        """
        if type(val) is not list:
            raise TypeError("setdico waiting a list")
        for key in eval(self.diconame):
            index = eval(self.diconame)[key][0]-offset
            if index < len(val) and index >= 0:
                if eval(self.diconame)[key][1] != "":
                    fct = 'self.' + eval(self.diconame)[key][1][0]
                    param = eval(self.diconame)[key][1][1]
                    value = eval(fct)(val[index], param, val, offset)
                else:
                    value = val[index]
                try:
                    eval(self.diconame)[key][2] = value
                except IndexError:
                    eval(self.diconame)[key].append(value)

    def __str__(self):
        """
        create a string with all dictionnary values
        :rtype:  string
        :return: string to be printed

        """
        chaine = ""
        for vname in sorted(eval(self.diconame)):
            chaine += vname + "\t is \t" + str(getattr(self, vname)) + "\n"
        return chaine

    def _time(self, blocktime, param, val, offset):
        """
        change a EVT time format to Obspy UTCDateTime format
        :param blocktime : time in sec after 1980/1/1
        :param param: parameter with milliseconds values (in val)
        :param val: list of value
        :param offset: Not used
        """
        frame_time = blocktime
        if param > 0:
            frame_milli = val[param-offset]
        else:
            frame_milli = 0
        frame_time += 315532800  # diff between 1970/1/1 and 1980/1/1
        time = UTCDateTime(frame_time) + frame_milli/1000.0
        time.precison = 3
        return time

    def _strnull(self, strn, param, val, offset):
        """
        change a C string (null teminated in Pythonstring
        :type str: string
        :param str: string to convert
        :param param: not used
        :param val: not used
        :param offset: not used
        :rtype: string
        """
        if type(strn) == str:
            newstr = strn.split('\0')[0]
        else:
            newstr = strn
        return newstr

    def _array(self, firstval, param, val, offset):
        """
        extract a list of values from val
        :param firstval: first value to extract (unused)
        :param param: a list with the size of the list, the dimesnion of the
                 structure, and the first value to read
        :param val: list of value
        :param offset: not used
        :rtype: list
        :return: a list of values
        """
        ret = []
        sizearray = param[0]
        sizestru = param[1]
        index0 = param[2]
        for i in range(sizearray):
            ret.append(val[index0-offset+(i*sizestru)])
        return ret

    def _arraynull(self, firstval, param, val, offset):
        """
        extract a list of values from val and change C string to python
        :param firstval: first value to extract (unused)
        :param param: a list with the size of the list, the dimesnion of the
                 structure, and the first value to read
        :param val: list of value
        :param offset:
        :rtype: list
        :return: a list of values
        """

        ret = []
        sizearray = param[0]
        sizestru = param[1]
        index0 = param[2]
        for i in range(sizearray):
            mystr = self._strnull(val[index0-offset+(i*sizestru)], '', '', '')
            ret.append(mystr)
        return ret

    def _instrument(self, code, param, val, offset):
        """
        change instrument type code to name
        :param code: code to convert
        :param param: not used
        :param val: not used
        :param offset: not used
        :rtype: string
        """
        dico = {0: 'QDR', 9: 'K2', 10: 'Makalu', 20: 'New Etna',
                30: 'Rock', 40: 'SSA2EVT'}
        if code in dico:
            return dico[code]
        else:
            raise EVTBadHeaderError("Bad Instrument Code")
