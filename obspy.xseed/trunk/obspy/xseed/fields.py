# -*- coding: utf-8 -*-

import re

from obspy.seed import utils


class SEEDTypeException(Exception):
    pass


class Field:
    """General SEED field."""
    
    def __init__(self, id, name, *args, **kwargs):
        self.id = id
        
        self.name = name
        self.version = kwargs.get('version', None)
        self.mask = kwargs.get('mask', None)
        self.optional = kwargs.get('optional', False)
        if self.id:
            self.field_id = "F%02d" % self.id
        else:
            self.field_id = None
        self.field_name = utils.toXMLTag(self.name)
        self.attribute_name = utils.toAttribute(self.name)
    
    def __str__(self):
        if self.id:
            return "F%02d" % self.id
    
    def _formatString(self, s, flags):
        """Using SEED specific flags to format strings.
        
        This method is partly adopted from fseed.py, the SEED builder for 
        SeisComP written by Andres Heinloo, GFZ Potsdam in 2005.
        """
        sn = str(s).strip()
        if not flags:
            return sn
        if 'T' in flags:
            # XXX: ignoring time flags for now - depending on the final format
            return sn
        rx_list = []
        if 'U' in flags:
            rx_list.append("[A-Z]")
        if 'L' in flags:
            rx_list.append("[a-z]")
        if 'N' in flags:
            rx_list.append("[0-9]")
        if 'P' in flags:
            rx_list.append("[^A-Za-z0-9 ]")
        if 'S' in flags:
            rx_list.append(" ")
        if '_' in flags:
            rx_list.append("_")
        if 'U' in flags and 'L' not in flags:
            sn = sn.upper()
        elif 'L' in flags and 'U' not in flags:
            sn = sn.lower()
        if 'S' in flags and 'X' not in flags:
            sn = sn.replace("_", " ")
        elif 'X' in flags and 'S' not in flags:
            sn = sn.replace(" ", "_")
        rx = "|".join(rx_list)
        sn = "".join(re.findall(rx, sn))
        if re.match("(" + rx + ")*$", sn) == None:
            msg = "Can't convert string %s with flags %s" % (s, flags)
            raise SEEDTypeException(msg) 
        return sn


class Integer(Field):
    
    def __init__(self, id, name, length, **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.default = 0
    
    def read(self, data):
        temp = data.read(self.length)
        try:
            temp = int(temp)
        except:
            msg = "No integer value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        return temp
    
    def write(self, data):
        format_str = "%%0%dd" % self.length
        try:
            temp = int(data)
        except:
            msg = "No integer value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        result = format_str % temp
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class Float(Field):
    """A float number with a fixed length and output mask."""
    
    def __init__(self, id, name, length, **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.default = 0
        if not self.mask:
            msg = "Float field %s requires a data mask." % self.field_name
            raise SEEDTypeException(msg)
    
    def read(self, data):
        temp = data.read(self.length)
        try:
            temp = float(temp)
        except:
            msg = "No float value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        return temp
    
    def write(self, data):
        format_str = "%%0%ds" % self.length
        try:
            temp = float(data)
        except:
            msg = "No float value found for %s." % self.field_name
            raise SEEDTypeException(msg)
        # special format for exponential output
        result = format_str % (self.mask % temp)
        if 'E' in self.mask or 'e' in self.mask:
            result = self.formatExponential(result)
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result

    def formatExponential(self, data):
        """Formats floats in a fixed exponential format.
        
        Different operation systems are delivering different output for the
        exponential format of floats. Here we ensure to deliver in a for SEED
        valid format independent of the OS. For speed issues we simple cut any 
        number ending with E+0XX or E-0XX down to E+XX or E-XX. This fails for 
        numbers XX>99, but should not occur, because the SEED standard does 
        not allow this values either.
        
        Python 2.5.2 (r252:60911, Feb 21 2008, 13:11:45) 
        [MSC v.1310 32 bit (Intel)] on win32
        >>> '%E' % 2.5
        '2.500000E+000'
        
        Python 2.5.2 (r252:60911, Apr  2 2008, 18:38:52)
        [GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on linux2
        >>> '%E' % 2.5
        '2.500000E+00'
        """
        data = data.upper()
        if data[-4] == 'E':
            return data
        if 'E+0' in data:
            return data.replace('E+0', 'E+')
        if 'E-0' in data:
            return data.replace('E-0', 'E-')
        msg = "Can't format float %s in field %s" % (data, self.field_name)
        raise SEEDTypeException(msg)


class FixedString(Field):
    
    def __init__(self, id, name, length, flags='', **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.flags = flags
        self.default = ' ' * length
    
    def read(self, data):
        return data.read(self.length).strip()
    
    def write(self, data):
        # Leave fixed length alphanumeric fields left justified (no leading 
        # spaces), and pad them with spaces (after the field’s contents).
        format_str = "%%-%ds" % self.length
        result = format_str % self._formatString(data, self.flags)
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class VariableString(Field):
    """Variable length ASCII string, ending with a tilde: ~ (ASCII 126).
    
    Variable length fields cannot have leading or trailing spaces. Character 
    counts for variable length fields do not include the tilde terminator. 
    """
    def __init__(self, id, name, min_length=0, max_length=None, flags=None, 
                 **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.flags = flags
        self.default = ' ' * min_length
    
    def read(self, data):
        buffer = ''
        if self.min_length:
            buffer = data.read(self.min_length)
            if '~' in buffer:
                return buffer.split('~')[0]
        temp = ''
        i = self.min_length
        while temp!='~':
            temp = data.read(1)
            if temp == '~':
                return buffer
            buffer += temp
            i=i+1
            if self.max_length and i>self.max_length:
                return buffer
        return buffer
    
    def write(self, data):
        result = self._formatString(data, self.flags)+'~'
        if self.max_length and len(result) > self.max_length+1:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        if len(result) < self.min_length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class MultipleLoop(Field):
    
    def __init__(self, name, index_field, data_fields, **kwargs):
        Field.__init__(self, None, name, **kwargs)
        if not isinstance(data_fields, list):
            data_fields = [data_fields]
        self.index_field = utils.toAttribute(index_field)
        self.length = 0
        self.data_fields = data_fields
    
    def getSubFields(self):
        temp = []
        for _i in range(0,self.length):
            temp2=[]
            for field in self.data_fields:
                temp2.append(field)
            temp.append(temp2)
        return temp


class SimpleLoop(Field):
    
    def __init__(self, index_field, data_field, **kwargs):
        Field.__init__(self, None, data_field.name, **kwargs)
        self.index_field = utils.toAttribute(index_field)
        self.length = 0
        self.data_field = data_field
    
    def getSubFields(self):
        temp = []
        for _i in range(0,self.length):
            temp.append(self.data_field)
        return temp