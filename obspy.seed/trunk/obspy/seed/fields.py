# -*- coding: utf-8 -*-


from obspy.seed import utils


class SEEDTypeException(Exception):
    pass

class Field:
    """General SEED field."""
    
    field_id = None
    version = None
    mask = None
    optional = False
    
    def __init__(self, id, name, *args, **kwargs):
        self.id = id
        self.name = name
        self.version = kwargs.get('version', None)
        self.mask = kwargs.get('mask', None)
        self.optional = kwargs.get('optional', False)
        if self.id:
            self.field_id = "F%02d" % self.id
        self.field_name = utils.toXMLTag(self.name)
        self.attribute_name = utils.toAttribute(self.name)
    
    def __str__(self):
        if self.id:
            return "F%02d" % self.id


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
    
    def __init__(self, id, name, length, **kwargs):
        Field.__init__(self, id, name, **kwargs)
        self.length = length
        self.default = 0
        if not self.mask:
            msg = "Float field %s requires a data mask." % self.field_name
            raise SEEDTypeException(msg)
    
    def read(self, data):
        temp = data.read(self.length)
        # XXX: some SEED writer are screwing up
        if temp[-1] in [' ', '-']:
            data.seek(-1,1)
            temp = temp[:-1]
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
        # XXX: very ugly
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
        result = format_str % data.strip()
        if len(result) != self.length:
            msg = "Invalid field length %d of %d in %s." % \
                  (len(result), self.length, self.field_name)
            raise SEEDTypeException(msg)
        return result


class VariableString(Field):
    """Variable length ASCII string, ending with a tilde: ~ (ASCII 126).
    
    Variable length fields cannot have leading or trailing spaces.
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
        result = str(data)+'~'
        # Character counts for variable length fields do not include the tilde 
        # terminator. 
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