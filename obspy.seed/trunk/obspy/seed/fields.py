# -*- coding: utf-8 -*-


class SEEDTypeException(Exception):
    pass

class Field:
    """General SEEF field."""
    
    def __str__(self):
        if self.id:
            return "F%02d" % self.id


class Integer(Field):
    
    value = None
    
    def __init__(self, id, name, length, version=None):
        self.id = id
        self.name = name
        self.length = length
        self.version = version
    
    def __call__(self):
        return self
    
    def read(self, data):
        temp = data.read(self.length)
        try:
            temp = int(temp)
        except:
            raise SEEDTypeException("No integer value found.")
        self.value = temp
        return temp


class Float(Field):
    
    def __init__(self, id, name, length, version=None):
        self.id = id
        self.name = name
        self.length = length
        self.version = version
    
    def __call__(self):
        return self
    
    def read(self, data):
        temp = data.read(self.length)
        # XXX: some SEED writer are screwing up
        if temp[-1] in [' ', '-']:
            data.seek(-1,1)
            temp = temp[:-1]
        try:
            temp = float(temp)
        except:
            raise SEEDTypeException("No float value found.")
        return temp


class FixedString(Field):
    
    def __init__(self, id, name, length, flags='', version=None):
        self.id = id
        self.name = name
        self.length = length
        self.version = version
    
    def read(self, data):
        return data.read(self.length).strip()


class VariableString(Field):
    """Variable length ASCII string, ending with a tilde: ~ (ASCII 126).
    
    Variable length fields cannot have leading or trailing spaces.
    """
    def __init__(self, id, name, min_length=0, max_length=None, flags=None, 
                 version=None):
        self.id = id
        self.name = name
        self.min_length = min_length
        self.max_length = max_length
        self.flags = flags
        self.version = version
    
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


class MultipleLoop(Field):
    
    def __init__(self, name, index_field, data_fields, version=None):
        if not isinstance(data_fields, list):
            data_fields = [data_fields]
        self.id = None
        self.index_field = index_field
        self.length = 0
        self.data_fields = data_fields
        self.name = name
        self.version = version
    
    def read(self, data):
        temp = []
        for _i in range(0,self.length):
            temp2=[]
            for field in self.data_fields:
                temp2.append(field)
            temp.append(temp2)
        return temp


class SimpleLoop(Field):
    
    def __init__(self, index_field, data_field, version=None):
        self.id = None
        self.index_field = index_field
        self.length = 0
        self.data_field = data_field
        self.name = data_field.name
        self.version = version
    
    def read(self, data):
        temp = []
        for _i in range(0,self.length):
            temp.append(self.data_field)
        return temp