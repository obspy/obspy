# -*- coding: utf-8 -*-


class SEEDTypeException(Exception):
    pass


class IntegerSEEDField:
    
    value = None
    
    def __init__(self, name, length):
        self.name = name
        self.length = length
    
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


class FloatSEEDField:
    
    def __init__(self, name, length):
        self.name = name
        self.length = length
    
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


class StringSEEDField:
    
    def __init__(self, name, length, flags=''):
        self.name = name
        self.length = length
    
    def read(self, data):
        return data.read(self.length).strip()


class VariableStringSEEDField:
    """Variable length ASCII string, ending with a tilde: ~ (ASCII 126).
    
    Variable length fields cannot have leading or trailing spaces.
    """
    def __init__(self, name, min_length=0, max_length=None, flags=None):
        self.name = name
        self.min_length = min_length
        self.max_length = max_length
        self.flags = flags
    
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


class LoopingSEEDField:
    
    def __init__(self, name, length_field, data_fields):
        if not isinstance(length_field, IntegerSEEDField):
            raise Exception('IntegerSEEDField expected for number of loops!')
        if not isinstance(data_fields, list):
            data_fields = [data_fields]
        self.length_field = length_field
        self.data_fields = data_fields
        self.name = name
    
    def read(self, data):
        temp = []
        for i in range(0,self.length_field.value):
            temp2={}
            for field in self.data_fields:
                name = field.name.title().replace(' ','')
                temp2[name] = field.read(data)
            temp.append(temp2)
        return temp


class SimpleLoopingSEEDField:
    
    def __init__(self, length_field, data_field):
        if not isinstance(length_field, IntegerSEEDField):
            raise Exception('IntegerSEEDField expected for number of loops!')
        self.length_field = length_field
        self.data_field = data_field
        self.name = data_field.name
    
    def read(self, data):
        temp = []
        for i in range(0,self.length_field.value):
            temp.append(self.data_field.read(data))
        return temp