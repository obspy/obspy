# -*- coding: utf-8 -*-
"""
The ObsPy core classes.

Don't inherit from anything inside this module. You will get circular imports!
"""
from obspy.numpy import array
from obspy.util import Stats, getParser
import os


__all__ = ['Trace']


class TraceBase(object):
    """
    Dummy class for Trace inheritance.
    
    Don't remove or inherit from this class or you will break something!
    """
    pass


class Trace(TraceBase):
    """
    A general seismic trace class containing a single channel.
    
    This class automatically searches for parsers. Do not inherit from
    this class.
    """
    methods = ['read', 'write', 'check']
    
    def __init__(self, *args, **kwargs):
        super(Trace, self).__init__(*args, **kwargs)
        self.filename = None
        self.format = None
        self.data = array()
        self.stats = Stats()
        self._formats = []
        # inherit from all obspy trace classes
        Trace.__bases__ = getParser()
        super(Trace, self).__init__()
        # create parser specific methods
        for base in Trace.__bases__:
            if not hasattr(base, '__format__'):
                continue
            if not isinstance(base.__format__, basestring):
                continue
            format = base.__format__.upper()
            self._formats.append(format)
            for method in self.methods:
                func_name = '%s%s' % (method, format)
                if not hasattr(base, method):
                    continue
                setattr(Trace, func_name, getattr(base, method))
    
    def read(self, filename, format=None, **kwargs):
        """
        Reads a trace from a given file.
        
        This method tries to detect automatically the file type by the file 
        name extension if no format option is given. Any additional keywords 
        are passed to the used file parser.
        """
        # check format
        if format and format not in self._formats:
            # specified format not known
            msg = "Unknown file format '%s'. " % (format)
            msg += "Use one of the following formats: %s" % self._formats
            raise NotImplementedError(msg)
        if not format:
            ext = os.path.splitext(filename)[1][1:].upper()
            if ext.upper() not in self._formats:
                msg = "Can't autodetect the file format. Please use the format"
                msg += "option: %s" % self._formats
                raise NotImplementedError(msg)
            format = ext
        # call parser class
        func_name = 'read%s' % (format)
        func = getattr(self, func_name)
        data=func(filename, **kwargs)
        self.format=format
        self.filename=filename
        return data
    
    def write(self, filename, format=None, **kwargs):
        """
        Writes the current trace into a file.
        """
        if not format:
            format=self.format or None
        # call parser class
        func_name = 'write%s' % (format)
        func = getattr(self, func_name)
        func(filename, **kwargs)
