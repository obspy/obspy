#!/usr/bin/env python
"""General parser class for uniform seismogram reading in different
formats. Classes for special type format seismogram reading inherit from
this general class.

It is assumed that the class, inherting from this class provides the following
attributes:

    julsec            - start time of seismogram in seconds since 1970 (float)
    trace             - the actual seismogram data (list of floats)
    dt                - sampling rate in seconds (float)

The Parser Class provides the following attributes and methods():

    npts              - number of data points
    date_time()       - pretty UTC date string computed from julsec
    date_to_julsec()  - convert arbitrary formated UTC date string to julsec

"""

import os,time

class Parser(object):
    """General Seismogram Parser Class"""
    
    def date_time(self):
        """Return pretty date string computed from julsec
        
        >>> p=Parser()
        >>> p.julsec = 0.0
        >>> p.date_time()
        '1970-01-01T00:00:00.000000'
        """
        
        intjulsec = int(self.julsec)
        musec = (self.julsec - intjulsec) * 1e6
        date = time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(intjulsec))
        return date + '.%06d' % musec
    
    def date_to_julsec(self,format,datestr):
        """Return julsec from formated UTC date-string 
        
        See time.strptime for format details
        >>> Parser().date_to_julsec('%Y%m%d%H%M%S','19700101000000')
        0.0
        """
        
        os.environ['TZ'] = 'UTC' # set environment timezone to UTC
        time.tzset()             # set time module timezone according to
                                 # environment
        return time.mktime(time.strptime(datestr,format))

    def is_attr(self,attr,typ,default,length=False,assertation=False,verbose=False):
        """True if attribute of cetain type or/and length exists
        
        Function is probably most useful for checking if necessary
        attributes are provided, e.g. when writing seismograms to file
        >>> p=Parser()
        >>> p.julsec = 0.0
        >>> p.is_attr('julsec',float,1.0)
        True
        >>> p.is_attr('julsec',int,123)
        False
        >>> p.julsec
        123
        """
        returnflag = True
        if not attr in self.__dict__.keys():
            if assertation:
                assert False,"%s attribute of Seismogram required" % attr
            if verbose:
                print "WARNING: %s attribute of Seismogram missing",
                print "forcing",attr,"=",default
            setattr(self,attr,default)
            returnflag = False
        if not isinstance(getattr(self,attr),typ):
            if verbose:
                print "WARNING: %s attribute of Seismogram not of type %s" % (attr,typ),
                print "forcing",attr,"=",default
            setattr(self,attr,default)
            returnflag = False
        if (length):
            if (len(getattr(self,attr)) > length):
                if verbose:
                    print "%s attribute of Seismogram is > %i" % (attribute,length)
                    print "forcing",attribute,"=",default
                attribute=default
                returnflag = False
        return returnflag

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
    doctest.master.summarize(True) # summary even if all tests passed correctly
