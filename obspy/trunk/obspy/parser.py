#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: parser.py
#  Purpose: Base Seismogram Parser Class
#   Author: Moritz Beyreuther
#    Email: moritz.beyreuther@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Moritz Beyreuther
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#---------------------------------------------------------------------
"""General parser class for uniform seismogram reading in different
formats. Classes for special type format seismogram reading inherit from
this general class.

It is assumed that the class, inherting from this class provides the following
attributes:

    station           - station name (string)
    julsec            - start time of seismogram in seconds since 1970 (float)
    trace             - the actual seismogram data (list of floats)
    df                - sampling rate in Hz (float)
    npts              - number of samples/data points (int)

The Parser Class provides the following attributes and methods():

    date_time()       - pretty UTC date string computed from julsec
    date_to_julsec()  - convert arbitrary formated UTC date string to julsec

"""

import os,time

class Parser(object):
    """General Seismogram Parser Class"""
    
    def __str__(self):
        """Overload to string method to pretty print attributes of class"""
        output = ''
        for attr,val in sorted(self.__dict__.iteritems()):
            if attr == 'trace':
                if len(val) > 9:
                    output += "%20s [%d,%d,%d,%d,...,%d,%d,%d,%d]\n" % tuple([attr]+val[0:4]+val[-5:-1])
                    continue
            output += "%20s %s\n"%(attr,str(val))
        return output


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
