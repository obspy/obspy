# -*- coding: utf-8 -*-
"""
JSON write support

JavaScript Object Notation is a text-based open standard designed for
human-readable data interchange. The JSON format is often used for serializing
and transmitting structured data over a network connection. It is used 
primarily to transmit data between a server and web application, serving as an
alternative to XML.

This module provides:
---------------------
Default : a class to create a "default" function accepted by the 
python json module Encoder classes, valid for obspy.core.event objects

get_dump_kwargs : function that wraps a Default function and some other params
into a dictionary suitable for passing to json.dumps. 

Example
-------
>>> import json, obspy
>>> c = obspy.core.event.Catalog()
>>> d = Default(omit_nulls=False)
>>> s = json.dumps(c, default=d)

"""
from obspy.core.event import ( AttribDict, Catalog, UTCDateTime, 
    ResourceIdentifier )


class Default(object):
    """
    Class to create a "default" function for the json.dump* functions
    which is passed to the JSONEncoder.
    
    """
    _catalog_attrib = ('events','comments', 'description','creation_info',
        'resource_id')

    OMIT_NULLS = None
    TIME_FORMAT = None

    def __init__(self, omit_nulls=True, time_format=None):
        """
        Create a "default" function for JSONEncoder for ObsPy objects

        :param bool omit_nulls: Leave out any null or empty values (True)
        :param str time_format: Format string passed to strftime (None)

        """
        # Allows customization of the function
        self.OMIT_NULLS = omit_nulls
        self.TIME_FORMAT = time_format

    def __call__(self, obj):
        """
        Deal with obspy event objects in JSON Encoder
        
        This function can be passed to the json module's
        'default' keyword parameter

        """
        # Most event objects have dict methods, construct a dict
        # and deal with special cases that don't
        if isinstance(obj, AttribDict):
            # Map to a serializable dict
            # Leave out nulls, empty strings, list, dicts, except for numbers
            if self.OMIT_NULLS:
                return { k:v for k,v in obj.iteritems() if v or v == 0 } 
            else:
                return { k:v for k,v in obj.iteritems() }
        elif isinstance(obj, Catalog):
            # Catalog isn't a dict
            return { k : getattr(obj, k) for k in self._catalog_attrib 
                if getattr(obj,k) 
            }
        elif isinstance(obj, UTCDateTime):
            if self.TIME_FORMAT is None:
                return str(obj)
            else:
                return obj.strftime(self.TIME_FORMAT)
        elif isinstance(obj, ResourceIdentifier):
            # Always want ID as a string
            return str(obj)
        else:
            return None


def get_dump_kwargs(minify=True, no_nulls=True, **kwargs):
    """
    Return dict of keyword args for json.dump or json.dumps

    :param bool minify: Use no spaces between separators (True)
    :param bool no_nulls: Omit null values and empty sequences/mappings (True)
    
    """
    if minify:
        kwargs["separators"] =  (',',':')
    kwargs["default"] = Default(omit_nulls=no_nulls)
    return kwargs


