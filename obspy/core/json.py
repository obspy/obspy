# -*- coding: utf-8 -*-
"""
JSON write support

JavaScript Object Notation is a text-based open standard designed for
human-readable data interchange. The JSON format is often used for serializing
and transmitting structured data over a network connection. It is used 
primarily to transmit data between a server and web application, serving as an
alternative to XML.

This module provides a class to create a "default" function accepted by the 
python json module, and versions of the json.dump and json.dumps functions
which can directly accept ObsPy Event objects.

"""
import json
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
    
    def __init__(self, omit_nulls=True):
        # Allows customization of the function
        self.OMIT_NULLS = omit_nulls

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
            # Use a special format strftime?
            return str(obj)
        elif isinstance(obj, ResourceIdentifier):
            # Always want ID as a string
            return str(obj)
        else:
            return None


def _parse_kwargs(kw_dict, **kwargs):
    """Prepare keyword arguments for json dump function"""
    if kwargs.get("compact"):
        kw_dict["separators"] =  (',',':')
        no_nulls = True
    else:
        no_nulls = False
    kw_dict["default"] = Default(omit_nulls=no_nulls)
    return kw_dict

def dumps(c, compact=True, **kwargs):
    """
    Return JSON string of an OBSPY Event-type object.

    :type c: obspy.core.event object
    :param c: ObsPy object to serialize
    :param bool compact: Use no spaces between separators and omit null values
    
    All other keyword params are passed to json.dumps

    """
    kwargs = _parse_kwargs(kwargs, compact=compact)
    return json.dumps(c, **kwargs)    

def dump(c, fp, compact=True, **kwargs):
    """
    Write OBSPY Event-type object to file as JSON.
    
    :type c: obspy.core.event object
    :param c: ObsPy object to serialize
    :type fp: file-like object
    :param fp: File to write JSON
    :param bool compact: Use no spaces between separators and omit null values

    All other keyword params are passed to json.dump
    
    """
    kwargs = _parse_kwargs(kwargs, compact=compact)
    return json.dump(c, fp, **kwargs)    

