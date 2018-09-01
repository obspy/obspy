"""
Module for converting Catalog objects to and from dictionaries.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import re

import obspy
import obspy.core.event as ev

UTC_KEYS = ("creation_time", "time", "reference")

EVENT_ATTRS = ev.Event._containers + [x[0] for x in ev.Event._properties]


def catalog_to_dict(obj):
    """
    Recursively convert a event related objects to a dict.

    :type obj: Any object associated with Obspy's event hierarchy.
    :param obj: The object to be converted to a dictionary.

    :return: A dict.
    """
    # if this is a non-recursible type (ie a leaf) return it
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    # if a sequence recurse each member
    elif isinstance(obj, (list, tuple)):
        if not len(obj):  # container is empty
            return obj
        else:
            return [catalog_to_dict(x) for x in obj]
    elif isinstance(obj, dict):  # if this is a dict recurse on each value
        return {key: catalog_to_dict(value) for key, value in obj.items()}
    else:  # else if this is an obspy class convert to dict
        return catalog_to_dict(_obj_to_dict(obj))


def dict_to_catalog(catalog_dict):
    """
    Create a catalog from a dictionary.

    :param catalog_dict: Catalog information in dictionary format.
    :type catalog_dict: dict.

    :return: An ObsPy :class:`~obspy.core.event.Catalog` object.
    """
    assert isinstance(catalog_dict, dict)
    return obspy.Catalog(**_parse_dict_class(catalog_dict))


def _get_params_from_docs(obj):
    """
    Attempt to figure out params for obj from the doc strings.
    """
    doc_list = obj.__doc__.splitlines(keepends=False)
    params_lines = [x for x in doc_list if ":param" in x]
    params = [x.split(":")[1].replace("param ", "") for x in params_lines]
    return params


def _getattr_factory(attributes):
    """
    Return a function that looks for attributes on an object and puts them
    into a dictionary. None will be returned if the object does not have any
    of the attributes.
    """

    def func(obj):
        out = {x: getattr(obj, x) for x in attributes if hasattr(obj, x)}
        return out or None  # return None rather than empty dict

    return func


def _camel2snake(name):
    """
    Function to convert CamelCase to snake_case.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2


def make_class_map():
    """
    Return a dict that maps names in QML to their obspy class.
    """
    # Loop classes in mod, convert to snake_case and add to dict
    out = {}
    for name, obj in ev.__dict__.items():
        if isinstance(obj, type):
            name_lower = _camel2snake(name)
            name_plural = name_lower + "s"
            out[name_lower] = obj
            out[name_plural] = obj
    # add UTCDateTimes
    for utckey in UTC_KEYS:
        out[utckey] = obspy.UTCDateTime
    # add quality stuff
    out["quality"] = ev.OriginQuality
    # add waveform ids
    out["waveform_id"] = obspy.core.event.WaveformStreamID
    # add nodal_planes (this one is a bit weird)
    out["nodal_plane_1"] = ev.NodalPlane
    out["nodal_plane_2"] = ev.NodalPlane
    out["nodal_planes"] = ev.NodalPlanes
    return out


# a cache for functions that convert obspy objects to dictionaries
_OBSPY_TO_DICT_FUNCS = {obspy.UTCDateTime: lambda x: str(x),
                        ev.Event: _getattr_factory(EVENT_ATTRS)}

# a cache for mapping attribute names to expected obspy classes
_OBSPY_CLASS_MAP = make_class_map()


def _obj_to_dict(obj):
    """
    Return the dict representation of object.

    Uses only public interfaces to in attempt to future-proof the
    serialization schemas.
    """
    try:
        return _OBSPY_TO_DICT_FUNCS[type(obj)](obj)
    except KeyError:
        params = _get_params_from_docs(obj)
        # create function for processing
        _OBSPY_TO_DICT_FUNCS[type(obj)] = _getattr_factory(params)
        # register function for future caching
        return _OBSPY_TO_DICT_FUNCS[type(obj)](obj)


def _parse_dict_class(input_dict):
    """
    Parse a dictionary, init expected obspy classes.
    """
    # get intersection between cdict
    class_set = set(_OBSPY_CLASS_MAP)
    cdict_set = set(input_dict)
    # get set of keys that are obspy classes in the current dict
    class_keys = class_set & cdict_set
    # iterate over keys that are also classes and recurse when needed
    for key in class_keys:
        cls = _OBSPY_CLASS_MAP[key]
        val = input_dict[key]
        if isinstance(val, list):
            out = []  # a blank list for storing outputs
            for item in val:
                out.append(_init_update(item, cls))
            input_dict[key] = out
        elif isinstance(val, dict):
            input_dict[key] = _init_update(val, cls)
        elif isinstance(val, str):
            input_dict[key] = cls(val)

    return input_dict


def _init_update(input_dict, cls):
    """
    init an object from cls and update its dict with indict.
    """
    if not input_dict:
        return input_dict
    obj = cls(**_parse_dict_class(input_dict))
    # some objects instantiate even with None param, set back to None.
    # Maybe not an issue after  #2185?
    for attr in set(obj.__dict__) & set(input_dict):
        if input_dict[attr] is None:
            setattr(obj, attr, None)
    return obj
