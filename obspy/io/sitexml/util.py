# -*- coding: utf-8 -*-
"""

This module provides enumerations defined in the
SiteXML schema.

:copyright:
   ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util import Enum
from collections.abc import Iterable

TopographySchemaA = Enum([
    "T1",
    "T2",
    "T3",
    "T4",
])
"""
Topography is a precise (quantitative) description of the ground surface 
features of a site. Schema A is the topography description scheme of the Italian Code.
T1 : Flat surface, isolated slopes and cliffs with average slop angle 
T2 : Slopes with average slope angle i>15
T3 : Ridges with crest width significantly less than the base width and average slope angle 15
T4 : Ridges with crest width significantly less than the base width and average slope angle i>30
Allowed values are:

* ``"T1"``
* ``"T2"``
* ``"T3"``
* ``"T4"``
"""

TopographySchemaB = Enum([
    "Valley",
    "Lower slope",
    "Flat",
    "Middle slope",
    "Upper slope",
    "Ridge",
])
"""
Topography is a precise (quantitative) description of the ground surface 
features of a site. Schema B is the one proposed by Burjanek et al. (2014).
For the precise definition of the allowed values refer to SERA Deliverable D7.1.
Allowed values are:

* ``"Valley"``
* ``"Lower slope"``
* ``"Flat"``
* ``"Middle slope"``
* ``"Upper slope"``
* ``"Ridge"``
"""

MorphologyType = Enum([
    "Plain",
    "Valley - Basin",
    "Slope",
    "Ridge",
])
"""
Qualitative description of the shape of the earth's surface.
Allowed values are:

* ``"Plain"``
* ``"Valley - Basin"``
* ``"Slope"``
* ``"Ridge"``
"""

EC8Class = Enum([
    "A", 
    "B", 
    "C", 
    "D", 
    "E", 
    "S1", 
    "S2", 
    "Undefined"
])
"""
Ground type according to Eurocode 8 (EC8 § 3.1.2, Table 3.1), 
based on the velocityS30Value and geotechnical description
Allowed values are:

* ``"A"``
* ``"B"`` 
* ``"C"`` 
* ``"D"`` 
* ``"E"`` 
* ``"S1"`` 
* ``"S2"`` 
* ``"Undefined"``

"""

ResonanceFrequencyMethod = Enum([
    "HVSR EARTHQUAKE RECORDS",
    "HVSR NOISE",
    "SSR EARTHQUAKE RECORDS",
    "SSR NOISE",
    "INFERRED",
])
"""
Method used for the estimation of the dominant frequency, f0, of a site.
Allowed values are:

* ``"HVSR EARTHQUAKE RECORDS"``
* ``"HVSR NOISE"``
* ``"SSR EARTHQUAKE RECORDS"``
* ``"SSR NOISE"``
* ``"INFERRED"``
"""

VelocityS30Method = Enum([
     "Geology", 
     "Topographic Slope", 
     "SPT", 
     "CPT", 
     "Laboratory", 
     "S-REFR", 
     "S-REFL", 
     "SASW", 
     "MASW", 
     "SWI", 
     "SPAC/F-K", 
     "ReMi", 
     "Crosshole", 
     "Downhole", 
     "Uphole", 
     "P-S Log", 
     "Seismic Cone", 
     "DH Strong Motion Arrays"
])
"""
Method used for the extraction of S-wave velocity profiles and, thus, 
of the average shear-wave velocity over the top 30 meters of the soil column, Vs30.
Allowed values are:

* ``"Geology"``
* ``"Topographic Slope"``
* ``"SPT"``
* ``"CPT"``
* ``"Laboratory"``
* ``"S-REFR"``
* ``"S-REFL"``
* ``"SASW"``
* ``"MASW"``
* ``"SWI"``
* ``"SPAC/F-K"``
* ``"ReMi"``
* ``"Crosshole"``
* ``"Downhole"``
* ``"Uphole"``
* ``"P-S Log"``
* ``"Seismic Cone"``
* ``"DH Strong Motion Arrays"``
"""

def _pretty_str(obj):
    return ", ".join(
        f"{key}='{value}'" for key, value in vars(obj).items() 
        if value is not None
    )

def _enum_property(attr_name, enum_type):
    """
    Method to produce getter/setter functions 
    and validate enum type values.
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None or value in enum_type:
            setattr(self, private_name, value)
        else:
            valid_values = [e for e in enum_type]
            raise ValueError(
                f"\nInvalid value for '{attr_name}'. \
                    Expected one of {valid_values}, but got '{value}'."
            )
    return property(getter, setter)

def _add_property(attr_name, wrapper_type):
    """
    Method to produce getter/setter functions 
    and wrap argument values into the appropriate type.
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None or isinstance(value, wrapper_type):
            setattr(self, private_name, value)
        else:
            try:
                setattr(self, private_name, wrapper_type(value))
            except Exception as e:
                raise TypeError(f"Could not convert {value} \
                                to {wrapper_type.__name__}: {e}")

    return property(getter, setter)

def _add_iterable_property(
    attr_name,
    wrapper_type,
    iterable_type=list,
    allow_none=True
):
    """
    Creates a property that wraps iterable elements into wrapper_type.

    :param attr_name: name of the attribute
    :param wrapper_type: class used to wrap each element
    :param iterable_type: list, tuple, etc.
    :param allow_none: whether None is allowed
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, values):
        if values is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise TypeError(f"{attr_name} cannot be None")

        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise TypeError(f"{attr_name} must be an iterable")

        wrapped_items = []
        for v in values:
            if isinstance(v, wrapper_type):
                wrapped_items.append(v)
            else:
                try:
                    wrapped_items.append(wrapper_type(v))
                except Exception as e:
                    raise TypeError(
                        f"Could not convert element {v} to {wrapper_type.__name__}: {e}"
                    )

        setattr(self, private_name, iterable_type(wrapped_items))

    return property(getter, setter)

def _validate_list_of_vwu(self, name, value):
    """
    Validates and standardizes a list of ValueWithUncertainty objects.
    Converts numbers to ValueWithUncertainty, keeps None, raises on bad types.
    """
    if value is None:
        return []

    if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable \
                    (e.g., a list of floats or ValueWithUncertainty).")

    validated = []
    for i, item in enumerate(value):
        if item is None:
            validated.append(None)
        elif isinstance(item, ValueWithUncertainty):
            validated.append(item)
        elif isinstance(item, (int, float)):
            validated.append(ValueWithUncertainty(item))
        else:
            raise TypeError(f"{name}[{i}] is not a valid type \
                    (expected int, float, ValueWithUncertainty, or None): {item}")
    
    return validated

def vwu_list_properties(*attributes):
    def decorator(cls):
        cls._validate_list_of_vwu = _validate_list_of_vwu

        for attr_name in attributes:
            private_name = f"_{attr_name}"

            def getter(self, name=private_name):
                return getattr(self, name)

            def setter(self, value, name=private_name, attr=attr_name):
                validated = self._validate_list_of_vwu(attr, value)
                setattr(self, name, validated)

            setattr(cls, attr_name, property(getter, setter))

        return cls
    return decorator