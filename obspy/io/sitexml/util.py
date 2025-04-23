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
Ground type according to Eurocode 8 (EC8 § 3.1.2, Table 3.1), based on the velocityS30Value and geotechnical description
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
* ``"Flat"``
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
Method by which velocityS30Value has been determined
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

def _sitexml_check_type(value, expected_type, param_name="value", allow_none=False):
    """
    Checks if a given value matches the expected type.

    :param value: The variable to check.
    :param expected_type: The expected type (or tuple of types).
    :param param_name: Optional name of the parameter for better error messages.
    :param allow_none: If True, allows None as a valid value.
    :raises TypeError: If the value is not of the expected type.
    """
    if allow_none and value is None:
        return value  # Allow None if specified

    if not isinstance(value, expected_type):
        raise TypeError(f"Expected '{param_name}' to be of type {expected_type.__name__}, "
                        f"but got {type(value).__name__} instead.")

    return value  # Return the value if the type is correct


def _sitexml_check_enum(value, enum_type, param_name="value", allow_none=False):
    """
    Checks if a given value is a valid member of an Enum.

    :param value: The value to check.
    :param enum_type: The Enum class to check against.
    :param param_name: Optional name of the parameter for error messages.
    :param allow_none: If True, allows None as a valid value.
    :raises ValueError: If the value is not a valid Enum member.
    :return: The validated value.
    """
    if allow_none and value is None:
        return value  # Allow None if specified
    
    if value not in enum_type:
        valid_values = [e for e in enum_type]  # Get all valid Enum names
        raise ValueError(f"\nInvalid value for '{param_name}'. Expected one of {valid_values}, but got '{value}'.")
    return value  # Return the value if it's valid

def _pretty_str(obj):
    return ", ".join(
        f"{key}='{value}'" for key, value in vars(obj).items() if value is not None
    )

#        if self.topologyA not in TopographySchemaA:
#            msg = ("topologyA must contain one of the following allowed values: ", ', '.join([x for x in TopographySchemaA]))
#            raise ValueError(msg)
