# -*- coding: utf-8 -*-
"""

This module provides enumerations defined in the
SiteXML schema and other helper functions.

:copyright:
   ORFEUS, 2026
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy

from obspy.core.event import ResourceIdentifier
from obspy.core.util import Enum
from obspy.core.util.base import ComparingObject

from collections.abc import Iterable


# SiteXML Specific Exceptions

class SiteXMLError(Exception):
    """
    Base class for SiteXML-specific exceptions.
    """


class SiteXMLValidationError(SiteXMLError, ValueError):
    """
    Raised when SiteXML content fails schema or structural validation.
    """


class SiteXMLImportError(SiteXMLError, ValueError):
    """
    Raised when SiteXML metadata imports cannot be completed.
    """


class SiteXMLIOError(SiteXMLError, OSError):
    """
    Raised when SiteXML-related input paths or files cannot be accessed.
    """


class BaseNode(ComparingObject):
    """
    The parent class for SERASite, SiteDescription, Analysis etc classes.
    """
    def copy(self):
        """
        Returns a deepcopy of the object.

        :rtype: same class as original object
        :return: Copy of current object.

        .. rubric:: Examples

        1. Create a station object and copy it

            >>> from obspy.io.sitexml.sitexml import read_sitexml
            >>> site = read_sitexml("site.xml")  # doctest: +SKIP
            >>> site2 = site.copy()  # doctest: +SKIP

           The two objects are not the same:

            >>> site is site2  # doctest: +SKIP

           But they have equal data (before applying further processing):

            >>> site == site2  # doctest: +SKIP

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``site3`` would also change the contents of
           ``site``.

            >>> site3 = site  # doctest: +SKIP
            >>> site is site3  # doctest: +SKIP
            >>> site == site3  # doctest: +SKIP
        """
        return copy.deepcopy(self)


TopographySchemaA = Enum([
    "T1",
    "T2",
    "T3",
    "T4",
])
"""
Formal topographic/terrain classification of a site.

**Schema A** is the topographic classification scheme of the
**Italian Code**.
Allowed values are:

* ``"T1"`` : Flat surface, isolated slopes and cliffs with average slop
  angle
* ``"T2"`` : Slopes with average slope angle i>15
* ``"T3"`` : Ridges with crest width significantly less than the base
  width and average slope angle 15
* ``"T4"`` : Ridges with crest width significantly less than the base
  width and average slope angle i>30
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
Formal topographic/terrain classification of a site.

**Schema B** is the terrain classification scheme proposed by
**Burjanek et al. (2014)**. For the precise definition of the
allowed values refer to `SERA Deliverable 7.1, Appendix I.
<https://www.itsak.gr/SiteXML/SERA_D7.1_Standard-for-site-condition-metadata.pdf>`_

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
Qualitative landform descriptor in the QuakeML-STC-derived site morphology
group.
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
Ground type according to Eurocode 8 (EC8 § 3.1.2, Table 3.1).

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
Method used for the estimation of the resonance frequency, f0, of a site.

.. note::
    Required by **EGD (European Geocharacterization Database)**
    when calculating the EGD specific resonance frequency quality index.
    For more information refer to `SERA Deliverable 7.1, Appendix II.
    <https://www.itsak.gr/SiteXML/SERA_D7.1_Standard-for-site-condition-metadata.pdf>`_

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
Method used to estimate the S-wave velocity profile and Vs30.

.. note::
    Required by **EGD (European Geocharacterization Database)**
    when calculating the EGD specific Vs30 quality index. For
    more information refer to `SERA Deliverable 7.1, Appendix IV.
    <https://www.itsak.gr/SiteXML/SERA_D7.1_Standard-for-site-condition-metadata.pdf>`_

Vs30 is the average shear-wave velocity in the upper 30 meters of the
soil column. Allowed values are:

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

Vs30MethodCombined = Enum([
    "1.0",
    "1.2",
])
"""
Whether multiple methods were combined to estimate Vs30.

.. note::
    Required by **EGD (European Geocharacterization Database)**
    for calculating the EGD specific Vs30 quality index. For
    more information refer to `SERA Deliverable 7.1, Appendix III & IV.
    <https://www.itsak.gr/SiteXML/SERA_D7.1_Standard-for-site-condition-metadata.pdf>`_

Allowed values are:

* ``"1.0"`` : if only one method has been used to estimate the Vs30 value
* ``"1.2"`` : if a combination of two or more methods has been applied to
  estimate the Vs30 value
"""

Vs30ManualIndex = Enum([
    "0.2",
    "0.4",
    "0.8",
    "1.0",
])
"""
Qualitative factor regarding the maximum Vs measurement depth.

.. note::
    Required by **EGD (European Geocharacterization Database)**
    for calculating the EGD specific Vs30 quality index.

This depth is commonly compared with the EC8 engineering bedrock depth,
where Vs >= 800 m/s.

The reasoning for introducing this index and description of its values is
provided in `SERA Deliverable 7.1, Appendix III.
<https://www.itsak.gr/SiteXML/SERA_D7.1_Standard-for-site-condition-metadata.pdf>`_

* ``"0.2"`` : Unknown/partly unknown stratigraphy
* ``"0.4"`` : Maximum depth of Vs measurements < 10m
* ``"0.8"`` : Maximum depth of Vs measurements 10-30m
* ``"1.0"`` : Maximum depth of Vs measurements > 30m
"""


def _pretty_str(obj):
    """
    Return a compact representation of non-empty public object attributes.

    :rtype: str
    """
    return ", ".join(
        f"{key}='{value}'" for key, value in vars(obj).items()
        if value is not None
    )


# Setters and getters for class attributes with validation

def _enum_property(attr_name, enum_type):
    """
    Method to produce getter/setter functions
    and validate enum type values.

    :rtype: property
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None or value in enum_type:
            setattr(self, private_name, value)
        else:
            valid_values = [e for e in enum_type]
            raise SiteXMLValidationError(
                f"\nInvalid value for '{attr_name}'. \
                    Expected one of {valid_values}, but got '{value}'."
            )
    return property(getter, setter)


def _enum_list_property(attr_name, enum_type, allow_none=True):
    """
    Validates an iterable of enum entries and stores canonical strings.

    - Accepts strings (case-insensitive) that match enum keys, and returns the
      canonical enum value string (original casing from Enum definition).
    - Stores a plain Python list for deepcopy safety.

    :rtype: property
    """

    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def _eval_enum(x):
        if not isinstance(x, str):
            raise SiteXMLValidationError(
                f"{attr_name} items must be strings, got {type(x).__name__}"
            )
        try:
            # Enum.get() lowercases internally and returns canonical string
            return enum_type.get(x)
        except KeyError:
            raise SiteXMLValidationError(
                f"Invalid {attr_name} entry {x!r}. "
                f"Allowed: {enum_type.values()}"
            )

    class _EnumList(list):
        def __init__(self, values=()):
            super().__init__(_eval_enum(value) for value in values)

        def append(self, value):
            super().append(_eval_enum(value))

        def insert(self, index, value):
            super().insert(index, _eval_enum(value))

        def extend(self, values):
            super().extend(_eval_enum(value) for value in values)

        def __setitem__(self, index, value):
            if isinstance(index, slice):
                super().__setitem__(index,
                                    [_eval_enum(item) for item in value])
            else:
                super().__setitem__(index, _eval_enum(value))

    def setter(self, values):
        if values is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise SiteXMLValidationError(f"{attr_name} cannot be None")

        if (not isinstance(values, Iterable)
                or isinstance(values, (str, bytes))):
            raise SiteXMLValidationError(
                f"{attr_name} must be an iterable of strings"
            )

        setattr(self, private_name, _EnumList(values))

    return property(getter, setter)


def _scalar_property(attr_name, value_type=None,
                     allow_none=True, allow_empty=True):
    """
    Creates a property for scalar values with optional requiredness checks.

    If ``value_type`` is not provided, values are stored unchanged. This keeps
    resource identifier fields type-flexible until their exact API type is
    reviewed separately.

    :rtype: property
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise SiteXMLValidationError(f"{attr_name} is required.")

        if isinstance(value, str) and not allow_empty and not value.strip():
            raise SiteXMLValidationError(f"{attr_name} cannot be empty.")

        if value_type is not None and not isinstance(value, value_type):
            try:
                value = value_type(value)
            except Exception as e:
                raise SiteXMLValidationError(
                    f"Could not convert {value} "
                    f"to {value_type.__name__}: {e}"
                )

        setattr(self, private_name, value)

    return property(getter, setter)


def _resource_id_property(attr_name, allow_none=True, allow_empty=True):
    """
    Creates a property for SiteXML resource identifier fields.

    The SiteXML API stores resource identifiers internally as plain strings.
    ``ResourceIdentifier`` inputs are accepted as a convenience and are
    normalized to their ``.id`` string value on assignment.

    :rtype: property
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise SiteXMLValidationError(f"{attr_name} is required.")

        if isinstance(value, ResourceIdentifier):
            value = value.id

        if not isinstance(value, str):
            raise SiteXMLValidationError(
                f"{attr_name} must be a string or ResourceIdentifier."
            )

        if not allow_empty and not value.strip():
            raise SiteXMLValidationError(f"{attr_name} cannot be empty.")

        setattr(self, private_name, value)

    return property(getter, setter)


def _wrapped_property(attr_name, wrapper_type, allow_none=True):
    """
    Method to produce getter/setter functions
    and wrap argument values into the appropriate type.

    :param attr_name: name of the attribute
    :param wrapper_type: class used to wrap each element
    :param allow_none: whether None is allowed
    :rtype: property
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise SiteXMLValidationError(f"{attr_name} is required.")
        if isinstance(value, wrapper_type):
            setattr(self, private_name, value)
        else:
            try:
                setattr(self, private_name, wrapper_type(value))
            except Exception as e:
                raise SiteXMLValidationError(
                    f"Could not convert {value} "
                    f"to {wrapper_type.__name__}: {e}"
                )

    return property(getter, setter)


def _wrapped_list_property(attr_name, wrapper_type, allow_none=True):
    """
    Creates a property that wraps iterable elements into wrapper_type.

    :param attr_name: name of the attribute
    :param wrapper_type: class used to wrap each element
    :param allow_none: whether None is allowed
    :rtype: property
    """
    private_name = f"_{attr_name}"

    def getter(self):
        return getattr(self, private_name)

    def setter(self, values):
        if values is None:
            if allow_none:
                setattr(self, private_name, None)
                return
            raise SiteXMLValidationError(f"{attr_name} cannot be None")

        if (not isinstance(values, Iterable)
                or isinstance(values, (str, bytes))):
            raise SiteXMLValidationError(f"{attr_name} must be an iterable")

        wrapped_items = []
        for v in values:
            if isinstance(v, wrapper_type):
                wrapped_items.append(v)
            else:
                try:
                    wrapped_items.append(wrapper_type(v))
                except Exception as e:
                    raise SiteXMLValidationError(
                        f"Could not convert element {v} to "
                        f"{wrapper_type.__name__}: {e}"
                    )

        setattr(self, private_name, wrapped_items)

    return property(getter, setter)


# TABULAR DATAFRAME UTILITIES

def _require_dataframe_columns(df, columns, context):
    """
    Raise if required dataframe columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SiteXMLImportError(
            f"{context} is missing required column(s): "
            + ", ".join(missing)
        )


def _read_cell(df_row, argument, indicator=None):
    """
    Return a non-empty cell value, optionally using an indicator prefix.

    :rtype: object or None
    """

    if indicator:
        if indicator + "_" + argument in df_row and \
                not _empty_value(df_row[indicator + "_" + argument]):
            return df_row[indicator + "_" + argument]
    else:
        if argument in df_row and \
                not _empty_value(df_row[argument]):
            return df_row[argument]

    return None


def _empty_value(value):
    """
    Return whether a tabular cell should be treated as missing.

    :rtype: bool
    """
    if value is None:
        return True
    if value.__class__.__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False

    return False


# Helper function for validating and splitting station codes

def _split_station_code(value):
    """
    Split a ``network.station`` code into FDSN network and station codes.

    :rtype: tuple[str, str]
    """
    message = (
        "station_code must use 'network.station' notation with a "
        "2 character uppercase alphanumeric FDSN network code and a "
        "3-5 character uppercase alphanumeric station code"
    )
    if not isinstance(value, str):
        raise SiteXMLValidationError("station_code must be a string or None")
    if value.count(".") != 1 or any(char.isspace() for char in value):
        raise SiteXMLValidationError(message)
    network_code, station_code = value.split(".")
    if len(network_code) != 2 or \
            not network_code.isascii() or \
            not network_code.isalnum() or \
            network_code != network_code.upper() or \
            not 3 <= len(station_code) <= 5 or \
            not station_code.isascii() or \
            not station_code.isalnum() or \
            station_code != station_code.upper():
        raise SiteXMLValidationError(message)
    return network_code, station_code
