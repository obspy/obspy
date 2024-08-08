# -*- coding: utf-8 -*-
"""
obspy.core.event.base - Classes for handling event metadata
===========================================================
This module provides a class hierarchy to consistently handle event metadata.
This class hierarchy is closely modelled after the de-facto standard format
`QuakeML <https://quake.ethz.ch/quakeml/>`_.

.. figure:: /_images/Event.png

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import warnings

import numpy as np

from obspy.core.event.resourceid import ResourceIdentifier
from obspy.core.event.header import DataUsedWaveType, ATTRIBUTE_HAS_ERRORS
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict


class QuantityError(AttribDict):
    """
    Uncertainty information for a physical quantity.

    :type uncertainty: float
    :param uncertainty: Uncertainty as the absolute value of symmetric
        deviation from the main value.
    :type lower_uncertainty: float
    :param lower_uncertainty: Uncertainty as the absolute value of deviation
        from the main value towards smaller values.
    :type upper_uncertainty: float
    :param upper_uncertainty: Uncertainty as the absolute value of deviation
        from the main value towards larger values.
    :type confidence_level: float
    :param confidence_level: Confidence level of the uncertainty, given in
        percent (0-100).
    """
    defaults = {"uncertainty": None, "lower_uncertainty": None,
                "upper_uncertainty": None, "confidence_level": None}
    warn_on_non_default_key = True

    def __init__(self, uncertainty=None, lower_uncertainty=None,
                 upper_uncertainty=None, confidence_level=None):
        super(QuantityError, self).__init__()
        self.uncertainty = uncertainty
        self.lower_uncertainty = lower_uncertainty
        self.upper_uncertainty = upper_uncertainty
        self.confidence_level = confidence_level

    def __bool__(self):
        """
        Boolean testing for QuantityError.

        QuantityError evaluates ``True`` if any of the default fields is not
        ``None``. Setting non default fields raises also an UserWarning which
        is the reason we have to skip those lines in the doctest below.

        >>> err = QuantityError()
        >>> bool(err)
        False
        >>> err.custom_field = "spam"  # doctest: +SKIP
        >>> bool(err)
        False
        >>> err.uncertainty = 0.05
        >>> bool(err)
        True
        >>> del err.custom_field  # doctest: +SKIP
        >>> bool(err)
        True
        """
        return any([getattr(self, key) is not None for key in self.defaults])

    def __eq__(self, other):
        if other is None and not bool(self):
            return True
        return super(QuantityError, self).__eq__(other)


def _bool(value):
    """
    A custom bool() implementation that returns
    True for any value (including zero) of int and float,
    and for (empty) strings.
    """
    if value == 0 or isinstance(value, str):
        return True
    return bool(value)


def _event_type_class_factory(class_name, class_attributes=[],
                              class_contains=[]):
    """
    Class factory to unify the creation of all the types needed for the event
    handling in ObsPy.

    The types oftentimes share attributes and setting them manually every time
    is cumbersome, error-prone and hard to do consistently. The classes created
    with this method will inherit from
    :class:`~obspy.core.util.attribdict.AttribDict`.

    Usage to create a new class type:

    The created class will assure that any given (key, type) attribute pairs
    will always be of the given type and will attempt to convert any given
    value to the correct type and raise an error otherwise. This happens to
    values given during initialization as well as values set when the object
    has already been created. A useful type is Enum if you want to restrict
    the acceptable values.

        >>> from obspy.core.util import Enum
        >>> MyEnum = Enum(["a", "b", "c"])
        >>> class_attributes = [ \
                ("resource_id", ResourceIdentifier), \
                ("creation_info", CreationInfo), \
                ("some_letters", MyEnum), \
                ("some_error_quantity", float, ATTRIBUTE_HAS_ERRORS), \
                ("description", str)]

    Furthermore the class can contain lists of other objects. There is not much
    to it so far. Giving the name of the created class is mandatory.

        >>> class_contains = ["comments"]
        >>> TestEventClass = _event_type_class_factory("TestEventClass", \
                class_attributes=class_attributes, \
                class_contains=class_contains)
        >>> assert(TestEventClass.__name__ == "TestEventClass")

    Now the new class type can be used.

        >>> test_event = TestEventClass(resource_id="event/123456", \
                creation_info={"author": "obspy.org", "version": "0.1"})

    All given arguments will be converted to the right type upon setting them.

        >>> test_event.resource_id
        ResourceIdentifier(id="event/123456")
        >>> print(test_event.creation_info)
        CreationInfo(author='obspy.org', version='0.1')

    All others will be set to None.

        >>> assert(test_event.description is None)
        >>> assert(test_event.some_letters is None)

    If the resource_id attribute of the created class type is set, the object
    the ResourceIdentifier refers to will be the class instance.

        >>> assert(id(test_event) == \
            id(test_event.resource_id.get_referred_object()))

    They can be set later and will be converted to the appropriate type if
    possible.

        >>> test_event.description = 1
        >>> assert(test_event.description == "1")

    Trying to set with an inappropriate value will raise an error.

        >>> test_event.some_letters = "d" # doctest:+ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Setting attribute "some_letters" failed. ...

    If you pass ``ATTRIBUTE_HAS_ERRORS`` as the third tuple item for the
    class_attributes, a error (type
    :class:`~obspy.core.event.base.QuantityError`) will be be created that will
    be named like the attribute with "_errors" appended.

        >>> assert(hasattr(test_event, "some_error_quantity_errors"))
        >>> test_event.some_error_quantity_errors  # doctest: +ELLIPSIS
        QuantityError(...)
    """
    class AbstractEventType(AttribDict):
        # Keep the class attributes in a class level list for a manual property
        # implementation that works when inheriting from AttribDict.
        _properties = []
        for item in class_attributes:
            _properties.append((item[0], item[1]))
            if len(item) == 3 and item[2] == ATTRIBUTE_HAS_ERRORS:
                _properties.append((item[0] + "_errors", QuantityError))
        _property_keys = [_i[0] for _i in _properties]
        _property_dict = {}
        for key, value in _properties:
            _property_dict[key] = value
        _containers = class_contains
        warn_on_non_default_key = True
        defaults = dict.fromkeys(class_contains, [])
        defaults.update(dict.fromkeys(_property_keys, None))
        do_not_warn_on = ["extra"]

        def __init__(self, *args, **kwargs):
            # Make sure the args work as expected. Therefore any specified
            # arg will overwrite a potential kwarg, e.g. arg at position 0 will
            # overwrite kwargs class_attributes[0].
            for _i, item in enumerate(args):
                # Use the class_attributes list here because it is not yet
                # polluted be the error quantities.
                kwargs[class_attributes[_i][0]] = item
            # Set all property values to None or the kwarg value.
            for key, _ in self._properties:
                value = kwargs.get(key, None)
                # special handling for resource id
                if key == "resource_id":
                    if kwargs.get("force_resource_id", False):
                        if value is None:
                            value = ResourceIdentifier()
                setattr(self, key, value)
            # Containers currently are simple lists.
            for name in self._containers:
                setattr(self, name, list(kwargs.get(name, [])))
            # All errors are QuantityError. If they are not set yet, set them
            # now.
            for key, _ in self._properties:
                if key.endswith("_errors") and getattr(self, key) is None:
                    setattr(self, key, QuantityError())

        def clear(self):
            """
            Clear the class
            :return:
            """
            super(AbstractEventType, self).clear()
            self.__init__(force_resource_id=False)

        def __str__(self, force_one_line=False):
            """
            Fairly extensive in an attempt to cover several use cases. It is
            always possible to change it in the child class.
            """
            # Get the attribute and containers that are to be printed. Only not
            # None attributes and non-error attributes are printed. The errors
            # will appear behind the actual value.
            # We use custom _bool() for testing getattr() since we want to
            # print int and float values that are equal to zero and empty
            # strings.
            attributes = [_i for _i in self._property_keys if not
                          _i.endswith("_errors") and _bool(getattr(self, _i))]
            containers = [_i for _i in self._containers if
                          _bool(getattr(self, _i))]

            # Get the longest attribute/container name to print all of them
            # nicely aligned.
            max_length = max(max([len(_i) for _i in attributes])
                             if attributes else 0,
                             max([len(_i) for _i in containers])
                             if containers else 0) + 1

            ret_str = self.__class__.__name__

            # Case 1: Empty object.
            if not attributes and not containers:
                return ret_str + "()"

            def get_value_repr(key):
                value = getattr(self, key)
                if isinstance(value, str):
                    value = value
                repr_str = value.__repr__()
                # Print any associated errors.
                error_key = key + "_errors"
                if self.get(error_key, False):
                    err_items = sorted(getattr(self, error_key).items())
                    repr_str += " [%s]" % ', '.join(
                        sorted([str(k) + "=" + str(v) for k, v in err_items
                                if v is not None]))
                return repr_str

            # Case 2: Short representation for small objects. Will just print a
            # single line.
            if len(attributes) <= 3 and not containers or\
               force_one_line:
                att_strs = ["%s=%s" % (_i, get_value_repr(_i))
                            for _i in attributes if _bool(getattr(self, _i))]
                ret_str += "(%s)" % ", ".join(att_strs)
                return ret_str

            # Case 3: Verbose string representation for large object.
            if attributes:
                format_str = "%" + str(max_length) + "s: %s"
                att_strs = [format_str % (_i, get_value_repr(_i))
                            for _i in attributes if _bool(getattr(self, _i))]
                ret_str += "\n\t" + "\n\t".join(att_strs)

            # For the containers just print the number of elements in each.
            if containers:
                # Print delimiter only if there are attributes.
                if attributes:
                    ret_str += '\n\t' + '---------'.rjust(max_length + 5)
                element_str = "%" + str(max_length) + "s: %i Elements"
                ret_str += "\n\t" + \
                    "\n\t".join(
                        [element_str % (_i, len(getattr(self, _i)))
                         for _i in containers])
            return ret_str

        def _repr_pretty_(self, p, cycle):
            p.text(str(self))

        def copy(self):
            return copy.deepcopy(self)

        def __repr__(self):
            return self.__str__(force_one_line=True)

        def __bool__(self):
            # We use custom _bool() for testing getattr() since we want
            # zero valued int and float and empty string attributes to be True.
            if any([_bool(getattr(self, _i))
                    for _i in self._property_keys + self._containers]):
                return True
            return False

        def __eq__(self, other):
            """
            Two instances are considered equal if all attributes and all lists
            are identical.
            """
            # Looping should be quicker on average than a list comprehension
            # because only the first non-equal attribute will already return.
            for attrib in self._property_keys:
                if not hasattr(other, attrib) or \
                   (getattr(self, attrib) != getattr(other, attrib)):
                    return False
            for container in self._containers:
                if not hasattr(other, container) or \
                   (getattr(self, container) != getattr(other, container)):
                    return False
            return True

        def __ne__(self, other):
            return not self.__eq__(other)

        def __setattr__(self, name, value):
            """
            Custom property implementation that works if the class is
            inheriting from AttribDict.
            """
            # avoid type casting of 'extra' attribute, to make it possible to
            # control ordering of extra tags by using an OrderedDict for
            # 'extra'.
            if name == 'extra':
                dict.__setattr__(self, name, value)
                return
            # Pass to the parent method if not a custom property.
            if name not in self._property_dict.keys():
                AttribDict.__setattr__(self, name, value)
                return
            attrib_type = self._property_dict[name]
            # If the value is None or already the correct type just set it.
            if (value is not None) and (type(value) is not attrib_type):
                # If it is a dict, and the attrib_type is no dict, than all
                # values will be assumed to be keyword arguments.
                if isinstance(value, dict):
                    new_value = attrib_type(**value)
                else:
                    new_value = attrib_type(value)
                if new_value is None:
                    msg = 'Setting attribute "%s" failed. ' % (name)
                    msg += 'Value "%s" could not be converted to type "%s"' % \
                        (str(value), str(attrib_type))
                    raise ValueError(msg)
                value = new_value

            # Make sure all floats are finite - otherwise this is most
            # likely a user error.
            if attrib_type is float and value is not None:
                if not np.isfinite(value):
                    msg = "On %s object: Value '%s' for '%s' is " \
                          "not a finite floating point value." % (
                              type(self).__name__, str(value), name)

                    raise ValueError(msg)

            AttribDict.__setattr__(self, name, value)
            # if value is a resource id bind or unbind the resource_id
            if isinstance(value, ResourceIdentifier):
                if name == "resource_id":  # bind the resource_id to self
                    self.resource_id.set_referred_object(self, warn=False)
                else:  # else unbind to allow event scoping later
                    value._parent_key = None

    class AbstractEventTypeWithResourceID(AbstractEventType):
        def __init__(self, force_resource_id=True, *args, **kwargs):
            kwargs["force_resource_id"] = force_resource_id
            super(AbstractEventTypeWithResourceID, self).__init__(*args,
                                                                  **kwargs)

    if "resource_id" in [item[0] for item in class_attributes]:
        base_class = AbstractEventTypeWithResourceID
    else:
        base_class = AbstractEventType

    # Set the class type name.
    setattr(base_class, "__name__", class_name)
    return base_class


__CreationInfo = _event_type_class_factory(
    "__CreationInfo",
    class_attributes=[("agency_id", str),
                      ("agency_uri", ResourceIdentifier),
                      ("author", str),
                      ("author_uri", ResourceIdentifier),
                      ("creation_time", UTCDateTime),
                      ("version", str)])


class CreationInfo(__CreationInfo):
    """
    CreationInfo is used to describe creation metadata (author, version, and
    creation time) of a resource.

    :type agency_id: str, optional
    :param agency_id: Designation of agency that published a resource.
    :type agency_uri: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param agency_uri: Resource Identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param author_uri: Resource Identifier of the author of a resource.
    :type creation_time: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param creation_time: Time of creation of a resource.
    :type version: str, optional
    :param version: Version string of a resource

    >>> info = CreationInfo(author="obspy.org", version="0.0.1")
    >>> print(info)
    CreationInfo(author='obspy.org', version='0.0.1')

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__TimeWindow = _event_type_class_factory(
    "__TimeWindow",
    class_attributes=[("begin", float),
                      ("end", float),
                      ("reference", UTCDateTime)])


class TimeWindow(__TimeWindow):
    """
    Describes a time window for amplitude measurements, given by a central
    point in time, and points in time before and after this central point. Both
    points before and after may coincide with the central point.

    :type begin: float
    :param begin: Absolute value of duration of time interval before reference
        point in time window. The value may be zero, but not negative. Unit: s
    :type end: float
    :param end: Absolute value of duration of time interval after reference
        point in time window. The value may be zero, but not negative.  Unit: s
    :type reference: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param reference: Reference point in time ("central" point).

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__CompositeTime = _event_type_class_factory(
    "__CompositeTime",
    class_attributes=[("year", int, ATTRIBUTE_HAS_ERRORS),
                      ("month", int, ATTRIBUTE_HAS_ERRORS),
                      ("day", int, ATTRIBUTE_HAS_ERRORS),
                      ("hour", int, ATTRIBUTE_HAS_ERRORS),
                      ("minute", int, ATTRIBUTE_HAS_ERRORS),
                      ("second", float, ATTRIBUTE_HAS_ERRORS)])


class CompositeTime(__CompositeTime):
    """
    Focal times differ significantly in their precision. While focal times of
    instrumentally located earthquakes are estimated precisely down to seconds,
    historic events have only incomplete time descriptions. Sometimes, even
    contradictory information about the rupture time exist. The CompositeTime
    type allows for such complex descriptions. If the specification is given
    with no greater accuracy than days (i.e., no time components are given),
    the date refers to local time. However, if time components are given, they
    have to refer to UTC.

    :type year: int
    :param year: Year or range of years of the event's focal time.
    :type year_errors: :class:`~obspy.core.event.base.QuantityError`
    :param year_errors: AttribDict containing error quantities.
    :type month: int
    :param month: Month or range of months of the event’s focal time.
    :type month_errors: :class:`~obspy.core.event.base.QuantityError`
    :param month_errors: AttribDict containing error quantities.
    :type day: int
    :param day: Day or range of days of the event’s focal time.
    :type day_errors: :class:`~obspy.core.event.base.QuantityError`
    :param day_errors: AttribDict containing error quantities.
    :type hour: int
    :param hour: Hour or range of hours of the event’s focal time.
    :type hour_errors: :class:`~obspy.core.event.base.QuantityError`
    :param hour_errors: AttribDict containing error quantities.
    :type minute: int
    :param minute: Minute or range of minutes of the event’s focal time.
    :type minute_errors: :class:`~obspy.core.event.base.QuantityError`
    :param minute_errors: AttribDict containing error quantities.
    :type second: float
    :param second: Second and fraction of seconds or range of seconds with
        fraction of the event’s focal time.
    :type second_errors: :class:`~obspy.core.event.base.QuantityError`
    :param second_errors: AttribDict containing error quantities.

    >>> print(CompositeTime(2011, 1, 1))
    CompositeTime(year=2011, month=1, day=1)
    >>> # Can also be instantiated with the uncertainties.
    >>> print(CompositeTime(year=2011, year_errors={"uncertainty":1}))
    CompositeTime(year=2011 [uncertainty=1])

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Comment = _event_type_class_factory(
    "__Comment",
    class_attributes=[("text", str),
                      ("resource_id", ResourceIdentifier),
                      ("creation_info", CreationInfo)])


class Comment(__Comment):
    """
    Comment holds information on comments to a resource as well as author and
    creation time information.

    :type text: str
    :param text: Text of comment.
    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of comment.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: Creation info for the comment.

    >>> comment = Comment(text="Some comment")
    >>> print(comment)  # doctest:+ELLIPSIS
    Comment(text='Some comment', resource_id=ResourceIdentifier(...))
    >>> comment = Comment(text="Some comment", force_resource_id=False)
    >>> print(comment)
    Comment(text='Some comment')
    >>> comment.resource_id = "comments/obspy-comment-123456"
    >>> print(comment) # doctest:+ELLIPSIS
    Comment(text='Some comment', resource_id=ResourceIdentifier(...))
    >>> comment.creation_info = {"author": "obspy.org"}
    >>> print(comment.creation_info)
    CreationInfo(author='obspy.org')

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__WaveformStreamID = _event_type_class_factory(
    "__WaveformStreamID",
    class_attributes=[("network_code", str),
                      ("station_code", str),
                      ("channel_code", str),
                      ("location_code", str),
                      ("resource_uri", ResourceIdentifier)])


class WaveformStreamID(__WaveformStreamID):
    """
    Reference to a stream description in an inventory.

    This is mostly equivalent to the combination of networkCode, stationCode,
    locationCode, and channelCode. However, additional information, e. g.,
    sampling rate, can be referenced by the resourceURI. It is recommended to
    use resourceURI as a flexible, abstract, and unique stream ID that allows
    to describe different processing levels, or resampled/filtered products of
    the same initial stream, without violating the intrinsic meaning of the
    legacy identifiers (network, station, channel, and location codes).
    However, for operation in the context of legacy systems, the classical
    identifier components are supported.

    :type network_code: str
    :param network_code: Network code.
    :type station_code: str
    :param station_code: Station code.
    :type location_code: str, optional
    :param location_code: Location code.
    :type channel_code: str, optional
    :param channel_code: Channel code.
    :type resource_uri:
        :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_uri: Resource identifier for the waveform stream.
    :type seed_string: str, optional
    :param seed_string: Provides an alternative initialization way by passing a
        SEED waveform string in the form network.station.location.channel, e.g.
        BW.FUR..EHZ, which will be used to populate the WaveformStreamID's
        attributes.
        It will only be used if the network, station, location and channel
        keyword argument are ALL None.

    .. rubric:: Example

    >>> # Can be initialized with a SEED string or with individual components.
    >>> stream_id = WaveformStreamID(network_code="BW", station_code="FUR",
    ...                              location_code="", channel_code="EHZ")
    >>> print(stream_id) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    WaveformStreamID
          network_code: 'BW'
          station_code: 'FUR'
          channel_code: 'EHZ'
         location_code: ''
    >>> stream_id = WaveformStreamID(seed_string="BW.FUR..EHZ")
    >>> print(stream_id) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    WaveformStreamID
          network_code: 'BW'
          station_code: 'FUR'
          channel_code: 'EHZ'
         location_code: ''
    >>> # Can also return the SEED string.
    >>> print(stream_id.get_seed_string())
    BW.FUR..EHZ
    """
    def __init__(self, network_code=None, station_code=None,
                 location_code=None, channel_code=None, resource_uri=None,
                 seed_string=None):
        # Use the seed_string if it is given and everything else is not.
        if (seed_string is not None) and (network_code is None) and \
           (station_code is None) and (location_code is None) and \
           (channel_code is None):
            try:
                network_code, station_code, location_code, channel_code = \
                    seed_string.split('.')
            except ValueError:
                warnings.warn("In WaveformStreamID.__init__(): " +
                              "seed_string was given but could not be parsed")
                pass
            if not any([bool(_i) for _i in [network_code, station_code,
                                            location_code, channel_code]]):
                network_code, station_code, location_code, channel_code = \
                    4 * [None]
        super(WaveformStreamID, self).__init__(network_code=network_code,
                                               station_code=station_code,
                                               location_code=location_code,
                                               channel_code=channel_code,
                                               resource_uri=resource_uri)

    def get_seed_string(self):
        """
        Return the seed string representation.

        The seed string is of the form:
            network.station.location.channel
        """
        return "%s.%s.%s.%s" % (
            self.network_code if self.network_code else "",
            self.station_code if self.station_code else "",
            self.location_code if self.location_code else "",
            self.channel_code if self.channel_code else "")

    id = property(get_seed_string)


__ConfidenceEllipsoid = _event_type_class_factory(
    "__ConfidenceEllipsoid",
    class_attributes=[("semi_major_axis_length", float),
                      ("semi_minor_axis_length", float),
                      ("semi_intermediate_axis_length", float),
                      ("major_axis_plunge", float),
                      ("major_axis_azimuth", float),
                      ("major_axis_rotation", float)])


class ConfidenceEllipsoid(__ConfidenceEllipsoid):
    """
    This class represents a description of the location uncertainty as a
    confidence ellipsoid with arbitrary orientation in space. See the QuakeML
    documentation for the full details

    :param semi_major_axis_length: Largest uncertainty, corresponding to the
        semi-major axis of the confidence ellipsoid. Unit: m
    :param semi_minor_axis_length: Smallest uncertainty, corresponding to the
        semi-minor axis of the confidence ellipsoid. Unit: m
    :param semi_intermediate_axis_length: Uncertainty in direction orthogonal
        to major and minor axes of the confidence ellipsoid. Unit: m
    :param major_axis_plunge: Plunge angle of major axis of confidence
        ellipsoid. Corresponds to Tait-Bryan angle φ. Unit: deg
    :param major_axis_azimuth: Azimuth angle of major axis of confidence
        ellipsoid. Corresponds to Tait-Bryan angle ψ. Unit: deg
    :param major_axis_rotation: This angle describes a rotation about the
        confidence ellipsoid’s major axis which is required to define the
        direction of the ellipsoid’s minor axis. Corresponds to Tait-Bryan
        angle θ.
        Unit: deg

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__DataUsed = _event_type_class_factory(
    "__DataUsed",
    class_attributes=[("wave_type", DataUsedWaveType),
                      ("station_count", int),
                      ("component_count", int),
                      ("shortest_period", float),
                      ("longest_period", float)])


class DataUsed(__DataUsed):
    """
    The DataUsed class describes the type of data that has been used for a
    moment-tensor inversion.

    :type wave_type: str
    :param wave_type: Type of waveform data.
        See :class:`~obspy.core.event.header.DataUsedWaveType` for allowed
        values.
    :type station_count: int, optional
    :param station_count: Number of stations that have contributed data of the
        type given in wave_type.
    :type component_count: int, optional
    :param component_count: Number of data components of the type given in
        wave_type.
    :type shortest_period: float, optional
    :param shortest_period: Shortest period present in data. Unit: s
    :type longest_period: float, optional
    :param longest_period: Longest period present in data. Unit: s

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
