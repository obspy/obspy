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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import collections
import copy
import inspect
import re
import warnings
import weakref
from copy import deepcopy
from uuid import uuid4

from obspy.core.event.header import DataUsedWaveType, ATTRIBUTE_HAS_ERRORS
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.util.decorator import deprecated


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
        ``None``.

        >>> err = QuantityError()
        >>> bool(err)
        False
        >>> err.custom_field = "spam"
        >>> bool(err)
        False
        >>> err.uncertainty = 0.05
        >>> bool(err)
        True
        >>> del err.custom_field
        >>> bool(err)
        True
        """
        return any([getattr(self, key) is not None for key in self.defaults])

    # Python 2 compatibility
    __nonzero__ = __bool__


def _bool(value):
    """
    A custom bool() implementation that returns
    True for any value (including zero) of int and float,
    and for (empty) strings.
    """
    if value == 0 or isinstance(value, (str, native_str)):
        return True
    return bool(value)


def _event_type_class_factory(class_name, class_attributes=[],
                              class_contains=[]):
    """
    Class factory to unify the creation of all the types needed for the event
    handling in ObsPy.

    The types oftentimes share attributes and setting them manually every time
    is cumbersome, error-prone and hard to do consistently. The classes created
    with this method will inherit from :class:`~obspy.core.util.AttribDict`.

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
                if isinstance(value, (str, native_str)):
                    value = native_str(value)
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

        # called for bool on PY2
        def __nonzero__(self):
            return self.__bool__()

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
            AttribDict.__setattr__(self, name, value)
            # If "name" is resource_id and value is not None, set the referred
            # object of the ResourceIdentifier to self.
            if name == "resource_id" and value is not None:
                self.resource_id.set_referred_object(self)

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
    setattr(base_class, "__name__", native_str(class_name))
    return base_class


class ResourceIdentifier(object):
    """
    Unique identifier of any resource so it can be referred to.

    In QuakeML many elements and types can have a unique id that other elements
    use to refer to it. This is called a ResourceIdentifier and it is used for
    the same purpose in the obspy.core.event classes.

    In QuakeML it has to be of the following regex form::

        (smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\.\*\(\)_~']
        [\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*

    e.g.

    * ``smi:sub.website.org/event/12345678``
    * ``quakeml:google.org/pick/unique_pick_id``

    smi stands for "seismological meta-information".

    In this class it can be any hashable object, e.g. most immutable objects
    like numbers and strings.

    :type id: str, optional
    :param id: A unique identifier of the element it refers to. It is
        not verified, that it actually is unique. The user has to take care of
        that. If no resource_id is given, uuid.uuid4() will be used to
        create one which assures uniqueness within one Python run.
        If no fixed id is provided, the ID will be built from prefix
        and a random uuid hash. The random hash part can be regenerated by the
        referred object automatically if it gets changed.
    :type prefix: str, optional
    :param prefix: An optional identifier that will be put in front of any
        automatically created resource id. The prefix will only have an effect
        if `id` is not specified (for a fixed ID string). Makes automatically
        generated resource ids more reasonable. By default "smi:local" is used
        which ensures a QuakeML conform resource identifier.
    :type referred_object: Python object, optional
    :param referred_object: The object this instance refers to. All instances
        created with the same resource_id will be able to access the object as
        long as at least one instance actual has a reference to it.

    .. rubric:: General Usage

    >>> ResourceIdentifier('2012-04-11--385392')
    ResourceIdentifier(id="2012-04-11--385392")
    >>> # If 'id' is not specified it will be generated automatically.
    >>> ResourceIdentifier()  # doctest: +ELLIPSIS
    ResourceIdentifier(id="smi:local/...")
    >>> # Supplying a prefix will simply prefix the automatically generated ID
    >>> ResourceIdentifier(prefix='event')  # doctest: +ELLIPSIS
    ResourceIdentifier(id="event/...")

    ResourceIdentifiers can, and oftentimes should, carry a reference to the
    object they refer to. This is a weak reference which means that if the
    object get deleted or runs out of scope, e.g. gets garbage collected, the
    reference will cease to exist.

    >>> from obspy.core.event import Event
    >>> event = Event()
    >>> import sys
    >>> ref_count = sys.getrefcount(event)
    >>> res_id = ResourceIdentifier(referred_object=event)
    >>> # The reference does not changed the reference count of the object.
    >>> print(ref_count == sys.getrefcount(event))
    True
    >>> # It actually is the same object.
    >>> print(event is res_id.get_referred_object())
    True
    >>> # Deleting it, or letting the garbage collector handle the object will
    >>> # invalidate the reference.
    >>> del event
    >>> print(res_id.get_referred_object())
    None

    The most powerful ability (and reason why one would want to use a resource
    identifier class in the first place) is that once a ResourceIdentifier with
    an attached referred object has been created, any other ResourceIdentifier
    instances with the same ID can retrieve that object. This works
    across all ResourceIdentifiers that have been instantiated within one
    Python run.
    This enables, e.g. the resource references between the different QuakeML
    elements to work in a rather natural way.

    >>> event_object = Event()
    >>> obj_id = id(event_object)
    >>> res_id = "obspy.org/event/test"
    >>> ref_a = ResourceIdentifier(res_id)
    >>> # The object is refers to cannot be found yet. Because no instance that
    >>> # an attached object has been created so far.
    >>> print(ref_a.get_referred_object())
    None
    >>> # This instance has an attached object.
    >>> ref_b = ResourceIdentifier(res_id, referred_object=event_object)
    >>> ref_c = ResourceIdentifier(res_id)
    >>> # All ResourceIdentifiers will refer to the same object.
    >>> assert(id(ref_a.get_referred_object()) == obj_id)
    >>> assert(id(ref_b.get_referred_object()) == obj_id)
    >>> assert(id(ref_c.get_referred_object()) == obj_id)

    The id can be converted to a valid QuakeML ResourceIdentifier by calling
    the convert_id_to_quakeml_uri() method. The resulting id will be of the
    form::

        smi:authority_id/prefix/id

    >>> res_id = ResourceIdentifier(prefix='origin')
    >>> res_id.convert_id_to_quakeml_uri(authority_id="obspy.org")
    >>> res_id # doctest:+ELLIPSIS
    ResourceIdentifier(id="smi:obspy.org/origin/...")
    >>> res_id = ResourceIdentifier(id='foo')
    >>> res_id.convert_id_to_quakeml_uri()
    >>> res_id
    ResourceIdentifier(id="smi:local/foo")
    >>> # A good way to create a QuakeML compatibly ResourceIdentifier from
    >>> # scratch is
    >>> res_id = ResourceIdentifier(prefix='pick')
    >>> res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
    >>> res_id  # doctest:+ELLIPSIS
    ResourceIdentifier(id="smi:obspy.org/pick/...")
    >>> # If the given ID is already a valid QuakeML
    >>> # ResourceIdentifier, nothing will happen.
    >>> res_id = ResourceIdentifier('smi:test.org/subdir/id')
    >>> res_id
    ResourceIdentifier(id="smi:test.org/subdir/id")
    >>> res_id.convert_id_to_quakeml_uri()
    >>> res_id
    ResourceIdentifier(id="smi:test.org/subdir/id")

    ResourceIdentifiers are considered identical if the IDs are
    the same.

    >>> # Create two different resource identifiers.
    >>> res_id_1 = ResourceIdentifier()
    >>> res_id_2 = ResourceIdentifier()
    >>> assert(res_id_1 != res_id_2)
    >>> # Equalize the IDs. NEVER do this. This just an example.
    >>> res_id_2.id = res_id_1.id = "smi:local/abcde"
    >>> assert(res_id_1 == res_id_2)

    ResourceIdentifier instances can be used as dictionary keys.

    >>> dictionary = {}
    >>> res_id = ResourceIdentifier(id="foo")
    >>> dictionary[res_id] = "bar1"
    >>> # The same ID can still be used as a key.
    >>> dictionary["foo"] = "bar2"
    >>> items = sorted(dictionary.items(), key=lambda kv: kv[1])
    >>> for k, v in items:  # doctest: +ELLIPSIS
    ...     print(repr(k), v)
    ResourceIdentifier(id="foo") bar1
    ...'foo' bar2
    """
    # Class (not instance) attribute that keeps track of all resource
    # identifier throughout one Python run. Will only store weak references and
    # therefore does not interfere with the garbage collection.
    # DO NOT CHANGE THIS FROM OUTSIDE THE CLASS.
    __resource_id_weak_dict = weakref.WeakValueDictionary()
    # Use an additional dictionary to track all resource ids.
    __resource_id_tracker = collections.defaultdict(int)

    def __init__(self, id=None, prefix="smi:local",
                 referred_object=None):
        # Create a resource id if None is given and possibly use a prefix.
        if id is None:
            self.fixed = False
            self._prefix = prefix
            self._uuid = str(uuid4())
        elif isinstance(id, ResourceIdentifier):
            self.__dict__.update(id.__dict__)
            return
        else:
            self.fixed = True
            self.id = id
        # Append the referred object in case one is given to the class level
        # reference dictionary.
        if referred_object is not None:
            self.set_referred_object(referred_object)

        # Increment the counter for the current resource id.
        ResourceIdentifier.__resource_id_tracker[self.id] += 1

    def __del__(self):
        if self.id not in ResourceIdentifier.__resource_id_tracker:
            return
        # Decrement the resource id counter.
        ResourceIdentifier.__resource_id_tracker[self.id] -= 1
        # If below or equal to zero, delete it and also delete it from the weak
        # value dictionary.
        if ResourceIdentifier.__resource_id_tracker[self.id] <= 0:
            del ResourceIdentifier.__resource_id_tracker[self.id]
            try:
                del ResourceIdentifier.__resource_id_weak_dict[self.id]
            except KeyError:
                pass

    @deprecated("Method 'getReferredObject' was renamed to "
                "'get_referred_object'. Use that instead.")  # noqa
    def getReferredObject(self):
        return self.get_referred_object()

    def get_referred_object(self):
        """
        Returns the object associated with the resource identifier.

        This works as long as at least one ResourceIdentifier with the same
        ID as this instance has an associate object.

        Will return None if no object could be found.
        """
        try:
            return ResourceIdentifier.__resource_id_weak_dict[self.id]
        except KeyError:
            return None

    @deprecated("Method 'setReferredObject' was renamed to "
                "'set_referred_object'. Use that instead.")  # noqa
    def setReferredObject(self, referred_object):
        return self.set_referred_object(referred_object)

    def set_referred_object(self, referred_object):
        """
        Sets the object the ResourceIdentifier refers to.

        If it already a weak reference it will be used, otherwise one will be
        created. If the object is None, None will be set.

        Will also append self again to the global class level reference list so
        everything stays consistent.
        """
        # If it does not yet exists simply set it.
        if self.id not in ResourceIdentifier.__resource_id_weak_dict:
            ResourceIdentifier.__resource_id_weak_dict[self.id] = \
                referred_object
            return
        # Otherwise check if the existing element the same as the new one. If
        # it is do nothing, otherwise raise a warning and set the new object as
        # the referred object.
        if ResourceIdentifier.__resource_id_weak_dict[self.id] == \
                referred_object:
            return
        msg = "The resource identifier '%s' already exists and points to " + \
              "another object: '%s'." +\
              "It will now point to the object referred to by the new " + \
              "resource identifier."
        msg = msg % (
            self.id,
            repr(ResourceIdentifier.__resource_id_weak_dict[self.id]))
        # Always raise the warning!
        warnings.warn_explicit(msg, UserWarning, __file__,
                               inspect.currentframe().f_back.f_lineno)
        ResourceIdentifier.__resource_id_weak_dict[self.id] = \
            referred_object

    @deprecated("Method 'convertIDToQuakeMLURI' was renamed to "
                "'convert_id_to_quakeml_uri'. Use that instead.")  # noqa
    def convertIDToQuakeMLURI(self, authority_id="local"):
        return self.convert_id_to_quakeml_uri(authority_id=authority_id)

    def convert_id_to_quakeml_uri(self, authority_id="local"):
        """
        Converts the current ID to a valid QuakeML URI.

        Only an invalid QuakeML ResourceIdentifier string it will be converted
        to a valid one.  Otherwise nothing will happen but after calling this
        method the user can be sure that the ID is a valid QuakeML URI.

        The resulting ID will be of the form
            smi:authority_id/prefix/resource_id

        :type authority_id: str, optional
        :param authority_id: The base url of the resulting string. Defaults to
            ``"local"``.
        """
        self.id = self.get_quakeml_uri(authority_id=authority_id)

    @deprecated("Method 'getQuakeMLURI' was renamed to "
                "'get_quakeml_uri'. Use that instead.")  # noqa
    def getQuakeMLURI(self, authority_id="local"):
        return self.get_quakeml_uri(authority_id=authority_id)

    def get_quakeml_uri(self, authority_id="local"):
        """
        Returns the ID as a valid QuakeML URI if possible. Does not
        change the ID itself.

        >>> res_id = ResourceIdentifier("some_id")
        >>> print(res_id.get_quakeml_uri())
        smi:local/some_id
        >>> # Did not change the actual resource id.
        >>> print(res_id.id)
        some_id
        """
        id = self.id
        if str(id).strip() == "":
            id = str(uuid4())

        regex = r"^(smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\." + \
                r"\*\(\)_~'][\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*$"
        result = re.match(regex, str(id))
        if result is not None:
            return id
        id = 'smi:%s/%s' % (authority_id, str(id))
        # Check once again just to be sure no weird symbols are stored in the
        # ID.
        result = re.match(regex, id)
        if result is None:
            msg = "Failed to create a valid QuakeML ResourceIdentifier."
            raise ValueError(msg)
        return id

    def copy(self):
        """
        Returns a copy of the ResourceIdentifier.

        >>> res_id = ResourceIdentifier()
        >>> res_id_2 = res_id.copy()
        >>> print(res_id is res_id_2)
        False
        >>> print(res_id == res_id_2)
        True
        """
        return deepcopy(self)

    @property
    def id(self):
        """
        Unique identifier of the current instance.
        """
        if self.fixed:
            return self.__dict__.get("id")
        else:
            id = self.prefix
            if not id.endswith("/"):
                id += "/"
            id += self.uuid
            return id

    @id.deleter
    def id(self):
        msg = "The resource id cannot be deleted."
        raise Exception(msg)

    @id.setter
    def id(self, value):
        self.fixed = True
        # XXX: no idea why I had to add bytes for PY2 here
        if not isinstance(value, (str, bytes)):
            msg = "attribute id needs to be a string."
            raise TypeError(msg)
        self.__dict__["id"] = value

    @property
    def prefix(self):
        return self._prefix

    @prefix.deleter
    def prefix(self):
        self._prefix = ""

    @prefix.setter
    def prefix(self, value):
        if not isinstance(value, (str, native_str)):
            msg = "prefix id needs to be a string."
            raise TypeError(msg)
        self._prefix = value

    @property
    def uuid(self):
        return self._uuid

    @uuid.deleter
    def uuid(self):
        """
        Deleting is uuid hash is forbidden and will not work.
        """
        msg = "The uuid cannot be deleted."
        raise Exception(msg)

    @uuid.setter
    def uuid(self, value):  # @UnusedVariable
        """
        Setting is uuid hash is forbidden and will not work.
        """
        msg = "The uuid cannot be set manually."
        raise Exception(msg)

    @property
    def resource_id(self):
        return self.id

    @resource_id.deleter
    def resource_id(self):
        del self.id

    @resource_id.setter
    def resource_id(self, value):
        self.id = value

    def __str__(self):
        return self.id

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return 'ResourceIdentifier(id="%s")' % self.id

    def __eq__(self, other):
        if self.id == other:
            return True
        if not isinstance(other, ResourceIdentifier):
            return False
        if self.id == other.id:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """
        Uses the same hash as the resource id. This means that class instances
        can be used in dictionaries and other hashed types.

        Both the object and it's id can still be independently used as
        dictionary keys.
        """
        # "Salt" the hash with a string so the hash of the object and a
        # string identical to the id can both be used as individual
        # dictionary keys.
        return hash("RESOURCE_ID") + self.id.__hash__()

    def regenerate_uuid(self):
        """
        Regenerates the uuid part of the ID. Does nothing for resource
        identifiers with a user-set, fixed id.
        """
        self._uuid = str(uuid4())


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
    :type agency_uri: :class:`~obspy.core.event.base.ResourceIdentifier`,
        optional
    :param agency_uri: Resource Identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: :class:`~obspy.core.event.base.ResourceIdentifier`,
        optional
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
    :type resource_id: :class:`~obspy.core.event.base.ResourceIdentifier`,
        optional
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
    :type resource_uri: :class:`~obspy.core.event.base.ResourceIdentifier`,
        optional
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

    @deprecated(
        "'getSEEDString' has been renamed to "  # noqa
        "'get_seed_string'. Use that instead.")
    def getSEEDString(self, *args, **kwargs):
        '''
        DEPRECATED: 'getSEEDString' has been renamed to
        'get_seed_string'. Use that instead.
        '''
        return self.get_seed_string(*args, **kwargs)

    def get_seed_string(self):
        return "%s.%s.%s.%s" % (
            self.network_code if self.network_code else "",
            self.station_code if self.station_code else "",
            self.location_code if self.location_code else "",
            self.channel_code if self.channel_code else "")


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
