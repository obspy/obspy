# -*- coding: utf-8 -*-
"""
obspy.core.event - Classes for handling event metadata
======================================================
This module provides a class hierarchy to consistently handle event metadata.
This class hierarchy is closely modelled after the de-facto standard
format `QuakeML <https://quake.ethz.ch/quakeml/>`_.

.. figure:: /_images/Event.png

.. note::

    For handling additional information not covered by the QuakeML standard and
    how to output it to QuakeML see the :ref:`ObsPy Tutorial <quakeml-extra>`.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str
from future import standard_library
with standard_library.hooks():
    import urllib.request

from obspy.core.event_header import PickOnset, PickPolarity, EvaluationMode, \
    EvaluationStatus, OriginUncertaintyDescription, OriginDepthType, \
    EventDescriptionType, EventType, EventTypeCertainty, OriginType, \
    AmplitudeCategory, AmplitudeUnit, DataUsedWaveType, MTInversionType, \
    SourceTimeFunctionType, MomentTensorCategory
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import uncompressFile, _readFromPlugin, \
    NamedTemporaryFile, AttribDict
from obspy.core.util.decorator import map_example_filename
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.decorator import deprecated_keywords, deprecated


import io
from pkg_resources import load_entry_point
from uuid import uuid4
from copy import deepcopy
import collections
import copy
import glob
import inspect
import os
import re
import warnings
import weakref


EVENT_ENTRY_POINTS = ENTRY_POINTS['event']
EVENT_ENTRY_POINTS_WRITE = ENTRY_POINTS['event_write']
ATTRIBUTE_HAS_ERRORS = True


@map_example_filename("pathname_or_url")
def readEvents(pathname_or_url=None, format=None, **kwargs):
    """
    Read event files into an ObsPy Catalog object.

    The :func:`~obspy.core.event.readEvents` function opens either one or
    multiple event files given via file name or URL using the
    ``pathname_or_url`` attribute.

    :type pathname_or_url: str or StringIO.StringIO, optional
    :param pathname_or_url: String containing a file name or a URL or a open
        file-like object. Wildcards are allowed for a file name. If this
        attribute is omitted, an example :class:`~obspy.core.event.Catalog`
        object will be returned.
    :type format: str, optional
    :param format: Format of the file to read (e.g. ``"QUAKEML"``). See the
        `Supported Formats`_ section below for a list of supported formats.
    :return: A ObsPy :class:`~obspy.core.event.Catalog` object.

    .. rubric:: _`Supported Formats`

    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.event.readEvents` function. The following table
    summarizes all known file formats currently supported by ObsPy.

    Please refer to the `Linked Function Call`_ of each module for any extra
    options available at the import stage.

    %s

    Next to the :func:`~obspy.core.event.readEvents` function the
    :meth:`~obspy.core.event.Catalog.write` method of the returned
    :class:`~obspy.core.event.Catalog` object can be used to export the data to
    the file system.
    """
    if pathname_or_url is None:
        # if no pathname or URL specified, return example catalog
        return _createExampleCatalog()
    elif not isinstance(pathname_or_url, (str, native_str)):
        # not a string - we assume a file-like object
        try:
            # first try reading directly
            catalog = _read(pathname_or_url, format, **kwargs)
        except TypeError:
            # if this fails, create a temporary file which is read directly
            # from the file system
            pathname_or_url.seek(0)
            with NamedTemporaryFile() as fh:
                fh.write(pathname_or_url.read())
                catalog = _read(fh.name, format, **kwargs)
        return catalog
    elif isinstance(pathname_or_url, bytes) and \
            pathname_or_url.strip().startswith(b'<'):
        # XML string
        return _read(io.BytesIO(pathname_or_url), format, **kwargs)
    elif "://" in pathname_or_url[:10]:
        # URL
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        with NamedTemporaryFile(suffix=suffix) as fh:
            fh.write(urllib.request.urlopen(pathname_or_url).read())
            catalog = _read(fh.name, format, **kwargs)
        return catalog
    else:
        pathname = pathname_or_url
        # File name(s)
        pathnames = glob.glob(pathname)
        if not pathnames:
            # try to give more specific information why the stream is empty
            if glob.has_magic(pathname) and not glob.glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not glob.has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)

        catalog = _read(pathnames[0], format, **kwargs)
        if len(pathnames) == 1:
            return catalog
        else:
            for filename in pathnames[1:]:
                catalog.extend(_read(filename, format, **kwargs).events)


@uncompressFile
def _read(filename, format=None, **kwargs):
    """
    Reads a single event file into a ObsPy Catalog object.
    """
    catalog, format = _readFromPlugin('event', filename, format=format,
                                      **kwargs)
    for event in catalog:
        event._format = format
    return catalog


def _createExampleCatalog():
    """
    Create an example catalog.
    """
    return readEvents('/path/to/neries_events.xml')


class QuantityError(AttribDict):
    uncertainty = None
    lower_uncertainty = None
    upper_uncertainty = None
    confidence_level = None


def _bool(value):
    """
    A custom bool() implementation that returns
    True for any value (including zero) of int and float,
    and for (empty) strings.
    """
    if value == 0 or isinstance(value, (str, native_str)):
        return True
    return bool(value)


def _eventTypeClassFactory(class_name, class_attributes=[], class_contains=[]):
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
    has already been created. A useful type are Enums if you want to restrict
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
        >>> TestEventClass = _eventTypeClassFactory("TestEventClass", \
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
            id(test_event.resource_id.getReferredObject()))

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
    class_attributes, an error QuantityError will be be created that will be
    named like the attribute with "_errors" appended.

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
                repr_str = getattr(self, key).__repr__()
                # Print any associated errors.
                error_key = key + "_errors"
                if hasattr(self, error_key) and\
                   _bool(getattr(self, error_key)):
                    err_items = sorted(getattr(self, error_key).items())
                    repr_str += " [%s]" % ', '.join(
                        [str(k) + "=" + str(v) for k, v in err_items])
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
                    ret_str += '\n\t---------'
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
                self.resource_id.setReferredObject(self)

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

    >>> event = Event()
    >>> import sys
    >>> ref_count = sys.getrefcount(event)
    >>> res_id = ResourceIdentifier(referred_object=event)
    >>> # The reference does not changed the reference count of the object.
    >>> print(ref_count == sys.getrefcount(event))
    True
    >>> # It actually is the same object.
    >>> print(event is res_id.getReferredObject())
    True
    >>> # Deleting it, or letting the garbage collector handle the object will
    >>> # invalidate the reference.
    >>> del event
    >>> print(res_id.getReferredObject())
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
    >>> print(ref_a.getReferredObject())
    None
    >>> # This instance has an attached object.
    >>> ref_b = ResourceIdentifier(res_id, referred_object=event_object)
    >>> ref_c = ResourceIdentifier(res_id)
    >>> # All ResourceIdentifiers will refer to the same object.
    >>> assert(id(ref_a.getReferredObject()) == obj_id)
    >>> assert(id(ref_b.getReferredObject()) == obj_id)
    >>> assert(id(ref_c.getReferredObject()) == obj_id)

    The id can be converted to a valid QuakeML ResourceIdentifier by calling
    the convertIDToQuakeMLURI() method. The resulting id will be of the form::

        smi:authority_id/prefix/id

    >>> res_id = ResourceIdentifier(prefix='origin')
    >>> res_id.convertIDToQuakeMLURI(authority_id="obspy.org")
    >>> res_id # doctest:+ELLIPSIS
    ResourceIdentifier(id="smi:obspy.org/origin/...")
    >>> res_id = ResourceIdentifier(id='foo')
    >>> res_id.convertIDToQuakeMLURI()
    >>> res_id
    ResourceIdentifier(id="smi:local/foo")
    >>> # A good way to create a QuakeML compatibly ResourceIdentifier from
    >>> # scratch is
    >>> res_id = ResourceIdentifier(prefix='pick')
    >>> res_id.convertIDToQuakeMLURI(authority_id='obspy.org')
    >>> res_id  # doctest:+ELLIPSIS
    ResourceIdentifier(id="smi:obspy.org/pick/...")
    >>> # If the given ID is already a valid QuakeML
    >>> # ResourceIdentifier, nothing will happen.
    >>> res_id = ResourceIdentifier('smi:test.org/subdir/id')
    >>> res_id
    ResourceIdentifier(id="smi:test.org/subdir/id")
    >>> res_id.convertIDToQuakeMLURI()
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

    @deprecated_keywords({'resource_id': 'id'})
    def __init__(self, id=None, prefix="smi:local",
                 referred_object=None):
        # Create a resource id if None is given and possibly use a prefix.
        if id is None:
            self.fixed = False
            self._prefix = prefix
            self._uuid = str(uuid4())
        else:
            self.fixed = True
            self.id = id
        # Append the referred object in case one is given to the class level
        # reference dictionary.
        if referred_object is not None:
            self.setReferredObject(referred_object)

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

    def getReferredObject(self):
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

    def setReferredObject(self, referred_object):
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

    def convertIDToQuakeMLURI(self, authority_id="local"):
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
        self.id = self.getQuakeMLURI(authority_id=authority_id)

    def getQuakeMLURI(self, authority_id="local"):
        """
        Returns the ID as a valid QuakeML URI if possible. Does not
        change the ID itself.

        >>> res_id = ResourceIdentifier("some_id")
        >>> print(res_id.getQuakeMLURI())
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
    @deprecated("Attribute 'resource_id' was renamed to 'id'. "
                "Use that instead.")
    def resource_id(self):
        return self.id

    @resource_id.deleter
    @deprecated("Attribute 'resource_id' was renamed to 'id'. "
                "Use that instead.")
    def resource_id(self):
        del self.id

    @resource_id.setter
    @deprecated("Attribute 'resource_id' was renamed to 'id'. "
                "Use that instead.")
    def resource_id(self, value):
        self.id = value

    def __str__(self):
        return self.id

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return 'ResourceIdentifier(id="%s")' % self.id

    def __eq__(self, other):
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
        return self.id.__hash__()

    def regenerate_uuid(self):
        """
        Regenerates the uuid part of the ID. Does nothing for resource
        identifiers with a user-set, fixed id.
        """
        self._uuid = str(uuid4())


__CreationInfo = _eventTypeClassFactory(
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
    :type agency_uri: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param agency_uri: Resource Identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: :class:`~obspy.core.event.ResourceIdentifier`, optional
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


__TimeWindow = _eventTypeClassFactory(
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
    :param reference: Reference point in time (“central” point).

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__CompositeTime = _eventTypeClassFactory(
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
    :param year: Year or range of years of the event’s focal time.
    :type year_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param year_errors: AttribDict containing error quantities.
    :type month: int
    :param month: Month or range of months of the event’s focal time.
    :type month_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param month_errors: AttribDict containing error quantities.
    :type day: int
    :param day: Day or range of days of the event’s focal time.
    :type day_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param day_errors: AttribDict containing error quantities.
    :type hour: int
    :param hour: Hour or range of hours of the event’s focal time.
    :type hour_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param hour_errors: AttribDict containing error quantities.
    :type minute: int
    :param minute: Minute or range of minutes of the event’s focal time.
    :type minute_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param minute_errors: AttribDict containing error quantities.
    :type second: float
    :param second: Second and fraction of seconds or range of seconds with
        fraction of the event’s focal time.
    :type second_errors: :class:`~obspy.core.util.attribdict.AttribDict`
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


__Comment = _eventTypeClassFactory(
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
    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param resource_id: Resource identifier of comment.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
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


__WaveformStreamID = _eventTypeClassFactory(
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
    :type resource_uri: :class:`~obspy.core.event.ResourceIdentifier`, optional
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
    >>> print(stream_id.getSEEDString())
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

    def getSEEDString(self):
        return "%s.%s.%s.%s" % (
            self.network_code if self.network_code else "",
            self.station_code if self.station_code else "",
            self.location_code if self.location_code else "",
            self.channel_code if self.channel_code else "")


__Amplitude = _eventTypeClassFactory(
    "__Amplitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("generic_amplitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("type", str),
                      ("category", AmplitudeCategory),
                      ("unit", AmplitudeUnit),
                      ("method_id", ResourceIdentifier),
                      ("period", float, ATTRIBUTE_HAS_ERRORS),
                      ("snr", float),
                      ("time_window", TimeWindow),
                      ("pick_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID),
                      ("filter_id", ResourceIdentifier),
                      ("scaling_time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("magnitude_hint", str),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Amplitude(__Amplitude):
    """
    This class represents a quantification of the waveform anomaly, usually a
    single amplitude measurement or a measurement of the visible signal
    duration for duration magnitudes.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Amplitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type generic_amplitude: float
    :param generic_amplitude: Measured amplitude value for the given
        waveformID. Note that this attribute can describe different physical
        quantities, depending on the type and category of the amplitude. These
        can be, e.g., displacement, velocity, or a period. If the only
        amplitude information is a period, it has to specified here, not in the
        period attribute. The latter can be used if the amplitude measurement
        contains information on, e.g., displacement and an additional period.
        Since the physical quantity described by this attribute is not fixed,
        the unit of measurement cannot be defined in advance. However, the
        quantity has to be specified in SI base units. The enumeration given in
        attribute unit provides the most likely units that could be needed
        here. For clarity, using the optional unit attribute is highly
        encouraged.
    :type generic_amplitude_errors:
        :class:`~obspy.core.util.attribdict.AttribDict`
    :param generic_amplitude_errors: AttribDict containing error quantities.
    :type type: str, optional
    :param type: Describes the type of amplitude using the nomenclature from
        Storchak et al. (2003). Possible values are:

        * unspecified amplitude reading (``'A'``),
        * amplitude reading for local magnitude (``'AML'``),
        * amplitude reading for body wave magnitude (``'AMB'``),
        * amplitude reading for surface wave magnitude (``'AMS'``), and
        * time of visible end of record for duration magnitude (``'END'``).

    :type category: str, optional
    :param category:  Amplitude category.  This attribute describes the way the
        waveform trace is evaluated to derive an amplitude value. This can be
        just reading a single value for a given point in time (point), taking a
        mean value over a time interval (mean), integrating the trace over a
        time interval (integral), specifying just a time interval (duration),
        or evaluating a period (period).
        Possible values are:

        * ``"point"``,
        * ``"mean"``,
        * ``"duration"``,
        * ``"period"``,
        * ``"integral"``,
        * ``"other"``

    :type unit: str, optional
    :param unit: Amplitude unit. This attribute provides the most likely
        measurement units for the physical quantity described in the
        genericAmplitude attribute. Possible values are specified as
        combinations of SI base units.
        Possible values are:

        * ``"m"``,
        * ``"s"``,
        * ``"m/s"``,
        * ``"m/(s*s)"``,
        * ``"m*s"``,
        * ``"dimensionless"``,
        * ``"other"``

    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Describes the method of amplitude determination.
    :type period: float, optional
    :param period: Dominant period in the timeWindow in case of amplitude
        measurements. Not used for duration magnitude.  Unit: s
    :type snr: float, optional
    :param snr: Signal-to-noise ratio of the spectrogram at the location the
        amplitude was measured.
    :type time_window: :class:`~obspy.core.event.TimeWindow`, optional
    :param time_window: Description of the time window used for amplitude
        measurement. Recommended for duration magnitudes.
    :type pick_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param pick_id: Refers to the ``resource_id`` of an associated
        :class:`~obspy.core.event.Pick` object.
    :type waveform_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param waveform_id: Identifies the waveform stream on which the amplitude
        was measured.
    :type filter_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param filter_id: Identifies the filter or filter setup used for filtering
        the waveform stream referenced by ``waveform_id``.
    :type scaling_time: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param scaling_time: Scaling time for amplitude measurement.
    :type scaling_time_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param scaling_time_errors: AttribDict containing error quantities.
    :type magnitude_hint: str, optional
    :param magnitude_hint: Type of magnitude the amplitude measurement is used
        for.  This is a free-text field because it is impossible to cover all
        existing magnitude type designations with an enumeration. Possible
        values are:

        * unspecified magnitude (``'M'``),
        * local magnitude (``'ML'``),
        * body wave magnitude (``'Mb'``),
        * surface wave magnitude (``'MS'``),
        * moment magnitude (``'Mw'``),
        * duration magnitude (``'Md'``)
        * coda magnitude (``'Mc'``)
        * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.

    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Amplitude. Allowed values are
        the following:

        * ``"manual"``
        * ``"automatic"``

    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Amplitude. Allowed values
        are the following:

        * ``"preliminary"``
        * ``"confirmed"``
        * ``"reviewed"``
        * ``"final"``
        * ``"rejected"``
        * ``"reported"``

    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: CreationInfo for the Amplitude object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Pick = _eventTypeClassFactory(
    "__Pick",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("waveform_id", WaveformStreamID),
                      ("filter_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("horizontal_slowness", float, ATTRIBUTE_HAS_ERRORS),
                      ("backazimuth", float, ATTRIBUTE_HAS_ERRORS),
                      ("slowness_method_id", ResourceIdentifier),
                      ("onset", PickOnset),
                      ("phase_hint", str),
                      ("polarity", PickPolarity),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Pick(__Pick):
    """
    A pick is the observation of an amplitude anomaly in a seismogram at a
    specific point in time. It is not necessarily related to a seismic event.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Pick.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param time: Observed onset time of signal (“pick time”).
    :type time_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param time_errors: AttribDict containing error quantities.
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`
    :param waveform_id: Identifies the waveform stream.
    :type filter_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param filter_id: Identifies the filter or filter setup used for filtering
        the waveform stream referenced by waveform_id.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the picker that produced the pick. This can be
        either a detection software program or a person.
    :type horizontal_slowness: float, optional
    :param horizontal_slowness: Observed horizontal slowness of the signal.
        Most relevant in array measurements. Unit: s·deg^(−1)
    :type horizontal_slowness_errors:
        :class:`~obspy.core.util.attribdict.AttribDict`
    :param horizontal_slowness_errors: AttribDict containing error quantities.
    :type backazimuth: float, optional
    :param backazimuth: Observed backazimuth of the signal. Most relevant in
        array measurements. Unit: deg
    :type backazimuth_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param backazimuth_errors: AttribDict containing error quantities.
    :type slowness_method_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param slowness_method_id: Identifies the method that was used to determine
        the slowness.
    :type onset: str, optional
    :param onset: Flag that roughly categorizes the sharpness of the onset.
        Allowed values are:

        * ``"emergent"``
        * ``"impulsive"``
        * ``"questionable"``

    :type phase_hint: str, optional
    :param phase_hint: Tentative phase identification as specified by the
        picker.
    :type polarity: str, optional
    :param polarity: Indicates the polarity of first motion, usually from
        impulsive onsets. Allowed values are:

        * ``"positive"``
        * ``"negative"``
        * ``"undecidable"``

    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Pick. Allowed values are the
        following:

        * ``"manual"``
        * ``"automatic"``

    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Pick. Allowed values are
        the following:

        * ``"preliminary"``
        * ``"confirmed"``
        * ``"reviewed"``
        * ``"final"``
        * ``"rejected"``
        * ``"reported"``

    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: CreationInfo for the Pick object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Arrival = _eventTypeClassFactory(
    "__Arrival",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("pick_id", ResourceIdentifier),
                      ("phase", str),
                      ("time_correction", float),
                      ("azimuth", float),
                      ("distance", float),
                      ("takeoff_angle", float, ATTRIBUTE_HAS_ERRORS),
                      ("time_residual", float),
                      ("horizontal_slowness_residual", float),
                      ("backazimuth_residual", float),
                      ("time_weight", float),
                      ("horizontal_slowness_weight", float),
                      ("backazimuth_weight", float),
                      ("earth_model_id", ResourceIdentifier),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Arrival(__Arrival):
    """
    Successful association of a pick with an origin qualifies this pick as an
    arrival. An arrival thus connects a pick with an origin and provides
    additional attributes that describe this relationship. Usually
    qualification of a pick as an arrival for a given origin is a hypothesis,
    which is based on assumptions about the type of arrival (phase) as well as
    observed and (on the basis of an earth model) computed arrival times, or
    the residual, respectively. Additional pick attributes like the horizontal
    slowness and backazimuth of the observed wave—especially if derived from
    array data—may further constrain the nature of the arrival.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Arrival.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type pick_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param pick_id: Refers to the resource_id of a Pick.
    :type phase: str
    :param phase: Phase identification. For possible values, please refer to
        the description of the Phase object.
    :type time_correction: float, optional
    :param time_correction: Time correction value. Usually, a value
        characteristic for the station at which the pick was detected,
        sometimes also characteristic for the phase type or the slowness.
        Unit: s
    :type azimuth: float, optional
    :param azimuth: Azimuth of station as seen from the epicenter. Unit: deg
    :type distance: float, optional
    :param distance: Epicentral distance. Unit: deg
    :type takeoff_angle: float, optional
    :param takeoff_angle: Angle of emerging ray at the source, measured against
        the downward normal direction. Unit: deg
    :type takeoff_angle_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param takeoff_angle_errors: AttribDict containing error quantities.
    :type time_residual: float, optional
    :param time_residual: Residual between observed and expected arrival time
        assuming proper phase identification and given the earth_model_ID of
        the Origin, taking into account the time_correction. Unit: s
    :type horizontal_slowness_residual: float, optional
    :param horizontal_slowness_residual: Residual of horizontal slowness and
        the expected slowness given the current origin (refers to attribute
        horizontal_slowness of class Pick).
    :type backazimuth_residual: float, optional
    :param backazimuth_residual: Residual of backazimuth and the backazimuth
        computed for the current origin (refers to attribute backazimuth of
        class Pick).
    :type time_weight: float, optional
    :param time_weight: Weight of the arrival time for computation of the
        associated Origin. Note that the sum of all weights is not required to
        be unity.
    :type horizontal_slowness_weight: float, optional
    :param horizontal_slowness_weight: Weight of the horizontal slowness for
        computation of the associated Origin. Note that the sum of all weights
        is not required to be unity.
    :type backazimuth_weight: float, optional
    :param backazimuth_weight: Weight of the backazimuth for computation of the
        associated Origin. Note that the sum of all weights is not required to
        be unity.
    :type earth_model_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param earth_model_id: Earth model which is used for the association of
        Arrival to Pick and computation of the residuals.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: CreationInfo for the Arrival object.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__OriginQuality = _eventTypeClassFactory(
    "__OriginQuality",
    class_attributes=[("associated_phase_count", int),
                      ("used_phase_count", int),
                      ("associated_station_count", int),
                      ("used_station_count", int),
                      ("depth_phase_count", int),
                      ("standard_error", float),
                      ("azimuthal_gap", float),
                      ("secondary_azimuthal_gap", float),
                      ("ground_truth_level", str),
                      ("minimum_distance", float),
                      ("maximum_distance", float),
                      ("median_distance", float)])


class OriginQuality(__OriginQuality):
    """
    This type contains various attributes commonly used to describe the quality
    of an origin, e. g., errors, azimuthal coverage, etc. Origin objects have
    an optional attribute of the type OriginQuality.

    :type associated_phase_count: int, optional
    :param associated_phase_count: Number of associated phases, regardless of
        their use for origin computation.
    :type used_phase_count: int, optional
    :param used_phase_count: Number of defining phases, i. e., phase
        observations that were actually used for computing the origin. Note
        that there may be more than one defining phase per station.
    :type associated_station_count: int, optional
    :param associated_station_count: Number of stations at which the event was
        observed.
    :type used_station_count: int, optional
    :param used_station_count: Number of stations from which data was used for
        origin computation.
    :type depth_phase_count: int, optional
    :param depth_phase_count: Number of depth phases (typically pP, sometimes
        sP) used in depth computation.
    :type standard_error: float, optional
    :param standard_error: RMS of the travel time residuals of the arrivals
        used for the origin computation. Unit: s
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Largest azimuthal gap in station distribution as seen
        from epicenter. For an illustration of azimuthal gap and secondary
        azimuthal gap (see below), see Fig. 5 of Bond ́ar et al. (2004).
        Unit: deg
    :type secondary_azimuthal_gap: float, optional
    :param secondary_azimuthal_gap: Secondary azimuthal gap in station
        distribution, i. e., the largest azimuthal gap a station closes.
        Unit: deg
    :type ground_truth_level: str, optional
    :param ground_truth_level: String describing ground-truth level, e. g. GT0,
        GT5, etc.
    :type minimum_distance: float, optional
    :param minimum_distance: Epicentral distance of station closest to the
        epicenter.  Unit: deg
    :type maximum_distance: float, optional
    :param maximum_distance: Epicentral distance of station farthest from the
        epicenter.  Unit: deg
    :type median_distance: float, optional
    :param median_distance: Median epicentral distance of used stations.
        Unit: deg

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__ConfidenceEllipsoid = _eventTypeClassFactory(
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


__OriginUncertainty = _eventTypeClassFactory(
    "__OriginUncertainty",
    class_attributes=[("horizontal_uncertainty", float),
                      ("min_horizontal_uncertainty", float),
                      ("max_horizontal_uncertainty", float),
                      ("azimuth_max_horizontal_uncertainty", float),
                      ("confidence_ellipsoid", ConfidenceEllipsoid),
                      ("preferred_description", OriginUncertaintyDescription),
                      ("confidence_level", float)])


class OriginUncertainty(__OriginUncertainty):
    """
    This class describes the location uncertainties of an origin.

    The uncertainty can be described either as a simple circular horizontal
    uncertainty, an uncertainty ellipse according to IMS1.0, or a confidence
    ellipsoid. If multiple uncertainty models are given, the preferred variant
    can be specified in the attribute ``preferred_description``.

    :type horizontal_uncertainty: float, optional
    :param horizontal_uncertainty: Circular confidence region, given by single
        value of horizontal uncertainty. Unit: m
    :type min_horizontal_uncertainty: float, optional
    :param min_horizontal_uncertainty: Semi-minor axis of confidence ellipse.
        Unit: m
    :type max_horizontal_uncertainty: float, optional
    :param max_horizontal_uncertainty: Semi-major axis of confidence ellipse.
        Unit: m
    :type azimuth_max_horizontal_uncertainty: float, optional
    :param azimuth_max_horizontal_uncertainty: Azimuth of major axis of
        confidence ellipse. Measured clockwise from South-North direction at
        epicenter. Unit: deg
    :type confidence_ellipsoid: :class:`~obspy.core.event.ConfidenceEllipsoid`,
        optional
    :param confidence_ellipsoid: Confidence ellipsoid
    :type preferred_description: str, optional
    :param preferred_description: Preferred uncertainty description. Allowed
        values are the following:

        * horizontal uncertainty
        * uncertainty ellipse
        * confidence ellipsoid

    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Origin = _eventTypeClassFactory(
    "__Origin",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("longitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("latitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("depth", float, ATTRIBUTE_HAS_ERRORS),
                      ("depth_type", OriginDepthType),
                      ("time_fixed", bool),
                      ("epicenter_fixed", bool),
                      ("reference_system_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("earth_model_id", ResourceIdentifier),
                      ("quality", OriginQuality),
                      ("origin_type", OriginType),
                      ("origin_uncertainty", OriginUncertainty),
                      ("region", str),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "arrivals", "composite_times"])


class Origin(__Origin):
    """
    This class represents the focal time and geographical location of an
    earthquake hypocenter, as well as additional meta-information. Origin can
    have objects of type OriginUncertainty and Arrival as child elements.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Origin.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param time: Focal time.
    :type time_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param time_errors: AttribDict containing error quantities.
    :type longitude: float
    :param longitude: Hypocenter longitude, with respect to the World Geodetic
        System 1984 (WGS84) reference system. Unit: deg
    :type longitude_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param longitude_errors: AttribDict containing error quantities.
    :type latitude: float
    :param latitude: Hypocenter latitude, with respect to the WGS84 reference
        system. Unit: deg
    :type latitude_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param latitude_errors: AttribDict containing error quantities.
    :type depth: float, optional
    :param depth: Depth of hypocenter with respect to the nominal sea level
        given by the WGS84 geoid. Positive values indicate hypocenters below
        sea level. For shallow hypocenters, the depth value can be negative.
        Note: Other standards use different conventions for depth measurement.
        As an example, GSE2.0, defines depth with respect to the local surface.
        If event data is converted from other formats to QuakeML, depth values
        may have to be modified accordingly. Unit: m
    :type depth_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param depth_errors: AttribDict containing error quantities.
    :type depth_type: str, optional
    :param depth_type: Type of depth determination. Allowed values are the
        following:

        * ``"from location"``
        * ``"from moment tensor inversion"``
        * ``"from modeling of broad-band P waveforms"``
        * ``"constrained by depth phases"``
        * ``"constrained by direct phases"``
        * ``"constrained by depth and direct phases"``
        * ``"operator assigned"``
        * ``"other"``

    :type time_fixed: bool, optional
    :param time_fixed: True if focal time was kept fixed for computation of the
        Origin.
    :type epicenter_fixed: bool, optional
    :param epicenter_fixed: True if epicenter was kept fixed for computation of
        Origin.
    :type reference_system_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param reference_system_id: Identifies the reference system used for
        hypocenter determination. This is only necessary if a modified version
        of the standard (with local extensions) is used that provides a
        non-standard coordinate system.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the method used for locating the event.
    :type earth_model_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param earth_model_id: Identifies the earth model used in method_id.
    :type arrivals: list of :class:`~obspy.core.event.Arrival`, optional
    :param arrivals: List of arrivals associated with the origin.
    :type composite_times: list of :class:`~obspy.core.event.CompositeTime`,
        optional
    :param composite_times: Supplementary information on time of rupture start.
        Complex descriptions of focal times of historic events are possible,
        see description of the CompositeTime type. Note that even if
        compositeTime is used, the mandatory time attribute has to be set, too.
        It has to be set to the single point in time (with uncertainties
        allowed) that is most characteristic for the event.
    :type quality: :class:`~obspy.core.event.OriginQuality`, optional
    :param quality: Additional parameters describing the quality of an Origin
        determination.
    :type origin_type: str, optional
    :param origin_type: Describes the origin type. Allowed values are the
        following:

        * ``"hypocenter"``
        * ``"centroid"``
        * ``"amplitude"``
        * ``"macroseismic"``
        * ``"rupture start"``
        * ``"rupture end"``

    :type origin_uncertainty: :class:`~obspy.core.event.OriginUncertainty`,
        optional
    :param origin_uncertainty: Describes the location uncertainties of an
        origin.
    :type region: str, optional
    :param region: Can be used to describe the geographical region of the
        epicenter location. Useful if an event has multiple origins from
        different agencies, and these have different region designations. Note
        that an event-wide region can be defined in the description attribute
        of an Event object. The user has to take care that this information
        corresponds to the region attribute of the preferred Origin.
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Origin. Allowed values are the
        following:

        * ``"manual"``
        * ``"automatic"``

    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of Origin. Allowed values are
        the following:

        * ``"preliminary"``
        * ``"confirmed"``
        * ``"reviewed"``
        * ``"final"``
        * ``"rejected"``
        * ``"reported"``

    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. rubric:: Example

    >>> from obspy.core.event import Origin
    >>> origin = Origin()
    >>> origin.resource_id = 'smi:ch.ethz.sed/origin/37465'
    >>> origin.time = UTCDateTime(0)
    >>> origin.latitude = 12
    >>> origin.latitude_errors.confidence_level = 95.0
    >>> origin.longitude = 42
    >>> origin.depth_type = 'from location'
    >>> print(origin)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Origin
        resource_id: ResourceIdentifier(id="smi:ch.ethz.sed/...")
               time: UTCDateTime(1970, 1, 1, 0, 0)
          longitude: 42.0
           latitude: 12.0 [confidence_level=95.0]
         depth_type: ...'from location'

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__StationMagnitudeContribution = _eventTypeClassFactory(
    "__StationMagnitudeContribution",
    class_attributes=[("station_magnitude_id", ResourceIdentifier),
                      ("residual", float),
                      ("weight", float)])


class StationMagnitudeContribution(__StationMagnitudeContribution):
    """
    This class describes the weighting of magnitude values from several
    StationMagnitude objects for computing a network magnitude estimation.

    :type station_magnitude_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param station_magnitude_id: Refers to the resource_id of a
        StationMagnitude object.
    :type residual: float, optional
    :param residual: Residual of magnitude computation.
    :type weight: float, optional
    :param weight: Weight of the magnitude value from class StationMagnitude
        for computing the magnitude value in class Magnitude. Note that there
        is no rule for the sum of the weights of all station magnitude
        contributions to a specific network magnitude. In particular, the
        weights are not required to sum up to unity.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Magnitude = _eventTypeClassFactory(
    "__Magnitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("mag", float, ATTRIBUTE_HAS_ERRORS),
                      ("magnitude_type", str),
                      ("origin_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("station_count", int),
                      ("azimuthal_gap", float),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "station_magnitude_contributions"])


class Magnitude(__Magnitude):
    """
    Describes a magnitude which can, but does not need to be associated with an
    origin.

    Association with an origin is expressed with the optional attribute
    originID. It is either a combination of different magnitude estimations, or
    it represents the reported magnitude for the given event.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Magnitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type mag: float
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.StationMagnitude`. If no estimations are
        available, this value can represent the reported magnitude.
    :type mag_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param mag_errors: AttribDict containing error quantities.
    :type magnitude_type: str, optional
    :param magnitude_type: Describes the type of magnitude. This is a free-text
        field because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are:

        * unspecified magnitude (``'M'``),
        * local magnitude (``'ML'``),
        * body wave magnitude (``'Mb'``),
        * surface wave magnitude (``'MS'``),
        * moment magnitude (``'Mw'``),
        * duration magnitude (``'Md'``)
        * coda magnitude (``'Mc'``)
        * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.

    :type origin_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param origin_id: Reference to an origin’s resource_id if the magnitude has
        an associated Origin.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and
        magnitude_type.
    :type station_count: int, optional
    :param station_count: Number of used stations for this magnitude
        computation.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Azimuthal gap for this magnitude computation.
        Unit: deg
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of Magnitude. Allowed values are
        the following:

        * ``"manual"``
        * ``"automatic"``

    :type evaluation_status:
        :class:`~obspy.core.event_header.EvaluationStatus`, optional
    :param evaluation_status: Evaluation status of Magnitude. Allowed values
        are the following:

        * ``"preliminary"``
        * ``"confirmed"``
        * ``"reviewed"``
        * ``"final"``
        * ``"rejected"``
        * ``"reported"``

    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type station_magnitude_contributions: list of
        :class:`~obspy.core.event.StationMagnitudeContribution`.
    :param station_magnitude_contributions: StationMagnitudeContribution
        instances associated with the Magnitude.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__StationMagnitude = _eventTypeClassFactory(
    "__StationMagnitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("origin_id", ResourceIdentifier),
                      ("mag", float, ATTRIBUTE_HAS_ERRORS),
                      ("station_magnitude_type", str),
                      ("amplitude_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class StationMagnitude(__StationMagnitude):
    """
    This class describes the magnitude derived from a single waveform stream.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param resource_id: Resource identifier of StationMagnitude.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type origin_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param origin_id: Reference to an origin’s ``resource_id`` if the
        StationMagnitude has an associated :class:`~obspy.core.event.Origin`.
    :type mag: float
    :param mag: Estimated magnitude.
    :type mag_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param mag_errors: AttribDict containing error quantities.
    :type station_magnitude_type: str, optional
    :param station_magnitude_type: See :class:`~obspy.core.event.Magnitude`
    :type amplitude_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param amplitude_id: Identifies the data source of the StationMagnitude.
        For magnitudes derived from amplitudes in waveforms (e.g., local
        magnitude ML), amplitudeID points to publicID in class Amplitude.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: See :class:`~obspy.core.event.Magnitude`
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`, optional
    :param waveform_id: Identifies the waveform stream. This element can be
        helpful if no amplitude is referenced, or the amplitude is not
        available in the context. Otherwise, it would duplicate the waveform_id
        provided there and can be omitted.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__EventDescription = _eventTypeClassFactory(
    "__EventDescription",
    class_attributes=[("text", str),
                      ("type", EventDescriptionType)])


class EventDescription(__EventDescription):
    """
    Free-form string with additional event description. This can be a
    well-known name, like 1906 San Francisco Earthquake. A number of categories
    can be given in type.

    :type text: str, optional
    :param text: Free-form text with earthquake description.
    :type type: str, optional
    :param type: Category of earthquake description. Values
        can be taken from the following:

        * ``"felt report"``
        * ``"Flinn-Engdahl region"``
        * ``"local time"``
        * ``"tectonic summary"``
        * ``"nearest cities"``
        * ``"earthquake name"``
        * ``"region name"``

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Tensor = _eventTypeClassFactory(
    "__Tensor",
    class_attributes=[("m_rr", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_pp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tp", float, ATTRIBUTE_HAS_ERRORS)])


class Tensor(__Tensor):
    """
    The Tensor class represents the six moment-tensor elements Mrr, Mtt, Mpp,
    Mrt, Mrp, Mtp in the spherical coordinate system defined by local upward
    vertical (r), North-South (t), and West-East (p) directions.

    :type m_rr: float
    :param m_rr: Moment-tensor element Mrr. Unit: Nm
    :type m_rr_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_rr_errors: AttribDict containing error quantities.
    :type m_tt: float
    :param m_tt: Moment-tensor element Mtt. Unit: Nm
    :type m_tt_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_tt_errors: AttribDict containing error quantities.
    :type m_pp: float
    :param m_pp: Moment-tensor element Mpp. Unit: Nm
    :type m_pp_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_pp_errors: AttribDict containing error quantities.
    :type m_rt: float
    :param m_rt: Moment-tensor element Mrt. Unit: Nm
    :type m_rt_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_rt_errors: AttribDict containing error quantities.
    :type m_rp: float
    :param m_rp: Moment-tensor element Mrp. Unit: Nm
    :type m_rp_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_rp_errors: AttribDict containing error quantities.
    :type m_tp: float
    :param m_tp: Moment-tensor element Mtp. Unit: Nm
    :type m_tp_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param m_tp_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__DataUsed = _eventTypeClassFactory(
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
    :param wave_type: Type of waveform data. This can be one of the following
        values:

        * ``"P waves"``,
        * ``"body waves"``,
        * ``"surface waves"``,
        * ``"mantle waves"``,
        * ``"combined"``,
        * ``"unknown"``

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


__SourceTimeFunction = _eventTypeClassFactory(
    "__SourceTimeFunction",
    class_attributes=[("type", SourceTimeFunctionType),
                      ("duration", float),
                      ("rise_time", float),
                      ("decay_time", float)])


class SourceTimeFunction(__SourceTimeFunction):
    """
    Source time function used in moment-tensor inversion.

    :type type: str
    :param type: Type of source time function. Values can be taken from the
        following:

        * ``"box car"``,
        * ``"triangle"``,
        * ``"trapezoid"``,
        * ``"unknown"``

    :type duration: float
    :param duration: Source time function duration. Unit: s
    :type rise_time: float, optional
    :param rise_time: Source time function rise time. Unit: s
    :type decay_time: float, optional
    :param decay_time: Source time function decay time. Unit: s

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__NodalPlane = _eventTypeClassFactory(
    "__NodalPlane",
    class_attributes=[("strike", float, ATTRIBUTE_HAS_ERRORS),
                      ("dip", float, ATTRIBUTE_HAS_ERRORS),
                      ("rake", float, ATTRIBUTE_HAS_ERRORS)])


class NodalPlane(__NodalPlane):
    """
    This class describes a nodal plane using the attributes strike, dip, and
    rake. For a definition of the angles see Aki & Richards (1980).

    :type strike: float
    :param strike: Strike angle of nodal plane. Unit: deg
    :type strike_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param strike_errors: AttribDict containing error quantities.
    :type dip: float
    :param dip: Dip angle of nodal plane. Unit: deg
    :type dip_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param dip_errors: AttribDict containing error quantities.
    :type rake: float
    :param rake: Rake angle of nodal plane. Unit: deg
    :type rake_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param rake_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Axis = _eventTypeClassFactory(
    "__Axis",
    class_attributes=[("azimuth", float, ATTRIBUTE_HAS_ERRORS),
                      ("plunge", float, ATTRIBUTE_HAS_ERRORS),
                      ("length", float, ATTRIBUTE_HAS_ERRORS)])


class Axis(__Axis):
    """
    This class describes an eigenvector of a moment tensor expressed in its
    principal-axes system. It uses the angles azimuth, plunge, and the
    eigenvalue length.

    :type azimuth: float
    :param azimuth: Azimuth of eigenvector of moment tensor expressed in
        principal-axes system. Measured clockwise from South-North direction at
        epicenter. Unit: deg
    :type azimuth_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param azimuth_errors: AttribDict containing error quantities.
    :type plunge: float
    :param plunge: Plunge of eigenvector of moment tensor expressed in
        principal-axes system. Measured against downward vertical direction at
        epicenter. Unit: deg
    :type plunge_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param plunge_errors: AttribDict containing error quantities.
    :type length: float
    :param length: Eigenvalue of moment tensor expressed in principal-axes
        system. Unit: Nm
    :type length_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param length_errors: AttribDict containing error quantities.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__NodalPlanes = _eventTypeClassFactory(
    "__NodalPlanes",
    class_attributes=[("nodal_plane_1", NodalPlane),
                      ("nodal_plane_2", NodalPlane),
                      ("preferred_plane", int)])


class NodalPlanes(__NodalPlanes):
    """
    This class describes the nodal planes of a double-couple moment-tensor
    solution. The attribute ``preferred_plane`` can be used to define which
    plane is the preferred one.

    :type nodal_plane_1: :class:`~obspy.core.event.NodalPlane`, optional
    :param nodal_plane_1: First nodal plane of double-couple moment tensor
        solution.
    :type nodal_plane_2: :class:`~obspy.core.event.NodalPlane`, optional
    :param nodal_plane_2: Second nodal plane of double-couple moment tensor
        solution.
    :type preferred_plane: int, optional
    :param preferred_plane: Indicator for preferred nodal plane of moment
        tensor solution. It can take integer values ``1`` or ``2``.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__PrincipalAxes = _eventTypeClassFactory(
    "__PrincipalAxes",
    class_attributes=[("t_axis", Axis),
                      ("p_axis", Axis),
                      ("n_axis", Axis)])


class PrincipalAxes(__PrincipalAxes):
    """
    This class describes the principal axes of a double-couple moment tensor
    solution. t_axis and p_axis are required, while n_axis is optional.

    :type t_axis: :class:`~obspy.core.event.Axis`
    :param t_axis: T (tension) axis of a double-couple moment tensor solution.
    :type p_axis: :class:`~obspy.core.event.Axis`
    :param p_axis: P (pressure) axis of a double-couple moment tensor solution.
    :type n_axis: :class:`~obspy.core.event.Axis`, optional
    :param n_axis: N (neutral) axis of a double-couple moment tensor solution.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__MomentTensor = _eventTypeClassFactory(
    "__MomentTensor",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("derived_origin_id", ResourceIdentifier),
                      ("moment_magnitude_id", ResourceIdentifier),
                      ("scalar_moment", float, ATTRIBUTE_HAS_ERRORS),
                      ("tensor", Tensor),
                      ("variance", float),
                      ("variance_reduction", float),
                      ("double_couple", float),
                      ("clvd", float),
                      ("iso", float),
                      ("greens_function_id", ResourceIdentifier),
                      ("filter_id", ResourceIdentifier),
                      ("source_time_function", SourceTimeFunction),
                      ("method_id", ResourceIdentifier),
                      ("category", MomentTensorCategory),
                      ("inversion_type", MTInversionType),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "data_used"])


class MomentTensor(__MomentTensor):
    """
    This class represents a moment tensor solution for an event. It is an
    optional part of a FocalMechanism description.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of MomentTensor.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type derived_origin_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param derived_origin_id: Refers to the resource_id of the Origin derived
        in the moment tensor inversion.
    :type moment_magnitude_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param moment_magnitude_id: Refers to the publicID of the Magnitude object
        which represents the derived moment magnitude.
    :type scalar_moment: float, optional
    :param scalar_moment: Scalar moment as derived in moment tensor inversion.
        Unit: Nm
    :type scalar_moment_errors: :class:`~obspy.core.util.attribdict.AttribDict`
    :param scalar_moment_errors: AttribDict containing error quantities.
    :type tensor: :class:`~obspy.core.event.Tensor`, optional
    :param tensor: Tensor object holding the moment tensor elements.
    :type variance: float, optional
    :param variance: Variance of moment tensor inversion.
    :type variance_reduction: float, optional
    :param variance_reduction: Variance reduction of moment tensor inversion,
        given in percent (Dreger 2003). This is a goodness-of-fit measure.
    :type double_couple: float, optional
    :param double_couple: Double couple parameter obtained from moment tensor
        inversion (decimal fraction between 0 and 1).
    :type clvd: float, optional
    :param clvd: CLVD (compensated linear vector dipole) parameter obtained
        from moment tensor inversion (decimal fraction between 0 and 1).
    :type iso: float, optional
    :param iso: Isotropic part obtained from moment tensor inversion (decimal
        fraction between 0 and 1).
    :type greens_function_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param greens_function_id: Resource identifier of the Green’s function used
        in moment tensor inversion.
    :type filter_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param filter_id: Resource identifier of the filter setup used in moment
        tensor inversion.
    :type source_time_function: :class:`~obspy.core.event.SourceTimeFunction`,
        optional
    :param source_time_function: Source time function used in moment-tensor
        inversion.
    :type data_used: list of :class:`~obspy.core.event.DataUsed`, optional
    :param data_used: Describes waveform data used for moment-tensor inversion.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Resource identifier of the method used for moment-tensor
        inversion.
    :type category: str, optional
    :param category: Moment tensor category. Values can be taken from the
        following:

        * ``"teleseismic"``,
        * ``"regional"``

    :type inversion_type: str, optional
    :param inversion_type: Moment tensor inversion type. Users should avoid to
        give contradictory information in inversion_type and method_id. Values
        can be taken from the following:

        * ``"general"``,
        * ``"zero trace"``,
        * ``"double couple"``

    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__FocalMechanism = _eventTypeClassFactory(
    "__FocalMechanism",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("triggering_origin_id", ResourceIdentifier),
                      ("nodal_planes", NodalPlanes),
                      ("principal_axes", PrincipalAxes),
                      ("azimuthal_gap", float),
                      ("station_polarity_count", int),
                      ("misfit", float),
                      ("station_distribution_ratio", float),
                      ("method_id", ResourceIdentifier),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("moment_tensor", MomentTensor),
                      ("creation_info", CreationInfo)],
    class_contains=['waveform_id', 'comments'])


class FocalMechanism(__FocalMechanism):
    """
    This class describes the focal mechanism of an event. It includes different
    descriptions like nodal planes, principal axes, and a moment tensor. The
    moment tensor description is provided by objects of the class MomentTensor
    which can be specified as child elements of FocalMechanism.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of FocalMechanism.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type triggering_origin_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param triggering_origin_id: Refers to the resource_id of the triggering
        origin.
    :type nodal_planes: :class:`~obspy.core.event.NodalPlanes`, optional
    :param nodal_planes: Nodal planes of the focal mechanism.
    :type principal_axes: :class:`~obspy.core.event.PrincipalAxes`, optional
    :param principal_axes: Principal axes of the focal mechanism.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Largest azimuthal gap in distribution of stations
        used for determination of focal mechanism. Unit: deg
    :type station_polarity_count: int, optional
    :param station_polarity_count:
    :type misfit: float, optional
    :param misfit: Fraction of misfit polarities in a first-motion focal
        mechanism determination. Decimal fraction between 0 and 1.
    :type station_distribution_ratio: float, optional
    :param station_distribution_ratio: Station distribution ratio (STDR)
        parameter. Indicates how the stations are distributed about the focal
        sphere (Reasenberg and Oppenheimer 1985). Decimal fraction between 0
        and 1.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Resource identifier of the method used for determination
        of the focal mechanism.
    :type waveform_id: list of :class:`~obspy.core.event.WaveformStreamID`,
        optional
    :param waveform_id: Refers to a set of waveform streams from which the
        focal mechanism was derived.
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of FocalMechanism. Allowed values
        are the following:

        * ``"manual"``
        * ``"automatic"``

    :type evaluation_status: str, optional
    :param evaluation_status: Evaluation status of FocalMechanism. Allowed
        values are the following:

        * ``"preliminary"``
        * ``"confirmed"``
        * ``"reviewed"``
        * ``"final"``
        * ``"rejected"``
        * ``"reported"``

    :type moment_tensor: :class:`~obspy.core.event.MomentTensor`, optional
    :param moment_tensor: Moment tensor description for this focal mechanism.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


__Event = _eventTypeClassFactory(
    "__Event",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("event_type", EventType),
                      ("event_type_certainty", EventTypeCertainty),
                      ("creation_info", CreationInfo)],
    class_contains=['event_descriptions', 'comments', 'picks', 'amplitudes',
                    'focal_mechanisms', 'origins', 'magnitudes',
                    'station_magnitudes'])


class Event(__Event):
    """
    The class Event describes a seismic event which does not necessarily need
    to be a tectonic earthquake. An event is usually associated with one or
    more origins, which contain information about focal time and geographical
    location of the event. Multiple origins can cover automatic and manual
    locations, a set of location from different agencies, locations generated
    with different location programs and earth models, etc. Furthermore, an
    event is usually associated with one or more magnitudes, and with one or
    more focal mechanism determinations.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Event.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type event_type: str, optional
    :param event_type: Describes the type of an event. Allowed values are the
        following:

        * ``"not existing"``
        * ``"not reported"``
        * ``"earthquake"``
        * ``"anthropogenic event"``
        * ``"collapse"``
        * ``"cavity collapse"``
        * ``"mine collapse"``
        * ``"building collapse"``
        * ``"explosion"``
        * ``"accidental explosion"``
        * ``"chemical explosion"``
        * ``"controlled explosion"``
        * ``"experimental explosion"``
        * ``"industrial explosion"``
        * ``"mining explosion"``
        * ``"quarry blast"``
        * ``"road cut"``
        * ``"blasting levee"``
        * ``"nuclear explosion"``
        * ``"induced or triggered event"``
        * ``"rock burst"``
        * ``"reservoir loading"``
        * ``"fluid injection"``
        * ``"fluid extraction"``
        * ``"crash"``
        * ``"plane crash"``
        * ``"train crash"``
        * ``"boat crash"``
        * ``"other event"``
        * ``"atmospheric event"``
        * ``"sonic boom"``
        * ``"sonic blast"``
        * ``"acoustic noise"``
        * ``"thunder"``
        * ``"avalanche"``
        * ``"snow avalanche"``
        * ``"debris avalanche"``
        * ``"hydroacoustic event"``
        * ``"ice quake"``
        * ``"slide"``
        * ``"landslide"``
        * ``"rockslide"``
        * ``"meteorite"``
        * ``"volcanic eruption"``

    :type event_type_certainty: str, optional
    :param event_type_certainty: Denotes how certain the information on event
        type is. Allowed values are the following:

        * ``"suspected"``
        * ``"known"``

    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    :type event_descriptions: list of
        :class:`~obspy.core.event.EventDescription`
    :param event_descriptions: Additional event description, like earthquake
        name, Flinn-Engdahl region, etc.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type picks: list of :class:`~obspy.core.event.Pick`
    :param picks: Picks associated with the event.
    :type amplitudes: list of :class:`~obspy.core.event.Amplitude`
    :param amplitudes: Amplitudes associated with the event.
    :type focal_mechanisms: list of :class:`~obspy.core.event.FocalMechanism`
    :param focal_mechanisms: Focal mechanisms associated with the event
    :type origins: list of :class:`~obspy.core.event.Origin`
    :param origins: Origins associated with the event.
    :type magnitudes: list of :class:`~obspy.core.event.Magnitude`
    :param magnitudes: Magnitudes associated with the event.
    :type station_magnitudes: list of
        :class:`~obspy.core.event.StationMagnitude`
    :param station_magnitudes: Station magnitudes associated with the event.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """
    def short_str(self):
        """
        Returns a short string representation of the current Event.

        Example:
        Time | Lat | Long | Magnitude of the first origin, e.g.
        2011-03-11T05:46:24.120000Z | +38.297, +142.373 | 9.1 MW
        """
        out = ''
        origin = None
        if self.origins:
            origin = self.preferred_origin() or self.origins[0]
            out += '%s | %+7.3f, %+8.3f' % (origin.time,
                                            origin.latitude,
                                            origin.longitude)
        if self.magnitudes:
            magnitude = self.preferred_magnitude() or self.magnitudes[0]
            out += ' | %s %-2s' % (magnitude.mag,
                                   magnitude.magnitude_type)
        if origin and origin.evaluation_mode:
            out += ' | %s' % (origin.evaluation_mode)
        return out

    def __str__(self):
        """
        Print a short summary at the top.
        """
        return "Event:\t%s\n\n%s" % (
            self.short_str(),
            "\n".join(super(Event, self).__str__().split("\n")[1:]))

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return super(Event, self).__str__(force_one_line=True)

    def preferred_origin(self):
        """
        Returns the preferred origin
        """
        try:
            return ResourceIdentifier(self.preferred_origin_id).\
                getReferredObject()
        except AttributeError:
            return None

    def preferred_magnitude(self):
        """
        Returns the preferred magnitude
        """
        try:
            return ResourceIdentifier(self.preferred_magnitude_id).\
                getReferredObject()
        except AttributeError:
            return None

    def preferred_focal_mechanism(self):
        """
        Returns the preferred focal mechanism
        """
        try:
            return ResourceIdentifier(self.preferred_focal_mechanism_id).\
                getReferredObject()
        except AttributeError:
            return None

    def write(self, filename, format, **kwargs):
        """
        Saves event information into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str
        :param format: The file format to use (e.g. ``"QUAKEML"``). See
            :meth:`Catalog.write()` for a list of supported formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            plugin's writer method.

        .. rubric:: Example

        >>> from obspy import readEvents
        >>> event = readEvents()[0]  # doctest: +SKIP
        >>> event.write("example.xml", format="QUAKEML")  # doctest: +SKIP
        """
        Catalog(events=[self]).write(filename, format, **kwargs)


class Catalog(object):
    """
    This class serves as a container for Event objects.

    :type events: list of :class:`~obspy.core.event.Event`, optional
    :param events: List of events
    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of the catalog.
    :type description: str, optional
    :param description: Description string that can be assigned to the
        earthquake catalog, or collection of events.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """
    def __init__(self, events=None, **kwargs):
        if not events:
            self.events = []
        else:
            self.events = events
        self.comments = kwargs.get("comments", [])
        self._set_resource_id(kwargs.get("resource_id", None))
        self.description = kwargs.get("description", "")
        self._set_creation_info(kwargs.get("creation_info", None))

    def _get_resource_id(self):
        return self.__dict__['resource_id']

    def _set_resource_id(self, value):
        if type(value) == dict:
            value = ResourceIdentifier(**value)
        elif type(value) != ResourceIdentifier:
            value = ResourceIdentifier(value)
        self.__dict__['resource_id'] = value

    resource_id = property(_get_resource_id, _set_resource_id)

    def _get_creation_info(self):
        return self.__dict__['creation_info']

    def _set_creation_info(self, value):
        if type(value) == dict:
            value = CreationInfo(**value)
        elif type(value) != CreationInfo:
            value = CreationInfo(value)
        self.__dict__['creation_info'] = value

    creation_info = property(_get_creation_info, _set_creation_info)

    def __add__(self, other):
        """
        Method to add two catalogs.
        """
        if isinstance(other, Event):
            other = Catalog([other])
        if not isinstance(other, Catalog):
            raise TypeError
        events = self.events + other.events
        return self.__class__(events=events)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.events.__delitem__(index)

    def __eq__(self, other):
        """
        __eq__ method of the Catalog object.

        :type other: :class:`~obspy.core.event.Catalog`
        :param other: Catalog object for comparison.
        :rtype: bool
        :return: ``True`` if both Catalogs contain the same events.

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> cat2 = cat.copy()
        >>> cat is cat2
        False
        >>> cat == cat2
        True
        """
        if not isinstance(other, Catalog):
            return False
        if self.events != other.events:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object.

        :return: Event objects
        """
        if index == "extra":
            return self.__dict__[index]
        if isinstance(index, slice):
            return self.__class__(events=self.events.__getitem__(index))
        else:
            return self.events.__getitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method of the Catalog object.

        :return: Catalog object
        """
        # see also http://docs.python.org/reference/datamodel.html
        return self.__class__(events=self.events[max(0, i):max(0, j):k])

    def __iadd__(self, other):
        """
        Method to add two catalog with self += other.

        It will extend the current Catalog object with the events of the given
        Catalog. Events will not be copied but references to the original
        events will be appended.

        :type other: :class:`~obspy.core.event.Catalog` or
            :class:`~obspy.core.event.Event`
        :param other: Catalog or Event object to add.
        """
        if isinstance(other, Event):
            other = Catalog(events=[other])
        if not isinstance(other, Catalog):
            raise TypeError
        self.extend(other.events)
        return self

    def __iter__(self):
        """
        Return a robust iterator for Events of current Catalog.

        Doing this it is safe to remove events from catalogs inside of
        for-loops using catalog's :meth:`~obspy.core.event.Catalog.remove`
        method. Actually this creates a new iterator every time a event is
        removed inside the for-loop.
        """
        return list(self.events).__iter__()

    def __len__(self):
        """
        Returns the number of Events in the Catalog object.
        """
        return len(self.events)

    count = __len__

    def __setitem__(self, index, event):
        """
        __setitem__ method of the Catalog object.
        """
        if not isinstance(index, (str, native_str)):
            self.events.__setitem__(index, event)
        else:
            super(Catalog, self).__setitem__(index, event)

    def __str__(self, print_all=False):
        """
        Returns short summary string of the current catalog.

        It will contain the number of Events in the Catalog and the return
        value of each Event's :meth:`~obspy.core.event.Event.__str__` method.

        :type print_all: bool, optional
        :param print_all: If True, all events will be printed, otherwise a
            maximum of ten event will be printed.
            Defaults to False.
        """
        out = str(len(self.events)) + ' Event(s) in Catalog:\n'
        if len(self) <= 10 or print_all is True:
            out += "\n".join([ev.short_str() for ev in self])
        else:
            out += "\n".join([ev.short_str() for ev in self[:2]])
            out += "\n...\n"
            out += "\n".join([ev.short_str() for ev in self[-2:]])
            out += "\nTo see all events call " + \
                   "'print(CatalogObject.__str__(print_all=True))'"
        return out

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__(print_all=p.verbose))

    def append(self, event):
        """
        Appends a single Event object to the current Catalog object.
        """
        if isinstance(event, Event):
            self.events.append(event)
        else:
            msg = 'Append only supports a single Event object as an argument.'
            raise TypeError(msg)

    def clear(self):
        """
        Clears event list (convenient method).

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> len(cat)
        3
        >>> cat.clear()
        >>> cat.events
        []
        """
        self.events = []

    def filter(self, *args, **kwargs):
        """
        Returns a new Catalog object only containing Events which match the
        specified filter rules.

        Valid filter keys are:

        * magnitude;
        * longitude;
        * latitude;
        * depth;
        * time;
        * standard_error;
        * azimuthal_gap;
        * used_station_count;
        * used_phase_count.

        Use ``inverse=True`` to return the Events that *do not* match the
        specified filter rules.

        :rtype: :class:`Catalog`
        :return: Filtered catalog. A new Catalog object with filtered
            Events as references to the original Events.

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> print(cat)
        3 Event(s) in Catalog:
        2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4 mb | manual
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0 ML | manual
        >>> cat2 = cat.filter("magnitude >= 4.0", "latitude < 40.0")
        >>> print(cat2)
        1 Event(s) in Catalog:
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        >>> cat3 = cat.filter("time > 2012-04-04T14:10",
        ...                   "time < 2012-04-04T14:20")
        >>> print(cat3)
        1 Event(s) in Catalog:
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        >>> cat4 = cat.filter("time > 2012-04-04T14:10",
        ...                   "time < 2012-04-04T14:20",
        ...                   inverse=True)
        >>> print(cat4)
        2 Event(s) in Catalog:
        2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4 mb | manual
        2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0 ML | manual
        """
        # Helper functions. Only first argument might be None. Avoid
        # unorderable types by checking first shortcut on positive is None
        # also for the greater stuff (is confusing but correct)
        def __is_smaller(value_1, value_2):
            if value_1 is None or value_1 < value_2:
                return True
            return False

        def __is_smaller_or_equal(value_1, value_2):
            if value_1 is None or value_1 <= value_2:
                return True
            return False

        def __is_greater(value_1, value_2):
            if value_1 is None or value_1 <= value_2:
                return False
            return True

        def __is_greater_or_equal(value_1, value_2):
            if value_1 is None or value_1 < value_2:
                return False
            return True

        # Map the function to the operators.
        operator_map = {"<": __is_smaller,
                        "<=": __is_smaller_or_equal,
                        ">": __is_greater,
                        ">=": __is_greater_or_equal}

        try:
            inverse = kwargs["inverse"]
        except KeyError:
            inverse = False

        events = list(self.events)
        for arg in args:
            try:
                key, operator, value = arg.split(" ", 2)
            except ValueError:
                msg = "%s is not a valid filter rule." % arg
                raise ValueError(msg)
            if key == "magnitude":
                temp_events = []
                for event in events:
                    if (event.magnitudes and event.magnitudes[0].mag and
                        operator_map[operator](
                            event.magnitudes[0].mag,
                            float(value))):
                        temp_events.append(event)
                events = temp_events
            elif key in ("longitude", "latitude", "depth", "time"):
                temp_events = []
                for event in events:
                    if (event.origins and key in event.origins[0] and
                        operator_map[operator](
                            event.origins[0].get(key),
                            UTCDateTime(value) if key == 'time' else
                            float(value))):
                        temp_events.append(event)
                events = temp_events
            elif key in ('standard_error', 'azimuthal_gap',
                         'used_station_count', 'used_phase_count'):
                temp_events = []
                for event in events:
                    if (event.origins and event.origins[0].quality and
                        key in event.origins[0].quality and
                        operator_map[operator](
                            event.origins[0].quality.get(key),
                            float(value))):
                        temp_events.append(event)
                events = temp_events
            else:
                msg = "%s is not a valid filter key" % key
                raise ValueError(msg)
        if inverse:
            events = [ev for ev in self.events if ev not in events]
        return Catalog(events=events)

    def copy(self):
        """
        Returns a deepcopy of the Catalog object.

        :rtype: :class:`~obspy.core.stream.Catalog`
        :return: Copy of current catalog.

        .. rubric:: Examples

        1. Create a Catalog and copy it

            >>> from obspy.core.event import readEvents
            >>> cat = readEvents()
            >>> cat2 = cat.copy()

           The two objects are not the same:

            >>> cat is cat2
            False

           But they have equal data:

            >>> cat == cat2
            True

        2. The following example shows how to make an alias but not copy the
           data. Any changes on ``st3`` would also change the contents of
           ``st``.

            >>> cat3 = cat
            >>> cat is cat3
            True
            >>> cat == cat3
            True
        """
        return copy.deepcopy(self)

    def extend(self, event_list):
        """
        Extends the current Catalog object with a list of Event objects.
        """
        if isinstance(event_list, list):
            for _i in event_list:
                # Make sure each item in the list is a event.
                if not isinstance(_i, Event):
                    msg = 'Extend only accepts a list of Event objects.'
                    raise TypeError(msg)
            self.events.extend(event_list)
        elif isinstance(event_list, Catalog):
            self.events.extend(event_list.events)
        else:
            msg = 'Extend only supports a list of Event objects as argument.'
            raise TypeError(msg)

    def write(self, filename, format, **kwargs):
        """
        Saves catalog into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str
        :param format: The file format to use (e.g. ``"QUAKEML"``). See the
            `Supported Formats`_ section below for a list of supported formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            plugin's writer method.

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> catalog = readEvents() # doctest: +SKIP
        >>> catalog.write("example.xml", format="QUAKEML") # doctest: +SKIP

        Writing single events into files with meaningful filenames can be done
        e.g. using event.id

        >>> for ev in catalog:  # doctest: +SKIP
        ...     filename = str(ev.resource_id) + ".xml"
        ...     ev.write(filename, format="QUAKEML") # doctest: +SKIP

        .. rubric:: _`Supported Formats`

        Additional ObsPy modules extend the parameters of the
        :meth:`~obspy.core.event.Catalog.write` method. The following
        table summarizes all known formats currently available for ObsPy.

        Please refer to the `Linked Function Call`_ of each module for any
        extra options available.

        %s
        """
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = EVENT_ENTRY_POINTS_WRITE[format]
            # search writeFormat method for given entry point
            writeFormat = load_entry_point(
                format_ep.dist.key, 'obspy.plugin.event.%s' % (format_ep.name),
                'writeFormat')
        except (IndexError, ImportError):
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(EVENT_ENTRY_POINTS)))
        writeFormat(self, filename, **kwargs)

    @deprecated_keywords({'date_colormap': 'colormap'})
    def plot(self, projection='cyl', resolution='l',
             continent_fill_color='0.9',
             water_fill_color='1.0',
             label='magnitude',
             color='depth',
             colormap=None, show=True, outfile=None,
             **kwargs):  # @UnusedVariable
        """
        Creates preview map of all events in current Catalog object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are:

            * ``"cyl"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)

            Defaults to "cyl"
        :type resolution: str, optional
        :param resolution: Resolution of the boundary database to use. Will be
            based directly to the basemap module. Possible values are:

            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)

            Defaults to ``"l"``
        :type continent_fill_color: Valid matplotlib color, optional
        :param continent_fill_color:  Color of the continents. Defaults to
            ``"0.9"`` which is a light gray.
        :type water_fill_color: Valid matplotlib color, optional
        :param water_fill_color: Color of all water bodies.
            Defaults to ``"white"``.
        :type label: str, optional
        :param label: Events will be labelled based on the chosen property.
            Possible values are:

            * ``"magnitude"``
            * ``None``

            Defaults to ``"magnitude"``
        :type color: str, optional
        :param color: The events will be color-coded based on the chosen
            property. Possible values are:

            * ``"date"``
            * ``"depth"``

            Defaults to ``"depth"``
        :type colormap: str, any matplotlib colormap, optional
        :param colormap: The colormap for color-coding the events.
            The event with the smallest property will have the
            color of one end of the colormap and the event with the biggest
            property the color of the other end with all other events in
            between.
            Defaults to None which will use the default colormap for the date
            encoding and a colormap going from green over yellow to red for the
            depth encoding.
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before showing it.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option, image
            will not be displayed interactively. The given path/filename is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.

        .. rubric:: Examples

        Cylindrical projection for global overview:

        >>> cat = readEvents()
        >>> cat.plot()  # doctest:+SKIP

        .. plot::

            from obspy import readEvents
            cat = readEvents()
            cat.plot()

        Orthographic projection:

        >>> cat.plot(projection="ortho")  # doctest:+SKIP

        .. plot::

            from obspy import readEvents
            cat = readEvents()
            cat.plot(projection="ortho")

        Local (azimuthal equidistant) projection:

        >>> cat.plot(projection="local")  # doctest:+SKIP

        .. plot::

            from obspy import readEvents
            cat = readEvents()
            cat.plot(projection="local")
        """
        from obspy.imaging.maps import plot_basemap
        import matplotlib.pyplot as plt

        if color not in ('date', 'depth'):
            raise ValueError('Events can be color coded by date or depth. '
                             "'%s' is not supported." % (color,))
        if label not in (None, 'magnitude', 'depth'):
            raise ValueError('Events can be labeled by magnitude or events can'
                             ' not be labeled. '
                             "'%s' is not supported." % (label,))

        # lat/lon coordinates, magnitudes, dates
        lats = []
        lons = []
        labels = []
        mags = []
        colors = []
        times = []
        for event in self:
            if not event.origins:
                msg = ("Event '%s' does not have an origin and will not be "
                       "plotted." % str(event.resource_id))
                warnings.warn(msg)
                continue
            if not event.magnitudes:
                msg = ("Event '%s' does not have a magnitude and will not be "
                       "plotted." % str(event.resource_id))
                warnings.warn(msg)
                continue
            origin = event.preferred_origin() or event.origins[0]
            lats.append(origin.latitude)
            lons.append(origin.longitude)
            times.append(origin.time)
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            mag = magnitude.mag
            mags.append(mag)
            labels.append(('  %.1f' % mag) if mag and label == 'magnitude'
                          else '')
            if color == 'date':
                c_ = event.origins[0].get('time')
            else:
                c_ = event.origins[0].get('depth') / 1e3
            colors.append(c_)

        # Create the colormap for date based plotting.
        if colormap is None:
            colormap = plt.get_cmap("RdYlGn_r")

        if len(lons) > 1:
            title = (
                "{event_count} events ({start} to {end}) "
                "- Color codes {colorcode}, size the magnitude".format(
                    event_count=len(self.events),
                    start=min(times).strftime("%Y-%m-%d"),
                    end=max(times).strftime("%Y-%m-%d"),
                    colorcode="origin time" if color == "date" else "depth"))
        else:
            title = "Event at %s" % times[0].strftime("%Y-%m-%d")

        if color not in ("date", "depth"):
            msg = "Invalid option for 'color' parameter (%s)." % color
            raise ValueError(msg)

        min_size = 2
        max_size = 30
        min_size_ = min(mags) - 1
        max_size_ = max(mags) + 1
        if len(lons) > 1:
            frac = [(0.2 + (_i - min_size_)) / (max_size_ - min_size_)
                    for _i in mags]
            size_plot = [(_i * (max_size - min_size)) ** 2 for _i in frac]
        else:
            size_plot = 15.0 ** 2

        fig = plot_basemap(lons, lats, size_plot, colors, labels,
                           projection=projection, resolution=resolution,
                           continent_fill_color=continent_fill_color,
                           water_fill_color=water_fill_color,
                           colormap=colormap, marker="o", title=title,
                           show=False, **kwargs)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
