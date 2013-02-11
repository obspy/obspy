# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.event_header import PickOnset, PickPolarity, EvaluationMode, \
    EvaluationStatus, OriginUncertaintyDescription, OriginDepthType, \
    EventDescriptionType, EventType, EventTypeCertainty, OriginType, \
    AmplitudeCategory, AmplitudeUnit, DataUsedWaveType, MTInversionType, \
    SourceTimeFunctionType, MomentTensorCategory
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import getExampleFile, uncompressFile, _readFromPlugin, \
    NamedTemporaryFile, AttribDict
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.decorator import deprecated_keywords
from pkg_resources import load_entry_point
from uuid import uuid4
import copy
import glob
import inspect
import numpy as np
import os
import re
import urllib2
import warnings
import weakref
import cStringIO
from lxml import etree


EVENT_ENTRY_POINTS = ENTRY_POINTS['event']
ATTRIBUTE_HAS_ERRORS = True


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
    :param format: Format of the file to read. One of ``"QUAKEML"``. See the
        `Supported Formats`_ section below for a full list of supported
        formats.
    :return: A ObsPy :class:`~obspy.core.event.Catalog` object.

    .. rubric:: _`Supported Formats`

    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.event.readEvents` function. The following table
    summarizes all known file formats currently supported by ObsPy.

    Please refer to the `Linked Function Call`_ of each module for any extra
    options available at the import stage.

    =======  ===================  ======================================
    Format   Required Module      _`Linked Function Call`
    =======  ===================  ======================================
    QUAKEML  :mod:`obspy.core`    :func:`obspy.core.quakeml.readQuakeML`
    =======  ===================  ======================================

    Next to the :func:`~obspy.core.event.readEvents` function the
    :meth:`~obspy.core.event.Catalog.write` method of the returned
    :class:`~obspy.core.event.Catalog` object can be used to export the data to
    the file system.
    """
    # if pathname starts with /path/to/ try to search in examples
    if isinstance(pathname_or_url, basestring) and \
       pathname_or_url.startswith('/path/to/'):
        try:
            pathname_or_url = getExampleFile(pathname_or_url[9:])
        except:
            # otherwise just try to read the given /path/to folder
            pass
    # create catalog
    cat = Catalog()
    if pathname_or_url is None:
        # if no pathname or URL specified, return example catalog
        cat = _createExampleCatalog()
    elif not isinstance(pathname_or_url, basestring):
        # not a string - we assume a file-like object
        pathname_or_url.seek(0)
        try:
            # first try reading directly
            catalog = _read(pathname_or_url, format, **kwargs)
            cat.extend(catalog.events)
        except TypeError:
            # if this fails, create a temporary file which is read directly
            # from the file system
            pathname_or_url.seek(0)
            fh = NamedTemporaryFile()
            fh.write(pathname_or_url.read())
            fh.close()
            cat.extend(_read(fh.name, format, **kwargs).events)
            os.remove(fh.name)
        pathname_or_url.seek(0)
    elif pathname_or_url.strip().startswith('<'):
        # XML string
        catalog = _read(cStringIO.StringIO(pathname_or_url), format, **kwargs)
        cat.extend(catalog.events)
    elif "://" in pathname_or_url:
        # URL
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        fh = NamedTemporaryFile(suffix=suffix)
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.close()
        cat.extend(_read(fh.name, format, **kwargs).events)
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in glob.iglob(pathname):
            cat.extend(_read(file, format, **kwargs).events)
        if len(cat) == 0:
            # try to give more specific information why the stream is empty
            if glob.has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not glob.has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
    return cat


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

        >>> from obspy.core.util.types import Enum
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
        ResourceIdentifier(resource_id="event/123456")
        >>> print test_event.creation_info
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
        >>> assert(test_event.description is "1")

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
                setattr(self, key, kwargs.get(key, None))
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
            self.__init__()

        def __str__(self):
            """
            Fairly extensive in an attempt to cover several use cases. It is
            always possible to change it in the child class.
            """
            # Get the attribute and containers that are to be printed. Only not
            # None attributes and non-error attributes are printed. The errors
            # will appear behind the actual value.
            attributes = [_i for _i in self._property_keys if not
                          _i.endswith("_errors") and getattr(self, _i)]
            containers = [_i for _i in self._containers if getattr(self, _i)]

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
                if hasattr(self, error_key) and getattr(self, error_key):
                    err_items = getattr(self, error_key).items()
                    err_items.sort()
                    repr_str += " [%s]" % ', '.join([str(key) + "=" +
                                str(value) for key, value in err_items])
                return repr_str

            # Case 2: Short representation for small objects. Will just print a
            # single line.
            if len(attributes) <= 3 and not containers:
                att_strs = ["%s=%s" % (_i, get_value_repr(_i))
                            for _i in attributes if getattr(self, _i)]
                ret_str += "(%s)" % ", ".join(att_strs)
                return ret_str

            # Case 3: Verbose string representation for large object.
            if attributes:
                format_str = "%" + str(max_length) + "s: %s"
                att_strs = [format_str % (_i, get_value_repr(_i))
                            for _i in attributes if getattr(self, _i)]
                ret_str += "\n\t" + "\n\t".join(att_strs)

            # For the containers just print the number of elements in each.
            if containers:
                # Print delimiter only if there are attributes.
                if attributes:
                    ret_str += '\n\t---------'
                element_str = "%" + str(max_length) + "s: %i Elements"
                ret_str += "\n\t" + \
                    "\n\t".join([element_str %
                    (_i, len(getattr(self, _i)))
                    for _i in containers])
            return ret_str

        def copy(self):
            return copy.deepcopy(self)

        def __repr__(self):
            return self.__str__()

        def __nonzero__(self):
            if any([bool(getattr(self, _i))
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

    # Set the class type name.
    setattr(AbstractEventType, "__name__", class_name)
    return AbstractEventType


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

    :type resource_id: Any hashable object, e.g. numbers, strings, tuples, ...
        optional
    :param resource_id: A unique identifier of the element it refers to. It is
        not verified, that it actually is unique. The user has to take care of
        that. If no resource_id is given, uuid.uuid4() will be used to
        create one which assures uniqueness within one Python run.
    :type prefix: str, optional
    :param prefix: An optional identifier that will be put in front of any
        automatically created resource_id. Will only have an effect if
        resource_id is not given. Makes automatically generated resource_ids
        more reasonable.
    :type referred_object: Python object, optional
    :param referred_object: The object this instance refers to. All instances
        created with the same resource_id will be able to access the object as
        long as at least one instance actual has a reference to it.

    .. rubric:: General Usage

    >>> res_id = ResourceIdentifier('2012-04-11--385392')
    >>> res_id
    ResourceIdentifier(resource_id="2012-04-11--385392")
    >>> # If no resource_id is given it will be generated automatically.
    >>> res_id # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="...")
    >>> # Supplying a prefix will simply prefix the automatically generated
    >>> # resource_id.
    >>> ResourceIdentifier(prefix='event') # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="event/...")

    ResourceIdentifiers can, and oftentimes should, carry a reference to the
    object they refer to. This is a weak reference which means that if the
    object get deleted or runs out of scope, e.g. gets garbage collected, the
    reference will cease to exist.

    >>> event = Event()
    >>> import sys
    >>> ref_count = sys.getrefcount(event)
    >>> res_id = ResourceIdentifier(referred_object=event)
    >>> # The reference does not changed the reference count of the object.
    >>> print ref_count == sys.getrefcount(event)
    True
    >>> # It actually is the same object.
    >>> print event is res_id.getReferredObject()
    True
    >>> # Deleting it, or letting the garbage collector handle the object will
    >>> # invalidate the reference.
    >>> del event
    >>> print res_id.getReferredObject()
    None

    The most powerful ability (and reason why one would want to use a resource
    identifier class in the first place) is that once a ResourceIdentifier with
    an attached referred object has been created, any other ResourceIdentifier
    instances with the same resource_id can retrieve that object. This works
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
    >>> print ref_a.getReferredObject()
    None
    >>> # This instance has an attached object.
    >>> ref_b = ResourceIdentifier(res_id, referred_object=event_object)
    >>> ref_c = ResourceIdentifier(res_id)
    >>> # All ResourceIdentifiers will refer to the same object.
    >>> assert(id(ref_a.getReferredObject()) == obj_id)
    >>> assert(id(ref_b.getReferredObject()) == obj_id)
    >>> assert(id(ref_c.getReferredObject()) == obj_id)

    Any hashable type can be used as a resource_id.

    >>> res_id = ResourceIdentifier((1,3))
    >>> # Using a non-hashable resource_id will result in an error.
    >>> res_id = ResourceIdentifier([1,2])
    Traceback (most recent call last):
        ...
    TypeError: resource_id needs to be a hashable type.
    >>> res_id = ResourceIdentifier()
    >>> res_id.resource_id = [1,2]
    Traceback (most recent call last):
        ...
    TypeError: resource_id needs to be a hashable type.

    The id can be converted to a valid QuakeML ResourceIdentifier by calling
    the convertIDToQuakeMLURI() method. The resulting id will be of the form
        smi:authority_id/prefix/resource_id

    >>> res_id = ResourceIdentifier(prefix='origin')
    >>> res_id.convertIDToQuakeMLURI(authority_id="obspy.org")
    >>> res_id # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="smi:obspy.org/origin/...")
    >>> res_id = ResourceIdentifier('foo')
    >>> res_id.convertIDToQuakeMLURI()
    >>> res_id
    ResourceIdentifier(resource_id="smi:local/foo")
    >>> # A good way to create a QuakeML compatibly ResourceIdentifier from
    >>> # scratch is
    >>> res_id = ResourceIdentifier(prefix='pick')
    >>> res_id.convertIDToQuakeMLURI(authority_id='obspy.org')
    >>> res_id  # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="smi:obspy.org/pick/...")
    >>> # If the given resource_id is already a valid QuakeML
    >>> # ResourceIdentifier, nothing will happen.
    >>> res_id = ResourceIdentifier('smi:test.org/subdir/id')
    >>> res_id
    ResourceIdentifier(resource_id="smi:test.org/subdir/id")
    >>> res_id.convertIDToQuakeMLURI()
    >>> res_id
    ResourceIdentifier(resource_id="smi:test.org/subdir/id")

    ResourceIdentifiers are considered identical if the resource_ids are
    the same.

    >>> # Create two different resource_ids.
    >>> res_id_1 = ResourceIdentifier()
    >>> res_id_2 = ResourceIdentifier()
    >>> assert(res_id_1 != res_id_2)
    >>> # Equalize the resource_ids. NEVER do this. This just an example.
    >>> res_id_2.resource_id = res_id_1.resource_id = 1
    >>> assert(res_id_1 == res_id_2)

    ResourceIdentifier instances can be used as dictionary keys.

    >>> dictionary = {}
    >>> res_id = ResourceIdentifier(resource_id="foo")
    >>> dictionary[res_id] = "bar"
    >>> # The same resource_id can still be used as a key.
    >>> dictionary["foo"] = "bar"
    >>> items = dictionary.items()
    >>> items.sort()
    >>> print items
    [(ResourceIdentifier(resource_id="foo"), 'bar'), ('foo', 'bar')]
    """
    # Class (not instance) attribute that keeps track of all resource
    # identifier throughout one Python run. Will only store weak references and
    # therefore does not interfere with the garbage collection.
    # DO NOT CHANGE THIS FROM OUTSIDE THE CLASS.
    __resource_id_weak_dict = weakref.WeakValueDictionary()

    def __init__(self, resource_id=None, prefix=None, referred_object=None):
        # Create a resource id if None is given and possibly use a prefix.
        if resource_id is None:
            resource_id = str(uuid4())
            if prefix is not None:
                resource_id = "%s/%s" % (prefix, resource_id)
        # Use the setter to assure only hashable ids are set.
        self.__setResourceID(resource_id)
        # Append the referred object in case one is given to the class level
        # reference dictionary.
        if referred_object is not None:
            self.setReferredObject(referred_object)

    def getReferredObject(self):
        """
        Returns the object associated with the resource identifier.

        This works as long as at least one ResourceIdentifier with the same
        resource_id as this instance has an associate object.

        Will return None if no object could be found.
        """
        try:
            return ResourceIdentifier.__resource_id_weak_dict[self]
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
        if not self in ResourceIdentifier.__resource_id_weak_dict:
            ResourceIdentifier.__resource_id_weak_dict[self] = referred_object
            return
        # Otherwise check if the existing element the same as the new one. If
        # it is do nothing, otherwise raise a warning and set the new object as
        # the referred object.
        if ResourceIdentifier.__resource_id_weak_dict[self] is referred_object:
            return
        msg = "The resource identifier already exists and points to " + \
              "another object. It will now point to the object " + \
              "referred to by the new resource identifier."
        # Always raise the warning!
        warnings.warn_explicit(msg, UserWarning, __file__,
                inspect.currentframe().f_back.f_lineno)
        ResourceIdentifier.__resource_id_weak_dict[self] = referred_object

    def convertIDToQuakeMLURI(self, authority_id="local"):
        """
        Converts the current resource_id to a valid QuakeML URI.

        Only an invalid QuakeML ResourceIdentifier string it will be converted
        to a valid one.  Otherwise nothing will happen but after calling this
        method the user can be sure that the resource_id is a valid QuakeML
        URI.

        The resulting resource_id will be of the form
            smi:authority_id/prefix/resource_id

        :type authority_id: str, optional
        :param authority_id: The base url of the resulting string. Defaults to
            ``"local"``.
        """
        quakeml_uri = self.getQuakeMLURI(authority_id=authority_id)
        if quakeml_uri == self.resource_id:
            return
        self.__setResourceID(quakeml_uri)

    def getQuakeMLURI(self, authority_id="local"):
        """
        Returns the resource_id as a valid QuakeML URI if possible. Does not
        change the resource_id itself.

        >>> res_id = ResourceIdentifier("some_id")
        >>> print res_id.getQuakeMLURI()
        smi:local/some_id
        >>> # Did not change the actual resource id.
        >>> print res_id.resource_id
        some_id
        """
        resource_id = self.resource_id
        if str(resource_id).strip() == "":
            resource_id = str(uuid4())
        regex = r"(smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\." + \
                r"\*\(\)_~'][\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*"
        result = re.match(regex, str(resource_id))
        if result is not None:
            return resource_id
        resource_id = 'smi:%s/%s' % (authority_id, str(resource_id))
        # Check once again just to be sure no weird symbols are stored in the
        # resource_id.
        result = re.match(regex, resource_id)
        if result is None:
            msg = "Failed to create a valid QuakeML ResourceIdentifier."
            raise Exception(msg)
        return resource_id

    def copy(self):
        """
        Returns a copy of the ResourceIdentifier.

        >>> res_id = ResourceIdentifier()
        >>> res_id_2 = res_id.copy()
        >>> print res_id is res_id_2
        False
        >>> print res_id == res_id_2
        True
        """
        return ResourceIdentifier(resource_id=self.resource_id)

    def __getResourceID(self):
        return self.__dict__.get("resource_id")

    def __delResourceID(self):
        """
        Deleting is forbidden and will not work.
        """
        msg = "The resource id cannot be deleted."
        raise Exception(msg)

    def __setResourceID(self, resource_id):
        # Check if the resource id is a hashable type.
        try:
            hash(resource_id)
        except TypeError:
            msg = "resource_id needs to be a hashable type."
            raise TypeError(msg)
        self.__dict__["resource_id"] = resource_id

    resource_id = property(__getResourceID, __setResourceID, __delResourceID,
                           "unique identifier of the current instance")

    def __str__(self):
        return self.resource_id

    def __repr__(self):
        return 'ResourceIdentifier(resource_id="%s")' % self.resource_id

    def __eq__(self, other):
        # The type check is necessary due to the used hashing method.
        if type(self) != type(other):
            return False
        if self.resource_id == other.resource_id:
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
        return self.resource_id.__hash__()


__CreationInfo = _eventTypeClassFactory("__CreationInfo",
    class_attributes=[("agency_id", str),
                      ("agency_uri", ResourceIdentifier),
                      ("author", str),
                      ("author_uri", ResourceIdentifier),
                      ("creation_time", UTCDateTime),
                      ("version", str)])


class CreationInfo(__CreationInfo):
    """
    CreationInfo is used to describe author, version, and creation time of a
    resource.

    :type agency_id: str, optional
    :param agency_id: Designation of agency that published a resource.
    :type agency_uri: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param agency_uri: Resource identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param author_uri: Resource identifier of the author of a resource.
    :type creation_time: UTCDateTime, optional
    :param creation_time: Time of creation of a resource.
    :type version: str, optional
    :param version: Version string of a resource.

    >>> info = CreationInfo(author="obspy.org", version="0.0.1")
    >>> print info
    CreationInfo(author='obspy.org', version='0.0.1')
    """


__TimeWindow = _eventTypeClassFactory("__TimeWindow",
    class_attributes=[("begin", float),
                      ("end", float),
                      ("reference", UTCDateTime)])


class TimeWindow(__TimeWindow):
    """
    Describes a time window for amplitude measurements.

    :type begin: float
    :param begin: Time interval before reference point in time window. Unit: s
    :type end: float
    :param end: Time interval after reference point in time window. Unit: s
    :type reference: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param reference: Reference point in time (“central” point).
    """


__CompositeTime = _eventTypeClassFactory("__CompositeTime",
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
    type allows for such complex descriptions.

    :type year: int
    :param year: Year or range of years of the event’s focal time
    :type year_errors: :class:`~obspy.core.util.AttribDict`
    :param year_errors: AttribDict containing error quantities.
    :type month: int
    :param month: Month or range of months of the event’s focal time.
    :type month_errors: :class:`~obspy.core.util.AttribDict`
    :param month_errors: AttribDict containing error quantities.
    :type day: int
    :param day: Day or range of days of the event’s focal time.
    :type day_errors: :class:`~obspy.core.util.AttribDict`
    :param day_errors: AttribDict containing error quantities.
    :type hour: int
    :param hour: Hour or range of hours of the event’s focal time.
    :type hour_errors: :class:`~obspy.core.util.AttribDict`
    :param hour_errors: AttribDict containing error quantities.
    :type minute: int
    :param minute: Minute or range of minutes of the event’s focal time.
    :type minute_errors: :class:`~obspy.core.util.AttribDict`
    :param minute_errors: AttribDict containing error quantities.
    :type second: float
    :param second: Second and fraction of seconds or range of seconds with
    :type second_errors: :class:`~obspy.core.util.AttribDict`
    :param second_errors: AttribDict containing error quantities.

    >>> print CompositeTime(2011, 1, 1)
    CompositeTime(year=2011, month=1, day=1)
    >>> # Can also be instantiated with the uncertainties.
    >>> print CompositeTime(year=2011, year_errors={"uncertainty":1})
    CompositeTime(year=2011 [uncertainty=1])
    """


__Comment = _eventTypeClassFactory("__Comment",
    class_attributes=[("text", str),
                      ("resource_id", ResourceIdentifier),
                      ("creation_info", CreationInfo)])


class Comment(__Comment):
    """
    Comment holds information on comments to a resource as well as author and
    creation time information.

    :type text: str, optional
    :param text: Text of comment.
    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param resource_id: Resource identifier of comment.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`
    :param creation_info: Creation info of comment.

    >>> comment = Comment("Some comment")
    >>> print comment
    Comment(text='Some comment')
    >>> comment.resource_id = "comments/obspy-comment-123456"
    >>> print comment # doctest:+ELLIPSIS
    Comment(text='Some comment', resource_id=ResourceIdentifier(...))
    >>> comment.creation_info = {"author": "obspy.org"}
    >>> print comment.creation_info
    CreationInfo(author='obspy.org')
    """


__WaveformStreamID = _eventTypeClassFactory("__WaveformStreamID",
    class_attributes=[("network_code", str),
                      ("station_code", str),
                      ("channel_code", str),
                      ("location_code", str),
                      ("resource_uri", ResourceIdentifier)])


class WaveformStreamID(__WaveformStreamID):
    """
    Pointer to a stream description in an inventory.

    This is mostly equivalent to the combination of network_code, station_code,
    location_code, and channel_code. However, additional information, e. g.,
    sampling rate, can be referenced by the resource_uri.

    :type network: str
    :param network: Network code.
    :type station: str
    :param station: Station code.
    :type location: str, optional
    :param location: Location code.
    :type channel: str, optional
    :param channel: Channel code.
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
    >>> print stream_id
    WaveformStreamID(network_code='BW', station_code='FUR', channel_code='EHZ')
    >>> stream_id = WaveformStreamID(seed_string="BW.FUR..EHZ")
    >>> print stream_id
    WaveformStreamID(network_code='BW', station_code='FUR', channel_code='EHZ')
    >>> # Can also return the SEED string.
    >>> print stream_id.getSEEDString()
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


__Amplitude = _eventTypeClassFactory("__Amplitude",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("generic_amplitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("type", str),
                      ("category", AmplitudeCategory),
                      ("unit", AmplitudeUnit),
                      ("method_id", ResourceIdentifier),
                      ("period", float),
                      ("snr", float),
                      ("time_window", TimeWindow),
                      ("pick_id", ResourceIdentifier),
                      ("waveform_id", ResourceIdentifier),
                      ("filter_id", ResourceIdentifier),
                      ("scaling_time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("magnitude_hint", str),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Amplitude(__Amplitude):
    """
    This class represents a single amplitude measurement or a measurement of
    the visible end of a record for duration magnitudes.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Pick.
    :type generic_amplitude: float
    :param generic_amplitude: Amplitude value.
    :type generic_amplitude_errors: :class:`~obspy.core.util.AttribDict`
    :param generic_amplitude_errors: AttribDict containing error quantities.
    :type type: str, optional
    :param type: Describes the type of amplitude using the nomenclature from
        Storchak et al. (2003). Possible values are:
            * unspecified amplitude reading (``'A'``),
            * amplitude reading for local magnitude (``'AL'``),
            * amplitude reading for body wave magnitude (``'AB'``),
            * amplitude reading for surface wave magnitude (``'AS'``), and
            * time of visible end of record for duration magnitude (``'END'``).
    :type category: str, optional
    :param category: Amplitude category. Possible values
        are:
            * ``"point"``,
            * ``"mean"``,
            * ``"duration"``,
            * ``"period"``,
            * ``"integral"``,
            * ``"other"``
    :type unit: str, optional
    :param unit: Amplitude unit. Possible values
        are:
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
    :param period: Measured period in the ``time_window`` in case of amplitude
        measurements. Not used for duration magnitude. Unit: s
    :type snr: float, optional
    :param snr: Signal-to-noise ratio of the spectrogram at the location the
        amplitude was measured.
    :type time_window: :class:`~obspy.core.event.TimeWindow`, optional
    :param time_window: Description of the time window used for amplitude
        measurement. Mandatory for duration magnitudes.
    :type pick_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param pick_id: Refers to the ``resource_id`` of an associated
        :class:`~obspy.core.event.Pick` object.
    :type waveform_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param waveform_id: Identifies the waveform stream on which the amplitude
        was measured.
    :type filter_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param filter_id: Identifies the filter or filter setup used for filtering
        the waveform stream referenced by ``waveform_id``.
    :type scaling_time: :class:`~obspy.core.UTCDateTime`, optional
    :param scaling_time: Scaling time for amplitude measurement.
    :type scaling_time_errors: :class:`~obspy.core.util.AttribDict`
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
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__Pick = _eventTypeClassFactory("__Pick",
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
    This class contains various attributes commonly used to describe a single
    pick, e.g. time, waveform id, onset, phase hint, polarity, etc

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Pick.
    :type time: :class:`~obspy.core.UTCDateTime`
    :param time: Pick time.
    :type time_errors: :class:`~obspy.core.util.AttribDict`
    :param time_errors: AttribDict containing error quantities.
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`
    :param waveform_id: Identifies the waveform stream.
    :type filter_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param filter_id: Identifies the filter setup used.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the method used to get the pick.
    :type horizontal_slowness: float, optional
    :param horizontal_slowness: Describes the horizontal slowness of the Pick.
    :type horizontal_slowness_errors: :class:`~obspy.core.util.AttribDict`
    :param horizontal_slowness_errors: AttribDict containing error quantities.
    :type backazimuth: float, optional
    :param backazimuth: Describes the backazimuth of the Pick.
    :type backazimuth_errors: :class:`~obspy.core.util.AttribDict`
    :param backazimuth_errors: AttribDict containing error quantities.
    :type slowness_method_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param slowness_method_id: Identifies the method used to derive the
        slowness.
    :type onset: str, optional
    :param onset: Describes the pick onset type. Allowed values are:
            * ``"emergent"``
            * ``"impulsive"``
            * ``"questionable"``
    :type phase_hint: str, optional
    :param phase_hint: Free-form text field describing the phase. In QuakeML
        this is a separate type but it just contains a single field containing
        the phase as a string.
    :type polarity: str, optional
    :param polarity: Describes the pick onset type. Allowed values are:
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
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__Arrival = _eventTypeClassFactory("__Arrival",
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
    the residual, respectively.  Additional pick attributes like the horizontal
    slowness and backazimuth of the observed wave - especially if derived from
    array data - may further constrain the nature of the arrival.
    [from the QuakeML Basic Event Description, Version 1.1, page 38]

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Arrival.
    :type pick_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param pick_id: Refers to a resource_id of associated Pick.
    :type phase: str
    :param phase: Phase identification. Free-form text field describing the
        phase. In QuakeML this is a separate type but it just contains a single
        field containing the phase as a string.
    :type time_correction: float, optional
    :param time_correction: Time correction value in seconds.
    :type azimuth: float, optional
    :param azimuth: Azimuth of station as seen from the epicenter in degree.
    :type distance: float, optional
    :param distance: Epicentral distance in degree.
    :type takeoff_angle: float, optional
    :param takeoff_angle: Take-off angle.
    :type takeoff_angle_errors: :class:`~obspy.core.util.AttribDict`
    :param takeoff_angle_errors: AttribDict containing error quantities.
    :type time_residual: float, optional
    :param time_residual: Residual between observed and expected arrival time
        assuming proper phase identification and given the earth_model_id of
        the Origin in seconds.
    :type horizontal_slowness_residual: float, optional
    :param horizontal_slowness_residual: Residual of horizontal slowness in
        seconds per degree.
    :type backazimuth_residual: float, optional
    :param backazimuth_residual: Residual of backazimuth in degree.
    :type time_weight: float, optional
    :param time_weight: Weight of this Arrival in the computation of the
        associated Origin.
    :type horizontal_slowness_weight: float, optional
    :param horizontal_slowness_weight: Weight of horizontal slowness.
    :type backazimuth_weight: float, optional
    :param backazimuth_weight: Weight of backazimuth.
    :type earth_model_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param earth_model_id: Earth model which is used for the association of
        Arrival to Pick and computation of the residuals.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__OriginQuality = _eventTypeClassFactory("__OriginQuality",
    class_attributes=[("associated_phase_count", int),
                      ("used_phase_count", int),
                      ("associated_station_count", int),
                      ("used_station_count", int),
                      ("depth_phase_count", int),
                      ("standard_error", float),
                      ("azimuthal_gap", float),
                      ("secondary_azimuthal_gap", float),
                      ("ground_truth_level", str),
                      ("maximum_distance", float),
                      ("minimum_distance", float),
                      ("median_distance", float)])


class OriginQuality(__OriginQuality):
    """
    This type contains various attributes commonly used to describe the quality
    of an origin, e.g., errors, azimuthal coverage, etc. Origin objects have
    an optional attribute of the type OriginQuality.

    :type associated_phase_count: int, optional
    :param associated_phase_count: Number of associated phases, regardless of
        their use for origin computation.
    :type used_phase_count: int, optional
    :param used_phase_count: Number of defining phases, i.e., phase
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
        from epicenter. Unit: deg
    :type secondary_azimuthal_gap: float, optional
    :param secondary_azimuthal_gap: Secondary azimuthal gap in station
        distribution, i. e., the largest azimuthal gap a station closes.
        Unit: deg
    :type ground_truth_level: str, optional
    :param ground_truth_level: String describing ground-truth level, e. g. GT0,
        GT5, etc.
    :type minimum_distance: float, optional
    :param minimum_distance: Distance Epicentral distance of station closest to
        the epicenter. Unit: deg
    :type maximum_distance: float, optional
    :param maximum_distance: Distance Epicentral distance of station farthest
        from the epicenter. Unit: deg
    :type median_distance: float, optional
    :param median_distance: Distance Median epicentral distance of used
        stations. Unit: deg
    """


__ConfidenceEllipsoid = _eventTypeClassFactory("__ConfidenceEllipsoid",
    class_attributes=[("semi_major_axis_length", float),
                      ("semi_minor_axis_length", float),
                      ("semi_intermediate_axis_length", float),
                      ("major_axis_plunge", float),
                      ("major_axis_azimuth", float),
                      ("major_axis_rotation", float)])


class ConfidenceEllipsoid(__ConfidenceEllipsoid):
    """
    This class represents a description of the location uncertainty as a
    confidence ellipsoid with arbitrary orientation in space.

    :param semi_major_axis_length: Largest uncertainty, corresponding to the
        semi-major axis of the confidence ellipsoid. Unit: m
    :param semi_minor_axis_length: Smallest uncertainty, corresponding to the
        semi-minor axis of the confidence ellipsoid. Unit: m
    :param semi_intermediate_axis_length: Uncertainty in direction orthogonal
        to major and minor axes of the confidence ellipsoid. Unit: m
    :param major_axis_plunge: Plunge angle of major axis of confidence
        ellipsoid. Unit: deg
    :param major_axis_azimuth: Azimuth angle of major axis of confidence
        ellipsoid. Unit: deg
    :param major_axis_rotation: This angle describes a rotation about the
        confidence ellipsoid’s major axis which is required to define the
        direction of the ellipsoid’s minor axis. A zero majorAxisRotation angle
        means that the minor axis lies in the plane spanned by the major axis
        and the vertical. Unit: deg
    """


__OriginUncertainty = _eventTypeClassFactory("__OriginUncertainty",
    class_attributes=[("horizontal_uncertainty", float),
                      ("min_horizontal_uncertainty", float),
                      ("max_horizontal_uncertainty", float),
                      ("azimuth_max_horizontal_uncertainty", float),
                      ("confidence_ellipsoid", ConfidenceEllipsoid),
                      ("preferred_description", OriginUncertaintyDescription)])


class OriginUncertainty(__OriginUncertainty):
    """
    This class describes the location uncertainties of an origin.

    The uncertainty can be described either as a simple circular horizontal
    uncertainty, an uncertainty ellipse according to IMS1.0, or a confidence
    ellipsoid. The preferred variant can be given in the attribute
    ``preferred_description``.

    :type horizontal_uncertainty: float, optional
    :param horizontal_uncertainty: Circular confidence region, given by single
        value of horizontal uncertainty. Unit: m
    :type min_horizontal_uncertainty: float, optional
    :param min_horizontal_uncertainty: Semi-major axis of confidence ellipse.
        Unit: m
    :type max_horizontal_uncertainty: float, optional
    :param max_horizontal_uncertainty: Semi-minor axis of confidence ellipse.
        Unit: m
    :type azimuth_max_horizontal_uncertainty: float, optional
    :param azimuth_max_horizontal_uncertainty: Azimuth of major axis of
        confidence ellipse. Unit: deg
    :type confidence_ellipsoid: :class:`~obspy.core.event.ConfidenceEllipsoid`,
        optional
    :param confidence_ellipsoid: Confidence ellipsoid
    :type preferred_description: str, optional
    :param preferred_description: Preferred uncertainty description. Allowed
        values are the following:
            * horizontal uncertainty
            * uncertainty ellipse
            * confidence ellipsoid
            * probability density function
    """


__Origin = _eventTypeClassFactory("__Origin",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", UTCDateTime, ATTRIBUTE_HAS_ERRORS),
                      ("latitude", float, ATTRIBUTE_HAS_ERRORS),
                      ("longitude", float, ATTRIBUTE_HAS_ERRORS),
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
    earthquake hypocenter, as well as additional meta-information.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Origin.
    :type time: :class:`~obspy.core.UTCDateTime`
    :param time: Focal time.
    :type time_errors: :class:`~obspy.core.util.AttribDict`
    :param time_errors: AttribDict containing error quantities.
    :type latitude: float
    :param latitude: Hypocenter latitude. Unit: deg
    :type latitude_errors: :class:`~obspy.core.util.AttribDict`
    :param latitude_errors: AttribDict containing error quantities.
    :type longitude: float
    :param longitude: Hypocenter longitude. Unit: deg
    :type longitude_errors: :class:`~obspy.core.util.AttribDict`
    :param longitude_errors: AttribDict containing error quantities.
    :type depth: float, optional
    :param depth: Depth of hypocenter. Unit: m
    :type depth_errors: :class:`~obspy.core.util.AttribDict`
    :param depth_errors: AttribDict containing error quantities.
    :type depth_type: str, optional
    :param depth_type: Type of depth determination. Allowed values are the
        following:
            * ``"from location"``
            * ``"constrained by depth phases"``
            * ``"constrained by direct phases"``
            * ``"operator assigned"``
            * ``"other"``
    :type time_fixed: bool, optional
    :param time_fixed: ``True`` if focal time was kept fixed for computation
        of the Origin.
    :type epicenter_fixed: bool, optional
    :param epicenter_fixed: ``True`` if epicenter was kept fixed for
        computation of Origin.
    :type reference_system_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param reference_system_id: Identifies the reference system used for
        hypocenter determination.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the method used for locating the event.
    :type earth_model_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param earth_model_id: Identifies the earth model used in ``method_id``.
    :type composite_times: list of :class:`~obspy.core.event.CompositeTime`,
        optional
    :param composite_times: Supplementary information on time of rupture start.
        Complex descriptions of focal times of historic event are possible,
        see description of the :class:`~obspy.core.event.CompositeTime` class.
    :type quality: :class:`~obspy.core.event.OriginQuality`, optional
    :param quality: Additional parameters describing the quality of an origin
        determination.
    :type type: str, optional
    :param type: Describes the origin type. Allowed values are the
        following:
            * ``"rupture start"``
            * ``"centroid"``
            * ``"rupture end"``
            * ``"hypocenter"``
            * ``"amplitude"``
            * ``"macroseismic"``
    :type region: str, optional
    :param region: Region name.
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
        resource_id: ResourceIdentifier(resource_id="smi:ch.ethz.sed/...")
               time: UTCDateTime(1970, 1, 1, 0, 0)
           latitude: 12.0 [confidence_level=95.0]
          longitude: 42.0
         depth_type: 'from location'
    """


__StationMagnitudeContribution = _eventTypeClassFactory(
    "__StationMagnitudeContribution",
    class_attributes=[("station_magnitude_id", ResourceIdentifier),
                      ("residual", float),
                      ("weight", float)])


class StationMagnitudeContribution(__StationMagnitudeContribution):
    """
    This class describes the weighting of magnitude values from Magnitude
    estimations.

    :type station_magnitude_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param station_magnitude_id: Refers to the resource_id of a
        StationMagnitude object.
    :type residual: float, optional
    :param residual: Residual of magnitude computation.
    :type weight: float, optional
    :param weight: Weight of the magnitude value from a StationMagnitude object
        for computing the magnitude value in class Magnitude.
    """


__Magnitude = _eventTypeClassFactory("__Magnitude",
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
    Describes a magnitude which can, but need not be associated with an Origin.

    Association with an origin is expressed with the optional attribute
    ``origin_id``. It is either a combination of different magnitude
    estimations, or it represents the reported magnitude for the given Event.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of Magnitude.
    :type mag: float
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.StationMagnitude`. If no estimations are
        available, this value can represent the reported magnitude.
    :type mag_errors: :class:`~obspy.core.util.AttribDict`
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
        should avoid to give contradictory information in method_id and type.
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
    :type evaluation_status: :class:`~obspy.core.event.EvaluationStatus`,
        optional
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
    """


__StationMagnitude = _eventTypeClassFactory("__StationMagnitude",
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
    :type origin_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param origin_id: Reference to an origin’s ``resource_id`` if the
        StationMagnitude has an associated :class:`~obspy.core.event.Origin`.
    :type mag: float
    :param mag: Estimated magnitude.
    :type mag_errors: :class:`~obspy.core.util.AttribDict`
    :param mag_errors: AttribDict containing error quantities.
    :type station_magnitude_type: str, optional
    :param station_magnitude_type: Describes the type of magnitude. This is a
        free-text field because it is impossible to cover all existing
        magnitude type designations with an enumeration. Possible values are:
            * unspecified magnitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type amplitude_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param amplitude_id: Identifies the data source of the StationMagnitude.
        For magnitudes derived from amplitudes in waveforms (e. g.,
        local magnitude ML), amplitude_id points to resource_id in class
        :class:`obspy.core.event.Amplitude`.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`, optional
    :param waveform_id: Identifies the waveform stream.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__EventDescription = _eventTypeClassFactory("__EventDescription",
    class_attributes=[("text", str),
                      ("type", EventDescriptionType)])


class EventDescription(__EventDescription):
    """
    Free-form string with additional event description. This can be a
    well-known name, like 1906 San Francisco Earthquake. A number of categories
    can be given in type.

    :type text: str, optional
    :param text: Free-form text with earthquake description.
    :type event_description_type: str, optional
    :param event_description_type: Category of earthquake description. Values
        can be taken from the following:
            * ``"felt report"``
            * ``"Flinn-Engdahl region"``
            * ``"local time"``
            * ``"tectonic summary"``
            * ``"nearest cities"``
            * ``"earthquake name"``
            * ``"region name"``
    """


__Tensor = _eventTypeClassFactory("__Tensor",
    class_attributes=[("m_rr", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_pp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rt", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_rp", float, ATTRIBUTE_HAS_ERRORS),
                      ("m_tp", float, ATTRIBUTE_HAS_ERRORS)])


class Tensor(__Tensor):
    """
    The Tensor class represents the six moment-tensor elements m_rr, m_tt,
    m_pp, m_rt, m_rp, m_tp, where r is up, t is south, and p is east. See
    Aki & Richards (1980) for conversions to other coordinate systems.

    :type m_rr: float
    :param m_rr: Moment-tensor element Mrr. Unit: Nm
    :type m_rr_errors: :class:`~obspy.core.util.AttribDict`
    :param m_rr_errors: AttribDict containing error quantities.
    :type m_tt: float
    :param m_tt: Moment-tensor element Mtt. Unit: Nm
    :type m_tt_errors: :class:`~obspy.core.util.AttribDict`
    :param m_tt_errors: AttribDict containing error quantities.
    :type m_pp: float
    :param m_pp: Moment-tensor element Mpp. Unit: Nm
    :type m_pp_errors: :class:`~obspy.core.util.AttribDict`
    :param m_pp_errors: AttribDict containing error quantities.
    :type m_rt: float
    :param m_rt: Moment-tensor element Mrt. Unit: Nm
    :type m_rt_errors: :class:`~obspy.core.util.AttribDict`
    :param m_rt_errors: AttribDict containing error quantities.
    :type m_rp: float
    :param m_rp: Moment-tensor element Mrp. Unit: Nm
    :type m_rp_errors: :class:`~obspy.core.util.AttribDict`
    :param m_rp_errors: AttribDict containing error quantities.
    :type m_tp: float
    :param m_tp: Moment-tensor element Mtp. Unit: Nm
    :type m_tp_errors: :class:`~obspy.core.util.AttribDict`
    :param m_tp_errors: AttribDict containing error quantities.
    """


__DataUsed = _eventTypeClassFactory("__DataUsed",
    class_attributes=[("wave_type", DataUsedWaveType),
                      ("station_count", int),
                      ("component_count", int),
                      ("shortest_period", float),
                      ("longest_period", float)])


class DataUsed(__DataUsed):
    """
    This class describes the type of data that has been used for a
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
    """


__SourceTimeFunction = _eventTypeClassFactory("__SourceTimeFunction",
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
    """


__NodalPlane = _eventTypeClassFactory("__NodalPlane",
    class_attributes=[("strike", float, ATTRIBUTE_HAS_ERRORS),
                      ("dip", float, ATTRIBUTE_HAS_ERRORS),
                      ("rake", float, ATTRIBUTE_HAS_ERRORS)])


class NodalPlane(__NodalPlane):
    """
    This class describes a nodal plane using the attributes strike, dip, and
    rake. For a definition of the angles see Aki & Richards (1980).

    :type strike: float
    :param strike: Strike angle of nodal plane. Unit: deg
    :type strike_errors: :class:`~obspy.core.util.AttribDict`
    :param strike_errors: AttribDict containing error quantities.
    :type dip: float
    :param dip: Dip angle of nodal plane. Unit: deg
    :type dip_errors: :class:`~obspy.core.util.AttribDict`
    :param dip_errors: AttribDict containing error quantities.
    :type rake: float
    :param rake: Rake angle of nodal plane. Unit: deg
    :type rake_errors: :class:`~obspy.core.util.AttribDict`
    :param rake_errors: AttribDict containing error quantities.
    """


__Axis = _eventTypeClassFactory("__Axis",
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
        principal-axes system. Unit: deg
    :type azimuth_errors: :class:`~obspy.core.util.AttribDict`
    :param azimuth_errors: AttribDict containing error quantities.
    :type plunge: float
    :param plunge: Plunge of eigenvector of moment tensor expressed in
        principal-axes system. Unit: deg
    :type plunge_errors: :class:`~obspy.core.util.AttribDict`
    :param plunge_errors: AttribDict containing error quantities.
    :type length: float
    :param length: Eigenvalue of moment
    :type length_errors: :class:`~obspy.core.util.AttribDict`
    :param length_errors: AttribDict containing error quantities.
    """


__NodalPlanes = _eventTypeClassFactory("__NodalPlanes",
    class_attributes=[("nodal_plane_1", NodalPlane),
                      ("nodal_plane_2", NodalPlane),
                      ("preferred_plane", int)])


class NodalPlanes(__NodalPlanes):
    """
    This class describes the nodal planes of a double-couple moment-tensor
    solution. The attribute preferredPlane can be used to define which plane is
    the preferred one.

    :type nodal_plane_1: :class:`~obspy.core.event.NodalPlane`, optional
    :param nodal_plane_1: First nodal plane of double-couple moment tensor
        solution.
    :type nodal_plane_2: :class:`~obspy.core.event.NodalPlane`, optional
    :param nodal_plane_2: Second nodal plane of double-couple moment tensor
        solution.
    :type preferred_plane: ``1`` or ``2``, optional
    :param preferred_plane: Indicator for preferred nodal plane of moment
        tensor solution. It can take integer values ``1`` or ``2``.
    """


__PrincipalAxes = _eventTypeClassFactory("__PrincipalAxes",
    class_attributes=[("t_axis", Axis),
                      ("p_axis", Axis),
                      ("n_axis", Axis)])


class PrincipalAxes(__PrincipalAxes):
    """
    This class describes the principal axes of a double-couple moment tensor
    solution. t_axis and p_axis are required, while n_axis is optional.

    :type t_axis: :class:`~obspy.core.event.Axis`
    :param t_axis: T axis of a double-couple moment tensor solution.
    :type p_axis: :class:`~obspy.core.event.Axis`
    :param p_axis: P axis of a double-couple moment tensor solution.
    :type n_axis: :class:`~obspy.core.event.Axis`, optional
    :param n_axis: N axis of a double-couple moment tensor solution.
    """


__MomentTensor = _eventTypeClassFactory("__MomentTensor",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("data_used", DataUsed),
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
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=['comments'])


class MomentTensor(__MomentTensor):
    """
    This class represents a moment tensor solution for an Event. It is part of
    a FocalMechanism description.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of MomentTensor.
    :type data_used: :class:`~obspy.core.event.DataUsed`, optional
    :param data_used: Describes waveform data used for moment-tensor inversion.
    :type derived_origin_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param derived_origin_id: Refers to the resource_id of the Origin derived
        in the moment tensor inversion.
    :type moment_magnitude_id: :class:`~obspy.core.event.ResourceIdentifier`,
        optional
    :param moment_magnitude_id: Refers to the resource_id of the Magnitude
        object which represents the derived moment magnitude.
    :type scalar_moment: float, optional
    :param scalar_moment: Scalar moment as derived in moment tensor inversion.
        Unit: Nm
    :type scalar_moment_errors: :class:`~obspy.core.util.AttribDict`
    :param scalar_moment_errors: AttribDict containing error quantities.
    :type tensor: :class:`~obspy.core.event.Tensor`, optional
    :param tensor: Tensor object holding the moment tensor elements.
    :type variance: float, optional
    :param variance: Variance of moment tensor inversion.
    :type variance_reduction: float, optional
    :param variance_reduction: Variance reduction of moment tensor inversion.
    :type double_couple: float, optional
    :param double_couple: Double couple parameter obtained from moment tensor
        inversion (fractional value between 0 and 1).
    :type clvd: float, optional
    :param clvd: CLVD (compensated linear vector dipole) parameter obtained
        from moment tensor inversion (fractional value between 0 and 1).
    :type iso: float, optional
    :param iso: Isotropic part obtained from moment tensor inversion
        (fractional value between 0 and 1).
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
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Resource identifier of the method used for moment-tensor
        inversion.
    :type category: str, optional
    :param category: Moment tensor category. Values can be taken from the
        following:
            * ``"teleseismic"``,
            * ``"regional"``
    :type inversion_type: str, optional
    :param inversion_type: Moment tensor inversion type. Values can be taken
        from the following:
            * ``"general"``,
            * ``"zero trace"``,
            * ``"double couple"``
    :type evaluation_mode: str, optional
    :param evaluation_mode: Evaluation mode of MomentTensor. Allowed values are
        the following:
            * ``"manual"``
            * ``"automatic"``
    :type evaluation_status: :class:`~obspy.core.event.EvaluationStatus`,
        optional
    :param evaluation_status: Evaluation status of MomentTensor. Allowed values
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
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__FocalMechanism = _eventTypeClassFactory("__FocalMechanism",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID),
                      ("triggering_origin_id", ResourceIdentifier),
                      ("nodal_planes", NodalPlanes),
                      ("principal_axes", PrincipalAxes),
                      ("azimuthal_gap", float),
                      ("station_polarity_count", int),
                      ("misfit", float),
                      ("station_distribution_ratio", float),
                      ("method_id", ResourceIdentifier),
                      ("moment_tensor", MomentTensor),
                      ("creation_info", CreationInfo)],
    class_contains=['comments'])


class FocalMechanism(__FocalMechanism):
    """
    This class describes the focal mechanism of an Event.

    It includes different descriptions like nodal planes, principal axes, and a
    moment tensor. The moment tensor description is provided by objects of the
    class MomentTensor which can be specified as child elements of
    FocalMechanism.

    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`
    :param resource_id: Resource identifier of FocalMechanism.
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
    :param station_polarity_count: Number of station polarities used for
        determination of focal mechanism.
    :type misfit: float, optional
    :param misfit: Fraction of misfit polarities in a first-motion focal
        mechanism determination. Fractional value between 0 and 1.
    :type station_distribution_ratio: float, optional
    :param station_distribution_ratio: Station distribution ratio (STDR)
        parameter. Indicates how the stations are distributed about the focal
        sphere. Fractional value between 0 and 1.
    :type method_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param method_id: Resource identifier of the method used for determination
        of the focal mechanism.
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`, optional
    :param waveform_id: Identifies the waveform stream.
    :type moment_tensor: :class:`~obspy.core.event.MomentTensor`, optional
    :param moment_tensor: Moment tensor description for this focal mechanism.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """


__Event = _eventTypeClassFactory("__Event",
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
    :type event_type: str, optional
    :param event_type: Describes the type of an event. Allowed values are the
        following:
            * ``"earthquake"``
            * ``"induced earthquake"``
            * ``"quarry blast"``
            * ``"explosion"``
            * ``"chemical explosion"``
            * ``"nuclear explosion"``
            * ``"landslide"``
            * ``"rockslide"``
            * ``"snow avalanche"``
            * ``"debris avalanche"``
            * ``"mine collapse"``
            * ``"building collapse"``
            * ``"volcanic eruption"``
            * ``"meteor impact"``
            * ``"plane crash"``
            * ``"sonic boom"``
            * ``"not existing"``
            * ``"null"``
            * ``"other"``
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
    """
    def __eq__(self, other):
        """
        Implements rich comparison of Event objects for "==" operator.

        Events are the same, if the have the same id.
        """
        # check if other object is a Event
        if not isinstance(other, Event):
            return False
        if (self.resource_id != other.resource_id):
            return False
        return True

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
        return "Event:\t%s\n\n%s" % (self.short_str(),
            "\n".join(super(Event, self).__str__().split("\n")[1:]))

    def preferred_origin(self):
        """
        Returns the preferred origin
        """
        try:
            return ResourceIdentifier(self.preferred_origin_id).\
                getReferredObject()
        except KeyError:
            return None

    def preferred_magnitude(self):
        """
        Returns the preferred origin
        """
        try:
            return ResourceIdentifier(self.preferred_magnitude_id).\
                getReferredObject()
        except KeyError:
            return None

    def preferred_focal_mechanism(self):
        """
        Returns the preferred origin
        """
        try:
            return ResourceIdentifier(self.preferred_focal_mechanism_id).\
                getReferredObject()
        except KeyError:
            return None


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

    def __getitem__(self, index):
        """
        __getitem__ method of the Catalog object.

        :return: Event objects
        """
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
        if not isinstance(index, basestring):
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
                   "'print CatalogObject.__str__(print_all=True)'"
        return out

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

    def filter(self, *args):
        """
        Returns a new Catalog object only containing Events which match the
        specified filter rules.

        :rtype: :class:`~obspy.core.stream.Catalog`
        :return: Filtered catalog. Only the Catalog object is a copy, the
            events are only references.

        >>> from obspy.core.event import readEvents
        >>> cat = readEvents()
        >>> print cat
        3 Event(s) in Catalog:
        2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4 mb | manual
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0 ML | manual
        >>> cat2 = cat.filter("magnitude >= 4.0", "latitude < 40.0")
        >>> print cat2
        1 Event(s) in Catalog:
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        >>> cat3 = cat.filter("time > 2012-04-04T14:10", \
                              "time < 2012-04-04T14:20")
        >>> print cat3
        1 Event(s) in Catalog:
        2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3 ML | manual
        """
        # Helper functions.
        def __is_smaller(value_1, value_2):
            if value_1 < value_2:
                return True
            return False

        def __is_smaller_or_equal(value_1, value_2):
            if value_1 <= value_2:
                return True
            return False

        def __is_greater(value_1, value_2):
            if value_1 > value_2:
                return True
            return False

        def __is_greater_or_equal(value_1, value_2):
            if value_1 >= value_2:
                return True
            return False

        # Map the function to the operators.
        operator_map = {"<": __is_smaller,
                        "<=": __is_smaller_or_equal,
                        ">": __is_greater,
                        ">=": __is_greater_or_equal}

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
                    if event.magnitudes and event.magnitudes[0].mag and \
                        operator_map[operator](event.magnitudes[0].mag,
                                               float(value)):
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

        :type filename: string
        :param filename: The name of the file to write.
        :type format: string
        :param format: The format to write must be specified. One of
            ``"QUAKEML"``. See the `Supported Formats`_ section below for a
            full list of supported formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            waveform writer method.

        .. rubric:: Example

        >>> from obspy.core.event import readEvents
        >>> catalog = readEvents() # doctest: +SKIP
        >>> catalog.write("example.xml", format="QUAKEML") # doctest: +SKIP

        Writing single events into files with meaningful filenames can be done
        e.g. using event.id

        >>> for ev in catalog: #doctest: +SKIP
        ...     ev.write("%s.xml" % ev.id, format="QUAKEML") #doctest: +SKIP

        .. rubric:: _`Supported Formats`

        Additional ObsPy modules extend the parameters of the
        :meth:`~obspy.core.event.Catalog.write` method. The following
        table summarizes all known formats currently available for ObsPy.

        Please refer to the `Linked Function Call`_ of each module for any
        extra options available.

        =======  ===================  =======================================
        Format   Required Module      _`Linked Function Call`
        =======  ===================  =======================================
        QUAKEML  :mod:`obspy.core`    :func:`obspy.core.quakeml.writeQuakeML`
        =======  ===================  =======================================
        """
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = EVENT_ENTRY_POINTS[format]
            # search writeFormat method for given entry point
            writeFormat = load_entry_point(format_ep.dist.key,
                'obspy.plugin.event.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError):
            msg = "Format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format, ', '.join(EVENT_ENTRY_POINTS)))
        writeFormat(self, filename, **kwargs)

    @deprecated_keywords({'date_colormap': 'colormap'})
    def plot(self, projection='cyl', resolution='l',
             continent_fill_color='0.8',
             water_fill_color='white',
             label='magnitude',
             color='date',
             colormap=None, **kwargs):  # @UnusedVariable
        """
        Creates preview map of all events in current Catalog object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are
            * ``"cyl"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)
            Defaults to "cyl"
        :type resolution: str, optional
        :param resolution: Resolution of the boundary database to use. Will be
            based directly to the basemap module. Possible values are
            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)
            Defaults to ``"l"``
        :type continent_fill_color: Valid matplotlib color, optional
        :param continent_fill_color:  Color of the continents. Defaults to
            ``"0.8"`` which is a light gray.
        :type water_fill_color: Valid matplotlib color, optional
        :param water_fill_color: Color of all water bodies.
            Defaults to ``"white"``.
        :type label: str, optional
        :param label:Events will be labeld based on the chosen property.
            Possible values are
            * ``"magnitude"``
            * ``None``
            Defaults to ``"magnitude"``
        :type color: str, optional
        :param color:The events will be color-coded based on the chosen
            proberty. Possible values are
            * ``"date"``
            * ``"depth"``
            Defaults to ``"date"``
        :type colormap: str, optional, any matplotlib colormap
        :param colormap: The colormap for color-coding the events.
            The event with the smallest property will have the
            color of one end of the colormap and the event with the biggest
            property the color of the other end with all other events in
            between.
            Defaults to None which will use the default colormap.

        .. rubric:: Example

        >>> cat = readEvents(\ # doctest:+SKIP
            "http://www.seismicportal.eu/services/event/search?magMin=8.0")
        >>> cat.plot() # doctest:+SKIP
        """
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import matplotlib as mpl

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
        for event in self:
            lats.append(event.origins[0].latitude)
            lons.append(event.origins[0].longitude)
            mag = event.magnitudes[0].mag
            mags.append(mag)
            labels.append(('    %.1f' % mag) if mag and label == 'magnitude'
                          else '')
            colors.append(event.origins[0].get('time' if color == 'date' else
                                               color))
        min_color = min(colors)
        max_color = max(colors)

        # Create the colormap for date based plotting.
        colormap = plt.get_cmap(colormap)
        scal_map = ScalarMappable(norm=Normalize(min_color, max_color),
                                  cmap=colormap)
        scal_map.set_array(np.linspace(0, 1, 1))

        fig = plt.figure()
        map_ax = fig.add_axes([0.03, 0.13, 0.94, 0.82])
        cm_ax = fig.add_axes([0.03, 0.05, 0.94, 0.05])
        plt.sca(map_ax)

        if projection == 'cyl':
            map = Basemap(resolution=resolution)
        elif projection == 'ortho':
            map = Basemap(projection='ortho', resolution=resolution,
                          area_thresh=1000.0, lat_0=sum(lats) / len(lats),
                          lon_0=sum(lons) / len(lons))
        elif projection == 'local':
            if min(lons) < -150 and max(lons) > 150:
                max_lons = max(np.array(lons) % 360)
                min_lons = min(np.array(lons) % 360)
            else:
                max_lons = max(lons)
                min_lons = min(lons)
            lat_0 = (max(lats) + min(lats)) / 2.
            lon_0 = (max_lons + min_lons) / 2.
            if lon_0 > 180:
                lon_0 -= 360
            deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
            deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
            height = (max(lats) - min(lats)) * deg2m_lat
            width = (max_lons - min_lons) * deg2m_lon
            margin = 0.2 * (width + height)
            height += margin
            width += margin
            map = Basemap(projection='aeqd', resolution=resolution,
                          area_thresh=1000.0, lat_0=lat_0, lon_0=lon_0,
                          width=width, height=height)
            # not most elegant way to calculate some round lats/lons

            def linspace2(val1, val2, N):
                """
                returns around N 'nice' values between val1 and val2
                """
                dval = val2 - val1
                round_pos = int(round(-np.log10(1. * dval / N)))
                delta = round(2. * dval / N, round_pos) / 2
                new_val1 = np.ceil(val1 / delta) * delta
                new_val2 = np.floor(val2 / delta) * delta
                N = (new_val2 - new_val1) / delta + 1
                return np.linspace(new_val1, new_val2, N)
            N1 = int(np.ceil(height / max(width, height) * 8))
            N2 = int(np.ceil(width / max(width, height) * 8))
            map.drawparallels(linspace2(lat_0 - height / 2 / deg2m_lat,
                                        lat_0 + height / 2 / deg2m_lat, N1),
                              labels=[0, 1, 1, 0])
            if min(lons) < -150 and max(lons) > 150:
                lon_0 %= 360
            meridians = linspace2(lon_0 - width / 2 / deg2m_lon,
                                  lon_0 + width / 2 / deg2m_lon, N2)
            meridians[meridians > 180] -= 360
            map.drawmeridians(meridians, labels=[1, 0, 0, 1])
        else:
            msg = "Projection %s not supported." % projection
            raise ValueError(msg)

        # draw coast lines, country boundaries, fill continents.
        map.drawcoastlines()
        map.drawcountries()
        map.fillcontinents(color=continent_fill_color,
                           lake_color=water_fill_color)
        # draw the edge of the map projection region (the projection limb)
        map.drawmapboundary(fill_color=water_fill_color)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(-180, 180, 30))
        map.drawparallels(np.arange(-90, 90, 30))

        # compute the native map projection coordinates for events.
        x, y = map(lons, lats)
        # plot labels
        for name, xpt, ypt, colorpt in zip(labels, x, y, colors):
            plt.text(xpt, ypt, name, weight="heavy",
                     color=scal_map.to_rgba(colorpt))
        min_size = 5
        max_size = 18
        min_mag = min(mags)
        max_mag = max(mags)
        # plot filled circles at the locations of the events.
        frac = np.array([(m - min_mag) / (max_mag - min_mag) for m in mags])
        mags_plot = min_size + frac * (max_size - min_size)
        colors_plot = [scal_map.to_rgba(c) for c in colors]
        map.scatter(x, y, marker='o', s=(mags_plot ** 2), c=colors_plot,
                    zorder=10)
        times = [event.origins[0].time for event in self.events]
        plt.title(("%i events (%s to %s)" % (len(self),
             str(min(times).strftime("%Y-%m-%d")),
             str(max(times).strftime("%Y-%m-%d")))) +
                 " - Color codes %s, size the magnitude" % (
                     "origin time" if color == "date" else "depth"))

        cb = mpl.colorbar.ColorbarBase(ax=cm_ax, cmap=colormap,
                                       orientation='horizontal')
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        color_range = max_color - min_color
        cb.set_ticklabels([
            _i.strftime('%Y-%b-%d') if color == 'date' else '%.1fkm' % _i
            for _i in [min_color, min_color + color_range * 0.25,
                       min_color + color_range * 0.50,
                       min_color + color_range * 0.75, max_color]])

        # map.colorbar(scal_map, location="bottom", ax=cm_ax)
        plt.show()


def validate(xml_file):
    """
    Validates a QuakeML file against the QuakeML 1.2 RC4 XML Schema. Returns
    either True or False.
    """
    # Get the schema location.
    schema_location = os.path.dirname(inspect.getfile(inspect.currentframe()))
    schema_location = os.path.join(schema_location, "docs", "QuakeML-1.2.xsd")

    xmlschema = etree.XMLSchema(etree.parse(schema_location))
    xmldoc = etree.parse(xml_file)

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if valid is not True:
        print "Error validating QuakeML file:"
        for entry in xmlschema.error_log:
            print "\t%s" % entry
    return valid


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
