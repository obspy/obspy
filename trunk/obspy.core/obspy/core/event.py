# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, getExampleFile, \
        uncompressFile, _readFromPlugin
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.event_header import *
from pkg_resources import load_entry_point
import copy
import glob
import os
import re
import urllib2
from uuid import uuid4
import warnings
import weakref


EVENT_ENTRY_POINTS = ENTRY_POINTS['waveform']


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
    :param format: Format of the file to read. Depending on your ObsPy
        installation one of ``"QUAKEML"``. See the `Supported Formats`_ section
        below for a full list of supported formats.
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
    elif "://" in pathname_or_url:
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        # some URL
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


def _eventTypeClassFactory(type_name, class_attributes=[], class_contains=[]):
    """
    Class factory to unify the creation of all the types needed for the event
    handling in ObsPy.

    The types oftentimes share attributes and setting them manually every time
    is cumbersome, error-prone and hard to do consistently.

    Usage to create a new class type:

    >>> ABCEnum = Enum(["a", "b", "c"])
    >>> # For every fixed type attribute, corresponding getter/setter methods
    >>> # will be created and the attribute will be a property of the resulting
    >>> # class.
    >>> # The third item in the tuple is interpreted as "is allowed to be
    >>> # None". Thus, if False, it will initialise with given types default
    >>> # constructor. Use only for types is makes sense.
    >>> class_attributes = [ \
            ("resource_id", ResourceIdentifier, False), \
            ("creation_info", CreationInfo), \
            ("some_letters", ABCEnum), \
            ("description", str)]
    >>> # Furthermore the class can contain lists of other objects. These will
    >>> # just be list class attributes and nothing else so far.
    >>> class_contains = ["comments"]
    >>> TestEventClass = _eventTypeClassFactory( \
            "TestEventClass", \
            class_attributes=class_attributes, \
            class_contains=class_contains)
    >>> assert(TestEventClass.__name__ == "TestEventClass")

    Now the new class type can be used.

    >>> test_event = TestEventClass(resource_id="event/123456", \
                        creation_info={"author": "obspy.org", \
                                       "version": "0.1"})
    >>> # All given arguments will be converted to the right type.
    >>> print test_event.resource_id
    ResourceIdentifier(resource_id="event/123456")
    >>> print test_event.creation_info
    CreationInfo(author='obspy.org', version='0.1')
    >>> # All others will be set to None.
    >>> assert(test_event.description is None)
    >>> assert(test_event.some_letters is None)
    >>> # They can be set later and be converted to appropriate type if
    >>> # possible.
    >>> test_event.description = 1
    >>> assert(test_event.description is "1")
    >>> # Trying to set with an inappropriate value will raise an error.
    >>> test_event.some_letters = "d" # doctest:+ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: Setting attribute "some_letters" failed. ...

    >>> # If you pass ``"False"`` as the third tuple item for the
    >>> # class_attributes, the type will be initialized even if no value was
    >>> # given.
    >>> TestEventClass = _eventTypeClassFactory("TestEventClass",\
            class_attributes=[("time_1", UTCDateTime, False),\
                               ("time_2", UTCDateTime)])
    >>> test_event = TestEventClass()
    >>> print test_event.time_1.__repr__() # doctest:+ELLIPSIS
    UTCDateTime(...)
    >>> print test_event.time_2
    None
    """
    class AbstractEventType(object):
        def __init__(self, *args, **kwargs):
            # Store a list of all attributes to be able to get a nice string
            # representation of the object.
            self.__attributes = []
            self.__containers = []
            # Make sure the args work as expected. This means any given args
            # will overwrite the kwargs if they are given.
            for _i, item in enumerate(args):
                kwargs[class_attributes[_i][0]] = item
            for attrib in class_attributes:
                attrib_name = attrib[0]
                if len(attrib) == 3 and attrib[2] is False:
                    attrib_type = attrib[1]
                    setattr(self, attrib_name, kwargs.get(attrib_name,
                                                          attrib_type()))
                else:
                    setattr(self, attrib_name, kwargs.get(attrib_name, None))
                self.__attributes.append(attrib_name)
            for list_name in class_contains:
                setattr(self, list_name, list(kwargs.get(list_name, [])))
                self.__containers.append(list_name)

        def __str__(self):
            """
            Fairly extensive in an attempt to cover several use cases. It is
            always possible to change it in the child class.
            """
            attributes = [_i for _i in self.__attributes if getattr(self, _i)]
            containers = [_i for _i in self.__containers if getattr(self, _i)]
            # Get the longest attribute/container name to print all of them
            # nicely aligned.
            max_length = max(max([len(_i) for _i in attributes]) \
                                 if attributes else 0,
                             max([len(_i) for _i in containers]) \
                             if containers else 0) + 1

            ret_str = self.__class__.__name__
            attrib_count = len([_i for _i in self.__attributes \
                                   if getattr(self, _i)])
            container_count = len([_i for _i in self.__containers \
                                   if getattr(self, _i)])
            if not attrib_count and not container_count:
                return ret_str + "()"

            # First print a representation of all attributes that are not None.
            if attrib_count:
                # A small number of attributes and no containers will just
                # print a single line.
                if attrib_count <= 6 and not self.__containers:
                    att_strs = ["%s=%s" % (_i, getattr(self, _i).__repr__()) \
                                for _i in self.__attributes \
                                if getattr(self, _i)]
                    ret_str += "(%s)" % ", ".join(att_strs)
                else:
                    format_str = "%" + str(max_length) + "s: %s"
                    att_strs = [format_str % (_i,
                                              getattr(self, _i).__repr__()) \
                                for _i in self.__attributes \
                                if getattr(self, _i)]
                    ret_str += "\n\t" + "\n\t".join(att_strs)

            # For the containers just print the number of elements in each.
            if container_count:
                # Print delimiter only if there are attributes.
                if self.__attributes:
                    ret_str += '\n\t---------'
                element_str = "%" + str(max_length) + "s: %i Elements"
                ret_str += "\n\t" + \
                    "\n\t".join([element_str % \
                    (_i, len(getattr(self, _i))) \
                    for _i in self.__containers])
            return ret_str

        def __repr__(self):
            return self.__str__()

        def __nonzero__(self):
            if any([bool(getattr(self, _i)) \
                    for _i in self.__attributes + self.__containers]):
                return True
            return False

        def __eq__(self, other):
            """
            Two instances are considered equal if all attributes and all lists
            are identical.
            """
            # Looping should be quicker on average than a list comprehension
            # because only the first non-equal attribute will already return.
            for attrib in self.__attributes:
                if not hasattr(other, attrib) or \
                   (getattr(self, attrib) != getattr(other, attrib)):
                    return False
            for container in self.__containers:
                if not hasattr(other, container) or \
                   (getattr(self, container) != getattr(other, container)):
                    return False
            return True

        def __ne__(self, other):
            return not self.__eq__(other)


    # Use this awkward construct to get around a problem with closures. See
    # http://code.activestate.com/recipes/502271/
    def _create_getter_and_setter(attrib_name, attrib_type):
        # The getter function does not do much.
        def getter(instance):
            return instance.__dict__[attrib_name]

        def setter(instance, value):
            # If the value is None or already the correct type just set it.
            if (value is not None) and (type(value) is not attrib_type):
                # If it is a dict, and the attrib_type is no dict, than all
                # values will be assumed to be keyword arguments.
                if isinstance(value, dict):
                    value = attrib_type(**value)
                else:
                    value = attrib_type(value)
                if value is None:
                    msg = 'Setting attribute "%s" failed. ' % (attrib_name)
                    msg += '"%s" could not be converted to type "%s"' % \
                        (str(value), str(attrib_type))
                    raise ValueError(msg)
            instance.__dict__[attrib_name] = value
        return (getter, setter)

    # Now actually set the class properties.
    for attrib in class_attributes:
        attrib_name = attrib[0]
        attrib_type = attrib[1]
        getter, setter = _create_getter_and_setter(attrib_name, attrib_type)
        setattr(AbstractEventType, attrib_name, property(getter, setter))

    # Set the class type name.
    setattr(AbstractEventType, "__name__", type_name)
    return AbstractEventType


class ResourceIdentifier(object):
    """
    Unique identifier of any resource so it can be referred to.

    In QuakeML many elements and types can have a unique id that other elements
    use to refer to it. This is called a ResourceIdentifier and it is used for
    the same purpose in the obspy.core.event classes.

    In QuakeML it has to be of the following regex form:

        (smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\.\*\(\)_~']
        [\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*

    e.g.

    * smi:sub.website.org/event/12345678
    * quakeml:google.org/pick/unique_pick_id

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

    General usage:

    >>> res_id = ResourceIdentifier('2012-04-11--385392')
    >>> print res_id
    ResourceIdentifier(resource_id="2012-04-11--385392")
    >>> # If no resource_id is given it will be generated automatically.
    >>> print res_id # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="...")
    >>> # Supplying a prefix will simply prefix the automatically generated
    >>> # resource_id.
    >>> print ResourceIdentifier(prefix='event') # doctest:+ELLIPSIS
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
    >>> print res_id # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="smi:obspy.org/origin/...")
    >>> res_id = ResourceIdentifier('foo')
    >>> res_id.convertIDToQuakeMLURI()
    >>> print res_id
    ResourceIdentifier(resource_id="smi:local/foo")
    >>> # A good way to create a QuakeML compatibly ResourceIdentifier from
    >>> # scratch is
    >>> res_id = ResourceIdentifier(prefix='pick')
    >>> res_id.convertIDToQuakeMLURI(authority_id='obspy.org')
    >>> print res_id # doctest:+ELLIPSIS
    ResourceIdentifier(resource_id="smi:obspy.org/pick/...")
    >>> # If the given resource_id is already a valid QuakeML
    >>> # ResourceIdentifier, nothing will happen.
    >>> res_id = ResourceIdentifier('smi:test.org/subdir/id')
    >>> print res_id
    ResourceIdentifier(resource_id="smi:test.org/subdir/id")
    >>> res_id.convertIDToQuakeMLURI()
    >>> print res_id
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
        warnings.warn(msg)
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
        regex = r"(smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\." + \
                r"\*\(\)_~'][\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*"
        result = re.match(regex, str(self.resource_id))
        if result is not None:
            return self.resource_id
        resource_id = 'smi:%s/%s' % (authority_id, str(self.resource_id))
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
        if not hasattr(resource_id, '__hash__') or \
           not callable(resource_id.__hash__):
            msg = "resource_id needs to be a hashable type."
            raise TypeError(msg)
        self.__dict__["resource_id"] = resource_id

    resource_id = property(__getResourceID, __setResourceID, __delResourceID,
                           "unique identifier of the current instance")

    def __str__(self):
        return 'ResourceIdentifier(resource_id="%s")' % self.resource_id

    def __repr__(self):
        return self.__str__()

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
    class_attributes=[("agency_id", str), ("agency_uri", ResourceIdentifier),
                      ("author", str), ("author_uri", ResourceIdentifier),
                      ("creation_time", UTCDateTime), ("version", str)])


class CreationInfo(__CreationInfo):
    """
    CreationInfo is used to describe author, version, and creation time of a
    resource.

    :type agency_id: str, optional
    :param agency_id: Designation of agency that published a resource.
    :type agency_uri: str, optional
    :param agency_uri: Resource identifier of the agency that published a
        resource.
    :type author: str, optional
    :param author: Name describing the author of a resource.
    :type author_uri: str, optional
    :param author_uri: Resource identifier of the author of a resource.
    :type creation_time: UTCDateTime, optional
    :param creation_time: Time of creation of a resource.
    :type version: str, optional
    :param version: Version string of a resource.

    >>> info = CreationInfo(author="obspy.org", version="0.0.1")
    >>> print info
    CreationInfo(author='obspy.org', version='0.0.1')
    """


__TimeQuantity = _eventTypeClassFactory("__TimeQuantity",
    class_attributes=[("value", UTCDateTime), ("uncertainty", float),
        ("lower_uncertainty", float), ("upper_uncertainty", float),
        ("confidence_level", float)])


class TimeQuantity(__TimeQuantity):
    """
    A Physical quantity represented by its measured or computed value and
    optional values for symmetric or upper and lower uncertainties.

    :type value: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param value: Value of the quantity.
    :type uncertainty: float, optional
    :param uncertainty: Symmetric uncertainty or boundary.
    :type lower_uncertainty: float, optional
    :param lower_uncertainty: Relative lower uncertainty or boundary.
    :type upper_uncertainty: float, optional
    :param upper_uncertainty: Relative upper uncertainty or boundary.
    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.

    >>> time = TimeQuantity("2012-01-01", uncertainty=1.0)
    >>> print time
    TimeQuantity(value=UTCDateTime(2012, 1, 1, 0, 0), uncertainty=1.0)
    >>> time.value = 0.0
    >>> print time
    TimeQuantity(value=UTCDateTime(1970, 1, 1, 0, 0), uncertainty=1.0)
    """
    # Provided for backwards compatibility.
    _value_type = UTCDateTime


__IntegerQuantity = _eventTypeClassFactory("__IntegerQuantity",
    class_attributes=[("value", int), ("uncertainty", int),
        ("lower_uncertainty", int), ("upper_uncertainty", int),
        ("confidence_level", float)])


class IntegerQuantity(__IntegerQuantity):
    """
    A Physical quantity represented by its measured or computed value and
    optional values for symmetric or upper and lower uncertainties.

    :type value: int
    :param value: Value of the quantity.
    :type uncertainty: int, optional
    :param uncertainty: Symmetric uncertainty or boundary.
    :type lower_uncertainty: int, optional
    :param lower_uncertainty: Relative lower uncertainty or boundary.
    :type upper_uncertainty: int, optional
    :param upper_uncertainty: Relative upper uncertainty or boundary.
    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.
    """
    # Provided for backwards compatibility.
    _value_type = int


__FloatQuantity = _eventTypeClassFactory("__FloatQuantity",
    class_attributes=[("value", float), ("uncertainty", float),
        ("lower_uncertainty", float), ("upper_uncertainty", float),
        ("confidence_level", float)])


class FloatQuantity(__FloatQuantity):
    """
    A Physical quantity represented by its measured or computed value and
    optional values for symmetric or upper and lower uncertainties.

    :type value: float
    :param value: Value of the quantity.
    :type uncertainty: float, optional
    :param uncertainty: Symmetric uncertainty or boundary.
    :type lower_uncertainty: float, optional
    :param lower_uncertainty: Relative lower uncertainty or boundary.
    :type upper_uncertainty: float, optional
    :param upper_uncertainty: Relative upper uncertainty or boundary.
    :type confidence_level: float, optional
    :param confidence_level: Confidence level of the uncertainty, given in
        percent.
    """
    # Provided for backwards compatibility.
    _value_type = float


__CompositeTime = _eventTypeClassFactory("__CompositeTime",
    class_attributes=[("year", IntegerQuantity, False),
                      ("month", IntegerQuantity, False),
                      ("day", IntegerQuantity, False),
                      ("hour", IntegerQuantity, False),
                      ("minute", IntegerQuantity, False),
                      ("second", FloatQuantity, False)])


class CompositeTime(__CompositeTime):
    """
    Focal times differ significantly in their precision. While focal times of
    instrumentally located earthquakes are estimated precisely down to seconds,
    historic events have only incomplete time descriptions. Sometimes, even
    contradictory information about the rupture time exist. The CompositeTime
    type allows for such complex descriptions.

    :type year: :class:`~obspy.core.event.IntegerQuantity`
    :param year: Year or range of years of the event’s focal time.
    :type month: :class:`~obspy.core.event.IntegerQuantity`
    :param month: Month or range of months of the event’s focal time.
    :type day: :class:`~obspy.core.event.IntegerQuantity`
    :param day: Day or range of days of the event’s focal time.
    :type hour: :class:`~obspy.core.event.IntegerQuantity`
    :param hour: Hour or range of hours of the event’s focal time.
    :type minute: :class:`~obspy.core.event.IntegerQuantity`
    :param minute: Minute or range of minutes of the event’s focal time.
    :type second: :class:`~obspy.core.event.FloatQuantity`
    :param second: Second and fraction of seconds or range of seconds with

    >>> time = CompositeTime(2011, 1, 1)
    >>> print time # doctest:+ELLIPSIS
    CompositeTime(year=IntegerQuantity(value=2011), month=IntegerQuantity(...
    >>> # Can also be instantiated with the uncertainties.
    >>> time = CompositeTime(year={"value":2011, "uncertainty":1})
    >>> print time
    CompositeTime(year=IntegerQuantity(value=2011, uncertainty=1))
    """


__Comment = _eventTypeClassFactory("__Comment",
    class_attributes=[("text", str), ("resource_id", ResourceIdentifier),
                      ("creation_info", CreationInfo)])


class Comment(__Comment):
    """
    Comment holds information on comments to a resource as well as author and
    creation time information.

    :type text: str, optional
    :param text: Text of comment.
    :type resource_id: :class:`~obspy.core.event.ResourceIdentifier`, optional
    :param resource_id: Identifier of comment.
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
                      ("resource_id", ResourceIdentifier)])


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
    :type resource_uri: str, optional
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
    >>> stream_id = WaveformStreamID(network_code="BW", station_code="FUR", \
                                     location_code="", channel_code="EHZ")
    >>> print stream_id
    WaveformStreamID(network_code='BW', station_code='FUR', channel_code='EHZ')
    >>> stream_id = WaveformStreamID(seed_string="BW.FUR..EHZ")
    >>> print stream_id
    WaveformStreamID(network_code='BW', station_code='FUR', channel_code='EHZ')
    >>> # Can also return the SEED string.
    print stream_id.getSEEDString()
    BW.FUR..EHZ
    """
    def __init__(self, network_code=None, station_code=None,
                 location_code=None, channel_code=None, resource_id=None,
                 seed_string=None):
        # Use the seed_string if it is given and everything else is not.
        if (seed_string is not None) and (network_code is None) and \
           (station_code is None) and (location_code is None) and \
           (channel_code is None):
            try:
                network_code, station_code, location_code, channel_code = \
                    seed_string.split('.')
            except ValueError:
                warnings.warn("In WaveformStreamID.__init__(): " + \
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
                                               resource_id=resource_id)

    def getSEEDString(self):
        return "%s.%s.%s.%s" % (\
            self.network_code if self.network_code else "",
            self.station_code if self.station_code else "",
            self.location_code if self.location_code else "",
            self.channel_code if self.channel_code else "")


__Pick = _eventTypeClassFactory("__Pick",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", TimeQuantity, False),
                      ("waveform_id", WaveformStreamID, False),
                      ("filter_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("horizontal_slowness", FloatQuantity, False),
                      ("backazimuth", FloatQuantity, False),
                      ("slowness_method_id", ResourceIdentifier),
                      ("pick_onset", PickOnset),
                      ("phase_hint", str),
                      ("pick_polarity", PickPolarity),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class Pick(__Pick):
    """
    This class contains various attributes commonly used to describe a single
    pick, e.g. time, waveform id, onset, phase hint, polarity, etc

    :type resource_id: str
    :param resource_id: Resource identifier of Pick.
    :type time: :class:`~obspy.core.event.TimeQuantity`
    :param time: Pick time.
    :type waveform_id: :class:`~obspy.core.event.WaveformStreamID`
    :param waveform_id: Identifies the waveform stream.
    :type filter_id: str, optional
    :param filter_id: Identifies the filter setup used.
    :type method_id: str, optional
    :param method_id: Identifies the method used to get the pick.
    :type horizontal_slowness: :class:`~obspy.core.event.FloatQuantity`,
        optional
    :param horizontal_slowness: Describes the horizontal slowness of the Pick.
    :type backazimuth: :class:`~obspy.core.event.FloatQuantity`, optional
    :param backazimuth: Describes the backazimuth of the Pick.
    :type slowness_method_id: str, optional
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
    :param polarity: Describes the pick onset type. Allowed values are
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
    class_attributes=[("pick_id", ResourceIdentifier),
                      ("phase", str),
                      ("time_correction", float),
                      ("azimuth", float),
                      ("distance", float),
                      ("time_residual", float),
                      ("horizontal_slowness_residual", float),
                      ("backazimuthal_residual", float),
                      ("time_used", bool),
                      ("horizontal_slowness_used", bool),
                      ("backazimuth_used", bool),
                      ("time_weight", float),
                      ("earth_model_id", ResourceIdentifier),
                      ("preliminary", bool),
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

    :type pick_id: str
    :param pick_id: Refers to a resource_id of a Pick.
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
    :type time_residual: float, optional
    :param time_residual: Residual between observed and expected arrival time
        assuming proper phase identification and given the earth_model_id of
        the Origin in seconds.
    :type horizontal_slowness_residual: float, optional
    :param horizontal_slowness_residual: Residual of horizontal slowness in
        seconds per degree.
    :type backazimuthal_residual: float, optional
    :param backazimuthal_residual: Residual of backazimuth in degree.
    :type time_used: bool, optional
    :param time_used: Boolean flag. True if arrival time was used for
        computation of the associated Origin.
    :type horizontal_slowness_used: bool, optional
    :param horizontal_slowness_used: Boolean flag. True if horizontal slowness
        was used for computation of the associated Origin.
    :type backazimuth_used: bool, optional
    :param backazimuth_used: Boolean flag. True if backazimuth was used for
        computation of the associated Origin.
    :type time_weight: :class:`~obspy.core.event.FloatQuantity`, optional
    :param time_weight: Weight of this Arrival in the computation of the
        associated Origin. (timeWeight in XSD file, weight in PDF).
    :type earth_model_id: str, optional
    :param earth_model_id: Earth model which is used for the association of
        Arrival to Pick and computation of the residuals.
    :type preliminary: bool, optional
    :param preliminary: Boolean flag. True if arrival designation is
        preliminary.
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
        values are the following::
            * horizontal uncertainty
            * uncertainty ellipse
            * confidence ellipsoid
            * probability density function
    """

__Origin = _eventTypeClassFactory("__Origin",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("time", TimeQuantity, False),
                      ("latitude", FloatQuantity, False),
                      ("longitude", FloatQuantity, False),
                      ("depth", FloatQuantity, False),
                      ("depth_type", OriginDepthType),
                      ("time_fixed", bool),
                      ("epicenter_fixed", bool),
                      ("reference_system_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("earth_model_id", ResourceIdentifier),
                      ("quality", OriginQuality),
                      ("origin_type", OriginType),
                      ("origin_uncertainty", OriginUncertainty),
                      ("evaluation_mode", EvaluationMode),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "arrivals", "composite_times"])


class Origin(__Origin):
    """
    This class represents the focal time and geographical location of an
    earthquake hypocenter, as well as additional meta-information.

    :type resource_id: str
    :param resource_id: Resource identifier of Origin.
    :type time: :class:`~obspy.core.event.TimeQuantity`
    :param time: Focal time.
    :type latitude: :class:`~obspy.core.event.FloatQuantity`
    :param latitude: Hypocenter latitude. Unit: deg
    :type longitude: :class:`~obspy.core.event.FloatQuantity`
    :param longitude: Hypocenter longitude. Unit: deg
    :type depth: :class:`~obspy.core.event.FloatQuantity`, optional
    :param depth: Depth of hypocenter. Unit: m
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
    :type reference_system_id: str, optional
    :param reference_system_id: Identifies the reference system used for
        hypocenter determination.
    :type method_id: str, optional
    :param method_id: Identifies the method used for locating the event.
    :type earth_model_id: str, optional
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
    >>> origin.latitude = {"value": 12, "confidence_level": 95}
    >>> origin.longitude = 42
    >>> origin.depth_type = 'from location'
    >>> print(origin)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Origin
        resource_id: ResourceIdentifier(resource_id="smi:ch.ethz.sed/...")
               time: TimeQuantity(value=UTCDateTime(1970, 1, 1, 0, 0))
           latitude: FloatQuantity(value=12.0, confidence_level=95.0)
          longitude: FloatQuantity(value=42.0)
         depth_type: 'from location'
    """


__StationMagnitudeContribution = _eventTypeClassFactory(\
    "__StationMagnitudeContribution",
    class_attributes=[("station_magnitude_id", ResourceIdentifier),
                      ("residual", float),
                      ("weight", float)])


class StationMagnitudeContribution(__StationMagnitudeContribution):
    """
    This class describes the weighting of magnitude values from Magnitude
    estimations.

    :type station_magnitude_id: ResourceIdentifier, optional
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
                      ("mag", FloatQuantity, False),
                      ("magnitude_type", str),
                      ("origin_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("station_count", int),
                      ("azimuthal_gap", float),
                      ("evaluation_status", EvaluationStatus),
                      ("creation_info", CreationInfo)],
    class_contains=["comments", "station_magnitude_contribution"])


class Magnitude(__Magnitude):
    """
    Describes a magnitude which can, but need not be associated with an Origin.

    Association with an origin is expressed with the optional attribute
    ``origin_id``. It is either a combination of different magnitude
    estimations, or it represents the reported magnitude for the given Event.

    :type resource_id: str
    :param resource_id: Resource identifier of Magnitude.
    :type mag: :class:`~obspy.core.event.FloatQuantity`
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.StationMagnitude`. If no estimations are
        available, this value can represent the reported magnitude.
    :type magnitude_type: str, optional
    :param magnitude_type: Describes the type of magnitude. This is a free-text
        field because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are
            * unspecified magnitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type origin_id: ResourceIdentifier, optional
    :param origin_id: Reference to an origin’s resource_id if the magnitude has
        an associated Origin.
    :type method_id: ResourceIdentifier, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type station_count, int, optional
    :param station_count Number of used stations for this magnitude
        computation.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Azimuthal gap for this magnitude computation.
        Unit: deg
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
                      ("mag", FloatQuantity, False),
                      ("station_magnitude_type", str),
                      ("amplitude_id", ResourceIdentifier),
                      ("method_id", ResourceIdentifier),
                      ("waveform_id", WaveformStreamID, False),
                      ("creation_info", CreationInfo)],
    class_contains=["comments"])


class StationMagnitude(__StationMagnitude):
    """
    This class describes the magnitude derived from a single waveform stream.

    :type resource_id: ResourceIdentifier, optional
,   :param resource_id: Resource identifier of StationMagnitude.
    :type origin_id: ResourceIdentifier, optional
    :param origin_id: Reference to an origin’s ``resource_id`` if the
        StationMagnitude has an associated :class:`~obspy.core.event.Origin`.
    :type mag: :class:`~obspy.core.event.FloatQuantity`
    :param mag: Estimated magnitude.
    :type station_magnitude_type: str, optional
    :param station_magnitude_type: Describes the type of magnitude. This is a
        free-text field because it is impossible to cover all existing
        magnitude type designations with an enumeration. Possible values are
            * unspecified magnitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type amplitude_id: ResourceIdentifier, optional
    :param amplitude_id: Identifies the data source of the StationMagnitude.
        For magnitudes derived from amplitudes in waveforms (e. g.,
        local magnitude ML), amplitude_id points to resource_id in class
        :class:`obspy.core.event.Amplitude`.
    :type method_id: ResourceIdentifier, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type waveform_id: WaveformStreamID, optional
    :param waveform_id: Identifies the waveform stream.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """

__EventDescription = _eventTypeClassFactory("__EventDescription",
    class_attributes=[("text", str), ("event_description_type",
                                      EventDescriptionType)])


class EventDescription(__EventDescription):
    """
    Free-form string with additional event description. This can be a
    well-known name, like 1906 San Francisco Earthquake. A number of categories
    can be given in type.

    :type text: str, optional
    :param text: Free-form text with earthquake description.
    :type event_description_type: str, optional
    :param event_description_type: Category of earthquake description. Values
        can be taken from
        the following:
            * ``"felt report"``
            * ``"Flinn-Engdahl region"``
            * ``"local time"``
            * ``"tectonic summary"``
            * ``"nearest cities"``
            * ``"earthquake name"``
            * ``"region name"``

    .. rubric:: Example
    """

__Event = _eventTypeClassFactory("__Event",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("event_type", EventType),
                      ("event_type_certainty", EventTypeCertainty),
                      ("creation_info", CreationInfo)],
    class_contains=['event-descriptions', 'comments', 'picks', 'amplitudes',
                    'station_magnitudes', 'focal_mechanisms', 'origins',
                    'magnitudes'])


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

    :type resource_id: ResourceIdentifier, optional
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
    :type station_magnitudes: list of
        :class:`~obspy.core.event.StationMagnitude`
    :param station_magnitudes: Station magnitudes associated with the event
    :type focal_mechanisms: list of :class:`~obspy.core.event.FocalMechanism`
    :param focal_mechanisms: Focal mechanisms associated with the event
    :type origins: list of :class:`~obspy.core.event.Origin`
    :param origins: Origins associated with the event.
    :type magnitudes: list of :class:`~obspy.core.event.Magnitude`
    :param magnitudes: Magnitudes associated with the event.
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
        if self.origins:
            out += '%s | %+7.3f, %+8.3f' % (self.origins[0].time.value,
                                            self.origins[0].latitude.value,
                                            self.origins[0].longitude.value)
        if self.magnitudes:
            out += ' | %s %-2s' % (self.magnitudes[0].mag.value,
                                   self.magnitudes[0].type)
        if self.origins and self.origins[0].evaluation_mode:
            out += ' | %s' % (self.origins[0].evaluation_mode)
        return out

__Catalog = _eventTypeClassFactory("__Catalog",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("description", str),
                      ("creation_info", CreationInfo)],
    class_contains=["events", "comments"])


class Catalog(__Catalog):
    """
    This class serves as a container for Event objects.

    :type events: list of :class:`~obspy.core.event.Event`, optional
    :param events: List of events
    :type resource_id: str, optional
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
            other = Catalog([other])
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
        self.events.__setitem__(index, event)

    def __str__(self):
        """
        Returns short summary string of the current catalog.

        It will contain the number of Events in the Catalog and the return
        value of each Event's :meth:`~obspy.core.event.Event.__str__` method.
        """
        out = str(len(self.events)) + ' Event(s) in Catalog:\n'
        out = out + "\n".join([ev.short_str() for ev in self])
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
        :param format: The format to write must be specified. Depending on your
            ObsPy installation one of ``"QUAKEML"``. See the
            `Supported Formats`_ section below for a full list of supported
            formats.
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

        Please refer to the *Linked Function Call* of each module for any extra
        options available.

        =======  ===================  =====================================
        Format   Required Module      Linked Function Call
        =======  ===================  =====================================
        QUAKEML  :mod:`obspy.core`   :func:`obspy.core.event.writeQUAKEML`
        =======  ===================  =====================================
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

    def plot(self, resolution='l', **kwargs):  # @UnusedVariable
        """
        Creates preview map of all events in current Catalog object.
        """
        from mpl_toolkits.basemap import Basemap
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        map = Basemap(resolution=resolution)
        # draw coast lines, country boundaries, fill continents.
        map.drawcoastlines()
        map.drawcountries()
        map.fillcontinents(color='0.8')
        # draw the edge of the map projection region (the projection limb)
        map.drawmapboundary()
        # lat/lon coordinates
        lats = []
        lons = []
        labels = []
        for i, event in enumerate(self.events):
            lats.append(event.preferred_origin.latitude)
            lons.append(event.preferred_origin.longitude)
            labels.append(' #%d' % i)
        # compute the native map projection coordinates for events.
        x, y = map(lons, lats)
        # plot filled circles at the locations of the events.
        map.plot(x, y, 'ro')
        # plot labels
        for name, xpt, ypt in zip(labels, x, y):
            plt.text(xpt, ypt, name, size='small')
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
