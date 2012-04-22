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
from obspy.core.util import NamedTemporaryFile, getExampleFile, Enum, \
    uncompressFile, AttribDict, _readFromPlugin
from obspy.core.util.base import ENTRY_POINTS
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
    :param format: Format of the file to read, currently only ``"QUAKEML"`` is
        supported.
    :return: A ObsPy :class:`~obspy.core.event.Catalog` object.
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


OriginUncertaintyDescription = Enum([
    "horizontal uncertainty",
    "uncertainty ellipse",
    "confidence ellipsoid",
    "probability density function",
])
AmplitudeCategory = Enum([
    "point",
    "mean",
    "duration",
    "period",
    "integral",
    "other",
])
OriginDepthType = Enum([
    "from location",
    "from moment tensor inversion",
    "from modeling of broad-band P waveforms",
    "constrained by depth phases",
    "constrained by direct phases",
    "operator assigned",
    "other",
])
OriginType = Enum([
    "hypocenter",
    "centroid",
    "amplitude",
    "macroseismic",
    "rupture start",
    "rupture end",
])
MTInversionType = Enum([
    "general",
    "zero trace",
    "double couple",
])
EvaluationMode = Enum([
    "manual",
    "automatic",
])
EvaluationStatus = Enum([
    "preliminary",
    "confirmed",
    "reviewed",
    "final",
    "rejected",
])
PickOnset = Enum([
    "emergent",
    "impulsive",
    "questionable",
])
DataUsedWaveType = Enum([
    "P waves",
    "body waves",
    "surface waves",
    "mantle waves",
    "combined",
    "unknown",
])
AmplitudeUnit = Enum([
    "m",
    "s",
    "m/s",
    "m/(s*s)",
    "m*s",
    "dimensionless",
    "other",
])
EventDescriptionType = Enum([
    "felt report",
    "Flinn-Engdahl region",
    "local time",
    "tectonic summary",
    "nearest cities",
    "earthquake name",
    "region name",
])
MomentTensorCategory = Enum([
    "teleseismic",
    "regional",
])
EventType = Enum([
    "earthquake",
    "induced earthquake",
    "quarry blast",
    "explosion",
    "chemical explosion",
    "nuclear explosion",
    "landslide",
    "rockslide",
    "snow avalanche",
    "debris avalanche",
    "mine collapse",
    "building collapse",
    "volcanic eruption",
    "meteor impact",
    "plane crash",
    "sonic boom",
    "not existing",
    "other",
    "null",
])
EventTypeCertainty = Enum([
    "known",
    "suspected",
])
SourceTimeFunctionType = Enum([
    "box car",
    "triangle",
    "trapezoid",
    "unknown",
])
PickPolarity = Enum([
    "positive",
    "negative",
    "undecidable",
])


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
    identifier class in the first place) is that once a ResourceIdentifer with
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
    ValueError: resource_id needs to be a hashable type.
    >>> res_id = ResourceIdentifier()
    >>> res_id.resource_id = [1,2]
    Traceback (most recent call last):
        ...
    ValueError: resource_id needs to be a hashable type.

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
    >>> print items # doctest:+ELLIPSIS
    [(<...ResourceIdentifier object at ...>, 'bar'), ('foo', 'bar')]
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
        # Straight copy from the QuakeML xsd file. Compiling the regex is not
        # worthwhile because recent expressions are cached within the re
        # module.
        regex = r"(smi|quakeml):[\w\d][\w\d\-\.\*\(\)_~']{2,}/[\w\d\-\." + \
                r"\*\(\)_~'][\w\d\-\.\*\(\)\+\?_~'=,;#/&amp;]*"
        result = re.match(regex, str(self.resource_id))
        if result is not None:
            return
        self.__setResourceID('smi:%s/%s' % (authority_id,
                                            str(self.resource_id)))
        # Check once again just to be sure no weird symbols are stored in the
        # resource_id.
        result = re.match(regex, self.resource_id)
        if result is None:
            msg = "Failed to create a valid QuakeML ResourceIdentifier."
            raise Exception(msg)

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
            raise ValueError(msg)
        self.__dict__["resource_id"] = resource_id

    resource_id = property(__getResourceID, __setResourceID, __delResourceID,
                           "unique identifier of the current instance")

    def __str__(self):
        return 'ResourceIdentifier(resource_id="%s")' % self.resource_id

    def __eq__(self, other):
        # The type check is necessary due to the used hashing method.
        if type(self) != type(other):
            return False
        if self.resource_id == other.resource_id:
            return True
        return False

    def __hash__(self):
        """
        Uses the same hash as the resource id. This means that class instances
        can be used in dictionaries and other hashed types.

        Both the object and it's id can still be independently used as
        dictionary keys.
        """
        return self.resource_id.__hash__()


class CreationInfo(AttribDict):
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
    """
    agency_id = None
    agency_uri = None
    author = None
    author_uri = None
    creation_time = None
    version = None


class _ValueQuantity(AttribDict):
    """
    Physical quantities that can be expressed numerically — either as integers,
    floating point numbers or UTCDateTime objects — are represented by their
    measured or computed values and optional values for symmetric or upper and
    lower uncertainties.

    :type value: int, float or :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param value: Value of the quantity. The unit is implicitly defined and
        depends on the context.
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
    _value_type = str
    value = None
    uncertainty = None
    lower_uncertainty = None
    upper_uncertainty = None
    confidence_level = None


class TimeQuantity(_ValueQuantity):
    _value_type = UTCDateTime


class FloatQuantity(_ValueQuantity):
    _value_type = float


class IntegerQuantity(_ValueQuantity):
    _value_type = int


class CompositeTime(AttribDict):
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
        fraction of the event’s focal time.
    """
    def __init__(self, year={}, month={}, day={}, hour={}, minute={},
                 second={}):
        self.year = IntegerQuantity(year)
        self.month = IntegerQuantity(month)
        self.day = IntegerQuantity(day)
        self.hour = IntegerQuantity(hour)
        self.minute = IntegerQuantity(minute)
        self.second = FloatQuantity(second)


class Comment(AttribDict):
    """
    Comment holds information on comments to a resource as well as author and
    creation time information.

    :type text: str
    :param text: Text of comment.
    :type id: str or None, optional
    :param id: Identifier of comment, in QuakeML resource identifier format.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`
    :param creation_info: Creation info of comment (author, version, creation
        time).
    """
    def __init__(self, text='', id=None, creation_info={}):
        self.text = text
        self.id = id
        self.creation_info = CreationInfo(creation_info)


class WaveformStreamID(AttribDict):
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
    """
    def __init__(self, network=None, station=None, location=None, channel=None,
                 resource_uri=None, seed_string=None):
        # Use the seed_string if it is given and everything else is not.
        if (seed_string is not None) and (network is None) and \
           (station is None) and (location is None) and (channel is None):
            try:
                network, station, location, channel = seed_string.split('.')
            except ValueError:
                warnings.warn("In WaveformStreamID.__init__(): " + \
                              "seed_string was given but could not be parsed")
                pass
        self.network = network or ''
        self.station = station or ''
        self.location = location
        self.channel = channel
        self.resource_uri = resource_uri


class Pick(AttribDict):
    """
    This class contains various attributes commonly used to describe a single
    pick, e.g. time, waveform id, onset, phase hint, polarity, etc

    :type public_id: str
    :param public_id: Resource identifier of Pick.
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
        this is a seperate type but it just contains a single field containing
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
    :type arrivals: list of :class:`~obspy.core.event.Arrival` objects
    :param arrivals: Child elements of the Pick object.
    """
    def __init__(self, public_id='', time={}, waveform_id={}, filter_id=None,
                 method_id=None, horizontal_slowness={}, backazimuth={},
                 slowness_method_id=None, onset=None, phase_hint=None,
                 polarity=None, evaluation_mode=None, evaluation_status=None,
                 comments=None, creation_info={}, arrivals=[]):
        self.public_id = public_id
        self.time = TimeQuantity(time)
        self.waveform_id = WaveformStreamID(waveform_id)
        self.filter_id = filter_id
        self.method_id = method_id
        self.horizontal_slowness = FloatQuantity(horizontal_slowness)
        self.backazimuth = FloatQuantity(backazimuth)
        self.slowness_method_id = slowness_method_id
        self.onset = PickOnset(onset)
        self.phase_hint = phase_hint
        self.polarity = PickPolarity(polarity)
        self.evaluation_mode = EvaluationMode(evaluation_mode)
        self.evaluation_status = EvaluationStatus(evaluation_status)
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)
        self.arrivals = arrivals

    def _getPickOnset(self):
        return self.__dict__.get('onset', None)

    def _setPickOnset(self, value):
        self.__dict__['onset'] = PickOnset(value)

    onset = property(_getPickOnset, _setPickOnset)

    def _getPickPolarity(self):
        return self.__dict__.get('polarity', None)

    def _setPickPolarity(self, value):
        self.__dict__['polarity'] = PickPolarity(value)

    polarity = property(_getPickPolarity, _setPickPolarity)

    def _getEvaluationMode(self):
        return self.__dict__.get('evaluation_mode', None)

    def _setEvaluationMode(self, value):
        self.__dict__['evaluation_mode'] = EvaluationMode(value)

    evaluation_mode = property(_getEvaluationMode, _setEvaluationMode)

    def _getEvaluationStatus(self):
        return self.__dict__.get('evaluation_status', None)

    def _setEvaluationStatus(self, value):
        self.__dict__['evaluation_status'] = EvaluationStatus(value)

    evaluation_status = property(_getEvaluationStatus, _setEvaluationStatus)


class Arrival(AttribDict):
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
    :param pick_id: Refers to a public_id of a Pick.
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
    :type time_weight: float, optional
    :param time_weight: Weight of this Arrival in the computation of the
        associated Origin.
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
    def __init__(self, pick_id='', phase='', time_correction=None,
                 azimuth=None, distance=None, time_residual=None,
                 horizontal_slowness_residual=None,
                 backazimuthal_residual=None, time_used=None,
                 horizontal_slowness_used=None, backazimuth_used=None,
                 time_weight=None, earth_model_id=None, preliminary=None,
                 comments=None, creation_info={}):
        self.pick_id = pick_id
        self.phase = phase
        self.time_correction = time_correction
        self.azimuth = azimuth
        self.distance = distance
        self.time_residual = time_residual
        self.horizontal_slowness_residual = horizontal_slowness_residual
        self.backazimuthal_residual = backazimuthal_residual
        self.time_used = time_used
        self.horizontal_slowness_used = horizontal_slowness_used
        self.backazimuth_used = backazimuth_used
        self.time_weight = time_weight  # timeWeight in XSD file, weight in PDF
        self.earth_model_id = earth_model_id
        if preliminary is not None:
            self.preliminary = bool(preliminary)
        else:
            self.preliminary = None
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)


class OriginQuality(AttribDict):
    """
    This class contains various attributes commonly used to describe the
    quality of an origin, e. g., errors, azimuthal coverage, etc.

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
    associated_phase_count = None
    used_phase_count = None
    associated_station_count = None
    used_station_count = None
    depth_phase_count = None
    standard_error = None
    azimuthal_gap = None
    secondary_azimuthal_gap = None
    ground_truth_level = None
    minimum_distance = None
    maximum_distance = None
    median_distance = None


class ConfidenceEllipsoid(AttribDict):
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
    semi_major_axis_length = None
    semi_minor_axis_length = None
    semi_intermediate_axis_length = None
    major_axis_plunge = None
    major_axis_azimuth = None
    major_axis_rotation = None


class OriginUncertainty(AttribDict):
    """
    This class describes the location uncertainties of an origin.

    The uncertainty can be described either as a simple circular horizontal
    uncertainty, an uncertainty ellipse according to IMS1.0, or a confidence
    ellipsoid. The preferred variant can be given in the attribute
    ``preferred_description``.

    :type preferred_description: str, optional
    :param preferred_description: Preferred uncertainty description. Allowed
        values are the following::
            * horizontal uncertainty
            * uncertainty ellipse
            * confidence ellipsoid
            * probability density function
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
    """
    horizontal_uncertainty = None
    min_horizontal_uncertainty = None
    max_horizontal_uncertainty = None
    azimuth_max_horizontal_uncertainty = None
    confidence_ellipsoid = ConfidenceEllipsoid()

    def _getOriginUncertaintyDescription(self):
        return self.__dict__.get('preferred_description', None)

    def _setOriginUncertaintyDescription(self, value):
        self.__dict__['preferred_description'] = \
            OriginUncertaintyDescription(value)

    preferred_description = property(_getOriginUncertaintyDescription,
                                     _setOriginUncertaintyDescription)


class Origin(AttribDict):
    """
    This class represents the focal time and geographical location of an
    earthquake hypocenter, as well as additional meta-information.

    :type public_id: str
    :param public_id: Resource identifier of Origin.
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
    >>> origin.public_id = 'smi:ch.ethz.sed/origin/37465'
    >>> origin.time.value = UTCDateTime(0)
    >>> origin.latitude.value = 12
    >>> origin.latitude.confidence_level = 95
    >>> origin.longitude.value = 42
    >>> origin.depth_type = 'from location'
    >>> print(origin)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
              public_id: smi:ch.ethz.sed/origin/37465
                   time: TimeQuantity({'value': UTCDateTime(1970, 1, 1, 0, 0)})
               latitude: FloatQuantity({'value': 12, 'confidence_level': 95})
              longitude: FloatQuantity({'value': 42})
               arrivals: []
               comments: []
        ...
    """
    def __init__(self, public_id='', time={}, latitude={},
                 longitude={}, depth={}, depth_type=None, time_fixed=None,
                 epicenter_fixed=None, reference_system_id=None,
                 method_id=None, earth_model_id=None, composite_times=None,
                 quality={}, type=None, evaluation_mode=None,
                 evaluation_status=None, origin_uncertainty={},
                 comments=None, creation_info={}, arrivals=None):
        # default attributes
        self.public_id = public_id
        self.time = TimeQuantity(time)
        self.latitude = FloatQuantity(latitude)
        self.longitude = FloatQuantity(longitude)
        self.depth = FloatQuantity(depth)
        self.depth_type = depth_type
        self.time_fixed = time_fixed
        self.epicenter_fixed = epicenter_fixed
        self.reference_system_id = reference_system_id
        self.method_id = method_id
        self.earth_model_id = earth_model_id
        self.composite_times = composite_times or []
        self.quality = OriginQuality(quality)
        self.type = type
        self.evaluation_mode = evaluation_mode
        self.evaluation_status = evaluation_status
        self.origin_uncertainty = OriginUncertainty(origin_uncertainty)
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)
        # child elements
        self.arrivals = arrivals or []

    def __str__(self):
        return self._pretty_str(['public_id', 'time', 'latitude', 'longitude'])

    def _getOriginDepthType(self):
        return self.__dict__.get('depth_type', None)

    def _setOriginDepthType(self, value):
        self.__dict__['depth_type'] = OriginDepthType(value)

    depth_type = property(_getOriginDepthType, _setOriginDepthType)

    def _getOriginType(self):
        return self.__dict__.get('type', None)

    def _setOriginType(self, value):
        self.__dict__['type'] = OriginType(value)

    type = property(_getOriginType, _setOriginType)

    def _getEvaluationMode(self):
        return self.__dict__.get('evaluation_mode', None)

    def _setEvaluationMode(self, value):
        self.__dict__['evaluation_mode'] = EvaluationMode(value)

    evaluation_mode = property(_getEvaluationMode, _setEvaluationMode)

    def _getEvaluationStatus(self):
        return self.__dict__.get('evaluation_status', None)

    def _setEvaluationStatus(self, value):
        self.__dict__['evaluation_status'] = EvaluationStatus(value)

    evaluation_status = property(_getEvaluationStatus, _setEvaluationStatus)


class Magnitude(AttribDict):
    """
    Describes a magnitude which can, but need not be associated with an Origin.

    Association with an origin is expressed with the optional attribute
    ``origin_id``. It is either a combination of different magnitude
    estimations, or it represents the reported magnitude for the given Event.

    :type public_id: str
    :param public_id: Resource identifier of Magnitude.
    :type mag: :class:`~obspy.core.event.FloatQuantity`
    :param mag: Resulting magnitude value from combining values of type
        :class:`~obspy.core.event.StationMagnitude`. If no estimations are
        available, this value can represent the reported magnitude.
    :type type: str, optional
    :param type: Describes the type of magnitude. This is a free-text field
        because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are
            * unspecified magitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type origin_id: str, optional
    :param origin_id: Reference to an origin’s public_id if the magnitude has
        an associated Origin.
    :type method_id: str, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type station_count, int, optional
    :param station_count Number of used stations for this magnitude
        computation.
    :type azimuthal_gap: float, optional
    :param azimuthal_gap: Azimuthal gap for this magnitude computation.
        Unit: deg
    :type evaluation_status: str, optional
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
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    def __init__(self, public_id='', mag={}, type=None, origin_id=None,
                 method_id=None, station_count=None, azimuthal_gap=None,
                 evaluation_status=None, comments=None, creation_info={}):
        # default attributes
        self.public_id = public_id
        self.mag = FloatQuantity(mag)
        self.type = type
        self.origin_id = origin_id
        self.method_id = method_id
        self.station_count = station_count
        self.azimuthal_gap = azimuthal_gap
        self.evaluation_status = evaluation_status
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)

    def __str__(self):
        return self._pretty_str(['magnitude'])

    def _getEvaluationStatus(self):
        return self.__dict__.get('evaluation_status', None)

    def _setEvaluationStatus(self, value):
        self.__dict__['evaluation_status'] = EvaluationStatus(value)

    evaluation_status = property(_getEvaluationStatus, _setEvaluationStatus)


class StationMagnitude(AttribDict):
    """
    This class describes the magnitude derived from a single waveform stream.

    :type public_id: str
    :param public_id: Resource identifier of StationMagnitude.
    :type origin_id: str, optional
    :param origin_id: Reference to an origins’s ``public_id`` if the
        StationMagnitude has an associated :class:`~obspy.core.event.Origin`.
    :type mag: :class:`~obspy.core.event.FloatQuantity`
    :param mag: Estimated magnitude.
    :type type: str, optional
    :param type: Describes the type of magnitude. This is a free-text field
        because it is impossible to cover all existing magnitude type
        designations with an enumeration. Possible values are
            * unspecified magitude (``'M'``),
            * local magnitude (``'ML'``),
            * body wave magnitude (``'Mb'``),
            * surface wave magnitude (``'MS'``),
            * moment magnitude (``'Mw'``),
            * duration magnitude (``'Md'``)
            * coda magnitude (``'Mc'``)
            * ``'MH'``, ``'Mwp'``, ``'M50'``, ``'M100'``, etc.
    :type amplitude_id: str, optional
    :param amplitude_id: Identifies the data source of the StationMagnitude.
        For magnitudes derived from amplitudes in waveforms (e. g.,
        local magnitude ML), amplitude_id points to public_id in class
        :class:`obspy.core.event.Amplitude`.
    :type method_id: str, optional
    :param method_id: Identifies the method of magnitude estimation. Users
        should avoid to give contradictory information in method_id and type.
    :type waveform_id: str, optional
    :param waveform_id: Identifies the waveform stream.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    def __init__(self, public_id='', origin_id='', mag={}, type=None,
                 amplitude_id=None, method_id=None, waveform_id={},
                 comments=None, creation_info={}):
        # default attributes
        self.public_id = public_id
        self.origin_id = origin_id
        self.mag = FloatQuantity(mag)
        self.type = type
        self.amplitude_id = amplitude_id
        self.method_id = method_id
        self.waveform_id = WaveformStreamID(waveform_id)
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)


class EventDescription(AttribDict):
    """
    Free-form string with additional event description. This can be a
    well-known name, like 1906 San Francisco Earthquake. A number of categories
    can be given in type.

    :type text: str
    :param text: Free-form text with earthquake description.
    :type type: str, optional
    :param type: Category of earthquake description. Values can be taken from
        the following:
            * ``"felt report"``
            * ``"Flinn-Engdahl region"``
            * ``"local time"``
            * ``"tectonic summary"``
            * ``"nearest cities"``
            * ``"earthquake name"``
            * ``"region name"``
    """
    text = ''
    type = None

    def _getEventDescriptionType(self):
        return self.__dict__.get('type', None)

    def _setEventDescriptionType(self, value):
        self.__dict__['type'] = EventDescriptionType(value)

    type = property(_getEventDescriptionType, _setEventDescriptionType)


class Event(object):
    """
    The class Event describes a seismic event which does not necessarily need
    to be a tectonic earthquake. An event is usually associated with one or
    more origins, which contain information about focal time and geographical
    location of the event. Multiple origins can cover automatic and manual
    locations, a set of location from different agencies, locations generated
    with different location programs and earth models, etc. Furthermore, an
    event is usually associated with one or more magnitudes, and with one or
    more focal mechanism determinations.

    :type public_id: str, optional
    :param public_id: Resource identifier of Event.
    :type preferred_origin_id: str, optional
    :param preferred_origin_id: Refers to the ``public_id`` of the preferred
        :class:`~obspy.core.event.Origin` object.
    :type preferred_magnitude_id: str, optional
    :param preferred_magnitude_id: Refers to the ``public_id`` of the preferred
        :class:`~obspy.core.event.Magnitude` object.
    :type preferred_focal_mechanism_id: str, optional
    :param preferred_focal_mechanism_id: Refers to the ``public_id`` of the
        preferred :class:`~obspy.core.event.FocalMechanism` object.
    :type type: str, optional
    :param type: Describes the type of an event. Allowed values are the
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
    :type type_certainty: str, optional
    :param type_certainty: Denotes how certain the information on event type
        is. Allowed values are the following:
            * ``"suspected"``
            * ``"known"``
    :type description: list of :class:`~obspy.core.event.EventDescription`
    :param description: Additional event description, like earthquake name,
        Flinn-Engdahl region, etc.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    def __init__(self, public_id='', preferred_origin_id=None,
                 preferred_magnitude_id=None,
                 preferred_focal_mechanism_id=None, type=None,
                 type_certainty=None, descriptions=None, comments=None,
                 creation_info={}, origins=None, magnitudes=None,
                 station_magnitudes=None, focal_mechanism=None, picks=None,
                 amplitudes=None):
        # default attributes
        self.public_id = public_id
        self.preferred_origin_id = preferred_origin_id
        self.preferred_magnitude_id = preferred_magnitude_id
        self.preferred_focal_mechanism_id = preferred_focal_mechanism_id
        self.type = type
        self.type_certainty = type_certainty
        self.descriptions = descriptions or []
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)
        # child elements
        self.origins = origins or []
        self.magnitudes = magnitudes or []
        self.station_magnitudes = station_magnitudes or []
        self.focal_mechanism = focal_mechanism or []
        self.picks = picks or []
        self.amplitudes = amplitudes or []

    def __eq__(self, other):
        """
        Implements rich comparison of Event objects for "==" operator.

        Events are the same, if the have the same id.
        """
        # check if other object is a Event
        if not isinstance(other, Event):
            return False
        if self.id != other.id:
            return False
        return True

    def __str__(self):
        out = ''
        if self.preferred_origin:
            out += '%s | %+7.3f, %+8.3f' % (self.preferred_origin.time.value,
                                       self.preferred_origin.latitude.value,
                                       self.preferred_origin.longitude.value)
        if self.preferred_magnitude:
            out += ' | %s %-2s' % (self.preferred_magnitude.mag.value,
                                   self.preferred_magnitude.type)
        if self.preferred_origin and self.preferred_origin.evaluation_mode:
            out += ' | %s' % (self.preferred_origin.evaluation_mode)
        return out

    def _getEventType(self):
        return self.__type

    def _setEventType(self, value):
        self.__type = EventType(value)

    type = property(_getEventType, _setEventType)

    def _getEventTypeCertainty(self):
        return self.__type_certainty

    def _setEventTypeCertainty(self, value):
        self.__type_certainty = EventTypeCertainty(value)

    type_certainty = property(_getEventTypeCertainty, _setEventTypeCertainty)

    def _getPreferredMagnitude(self):
        if self.magnitudes:
            return self.magnitudes[0]
        return None

    preferred_magnitude = property(_getPreferredMagnitude)

    def _getPreferredOrigin(self):
        if self.origins:
            return self.origins[0]
        return None

    preferred_origin = property(_getPreferredOrigin)

    def _getPreferredFocalMechanism(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0]
        return None

    preferred_focal_mechanism = property(_getPreferredFocalMechanism)

    def _getTime(self):
        return self.preferred_origin.time.value

    time = datetime = property(_getTime)

    def _getLatitude(self):
        return self.preferred_origin.latitude.value

    latitude = lat = property(_getLatitude)

    def _getLongitude(self):
        return self.preferred_origin.longitude.value

    longitude = lon = property(_getLongitude)

    def _getMagnitude(self):
        return self.preferred_magnitude.magnitude.value

    magnitude = mag = property(_getMagnitude)

    def _getMagnitudeType(self):
        return self.preferred_magnitude.type

    magnitude_type = mag_type = property(_getMagnitudeType)

    def getId(self):
        """
        Returns the identifier of the event.

        :rtype: str
        :return: event identifier
        """
        if self.public_id is None:
            return ''
        return "%s" % (self.public_id)

    id = property(getId)


class Catalog(object):
    """
    This class serves as a container for Event objects.

    :type events: list of :class:`~obspy.core.event.Event`, optional
    :param events: List of events
    :type public_id: str, optional
    :param public_id: Resource identifier of the catalog.
    :type description: str, optional
    :param description: Description string that can be assigned to the
        earthquake catalog, or collection of events.
    :type comments: list of :class:`~obspy.core.event.Comment`, optional
    :param comments: Additional comments.
    :type creation_info: :class:`~obspy.core.event.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    """
    def __init__(self, events=None, public_id='', description=None,
                 comments=None, creation_info={}):
        """
        Initializes a Catalog object.
        """
        # default attributes
        self.public_id = public_id
        self.description = description
        self.comments = comments or []
        self.creation_info = CreationInfo(creation_info)
        # child elements
        self.events = events or []

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
        out = out + "\n".join([ev.__str__() for ev in self])
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
        QUAKEML  :mod:`obspy.mseed`   :func:`obspy.core.event.writeQUAKEML`
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
