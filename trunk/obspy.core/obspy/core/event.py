# -*- coding: utf-8 -*-
"""
Module for handling ObsPy Catalog and Event objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from glob import glob, iglob, has_magic
from obspy.core.util import NamedTemporaryFile, getExampleFile, uncompressFile
from obspy.core.util.attribdict import AttribDict
from obspy.core.util.base import _readFromPlugin
import copy
import os
import urllib2


def readEvents(pathname_or_url=None):
    """
    Read event files into an ObsPy Catalog object.

    The :func:`~obspy.core.event.readEvents` function opens either one or
    multiple event files given via file name or URL using the
    ``pathname_or_url`` attribute.

    :type pathname_or_url: string, optional
    :param pathname_or_url: String containing a file name or a URL. Wildcards
        are allowed for a file name. If this attribute is omitted, a Catalog
        object with an example data set will be created.
    :type format: string, optional
    :return: A ObsPy :class:`~obspy.core.event.Catalog` object.
    """
    # if no pathname or URL specified, make example stream
    if not pathname_or_url:
        return _createExampleCatalog()
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
    if "://" in pathname_or_url:
        # extract extension if any
        suffix = os.path.basename(pathname_or_url).partition('.')[2] or '.tmp'
        # some URL
        fh = NamedTemporaryFile(suffix=suffix)
        fh.write(urllib2.urlopen(pathname_or_url).read())
        fh.close()
        cat.extend(_read(fh.name).events)
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in iglob(pathname):
            cat.extend(_read(file).events)
        if len(cat) == 0:
            # try to give more specific information why the stream is empty
            if has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
    return cat


@uncompressFile
def _read(filename, format=None, **kwargs):
    """
    Reads a single event file into a ObsPy Catalog object.
    """
    catalog, format = _readFromPlugin('event', filename, format=None, **kwargs)
    for event in catalog:
        event._format = format
    return catalog


def _createExampleCatalog():
    """
    Create an example catalog.
    """
    return readEvents('/path/to/neries_events.xml')


class Origin(AttribDict):
    """
    Contains a single earthquake origin.
    """
    def __init__(self, **kwargs):
        self.public_id = None
        self.time = None
        self.latitude = None
        self.longitude = None
        self.quality = AttribDict()
        self.creation_info = AttribDict()
        self.origin_uncertainty = AttribDict()
        self.update(kwargs)

    def __str__(self):
        return self._pretty_str(['time', 'latitude', 'longitude'])


class Magnitude(AttribDict):
    """
    Contains a single earthquake magnitude.
    """
    def __init__(self, **kwargs):
        self.public_id = None
        self.magnitude = None
        self.type = None
        self.creation_info = AttribDict()
        self.update(kwargs)

    def __str__(self):
        return self._pretty_str(['magnitude'])


class Event(object):
    """
    Seismological event containing origins, picks, magnitudes, etc.
    """
    public_id = None
    type = None
    type_certainty = None
    description = None
    comment = None
    creation_info = None
    picks = []
    amplitudes = []
    station_magnitudes = []
    focal_mechanism = []
    origins = []
    magnitudes = []

    def __eq__(self, other):
        """
        Implements rich comparison of Event objects for "==" operator.

        Events are the same, if the have the same id.
        """
        # check if other object is a Trace
        if not isinstance(other, Event):
            return False
        if self.id != other.id:
            return False
        return True

    def __str__(self):
        out = ''
        if self.preferred_origin:
            out += '%s | %+7.3f, %+8.3f' % (self.preferred_origin.time,
                                       self.preferred_origin.latitude,
                                       self.preferred_origin.longitude)
        if self.preferred_magnitude:
            out += ' | %s %-2s' % (self.preferred_magnitude.magnitude,
                                   self.preferred_magnitude.type)
        if self.preferred_origin and self.preferred_origin.evaluation_mode:
            out += ' | %s' % (self.preferred_origin.evaluation_mode)
        return out

    def _getPreferredMagnitude(self):
        if self.magnitudes:
            return self.magnitudes[0]
        return None

    preferred_magnitude = property(_getPreferredMagnitude)

    def _getPreferredMagnitudeID(self):
        if self.magnitudes:
            return self.magnitudes[0].public_id
        return None

    preferred_magnitude_id = property(_getPreferredMagnitudeID)

    def _getPreferredOrigin(self):
        if self.origins:
            return self.origins[0]
        return None

    preferred_origin = property(_getPreferredOrigin)

    def _getPreferredOriginID(self):
        if self.origins:
            return self.origins[0].public_id
        return None

    preferred_origin_id = property(_getPreferredOriginID)

    def _getPreferredFocalMechanism(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0]
        return None

    preferred_focal_mechanism = property(_getPreferredFocalMechanism)

    def _getPreferredFocalMechanismID(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0].public_id
        return None

    preferred_focal_mechanism_id = property(_getPreferredFocalMechanismID)

    def _getTime(self):
        return self.preferred_origin.time

    time = datetime = property(_getTime)

    def _getLatitude(self):
        return self.preferred_origin.latitude

    latitude = lat = property(_getLatitude)

    def _getLongitude(self):
        return self.preferred_origin.longitude

    longitude = lon = property(_getLongitude)

    def _getMagnitude(self):
        return self.preferred_magnitude.magnitude

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
    Seismological event catalog containing a list of events.
    """
    public_id = None
    description = None
    comment = None
    creation_info = None

    def __init__(self, events=None):
        self.events = []
        if isinstance(events, Event):
            events = [events]
        if events:
            self.events.extend(events)

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

    def write(self, filename, format):
        """
        Exports catalog to file system using given format.
        """
        raise NotImplementedError

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
