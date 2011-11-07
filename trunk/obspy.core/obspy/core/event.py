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
import os
import urllib2


def readQuakeML(pathname_or_url=None):
    """
    Read QuakeML files into an ObsPy Catalog object.

    The :func:`~obspy.core.event.readQuakeML` function opens either one or
    multiple QuakeML formated event files given via file name or URL using the
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
        cat.extend(_readQuakeML(fh.name).traces)
        os.remove(fh.name)
    else:
        # file name
        pathname = pathname_or_url
        for file in iglob(pathname):
            cat.extend(_readQuakeML(file).traces)
        if len(cat) == 0:
            # try to give more specific information why the stream is empty
            if has_magic(pathname) and not glob(pathname):
                raise Exception("No file matching file pattern: %s" % pathname)
            elif not has_magic(pathname) and not os.path.isfile(pathname):
                raise IOError(2, "No such file or directory", pathname)
    return cat


@uncompressFile
def _readQuakeML(filename):
    """
    Reads a single QuakeML file and returns a ObsPy Catalog object.
    """
    pass


def _createExampleCatalog():
    """
    Create an example catalog.
    """
    cat = Catalog()
    return cat


class Event(object):
    """
    Seismological event containing origin, picks, magnitudes, etc.
    """
    def __init__(self, origins=[], magnitudes=[], amplitudes=[], picks=[],
                 focal_mechanism=[], station_magnitudes=[], type=''):
        self.focal_mechanism = focal_mechanism
        self.origins = origins
        self.magnitudes = magnitudes
        self.station_magnitudes = station_magnitudes
        self.picks = picks
        self.amplitudes = amplitudes
        self.type = type

    def __str__(self):
        return 'Event (%s)' % (id(self))

    def getPreferredMagnitude(self):
        if self.magnitudes:
            return self.magnitudes[0]
        return None

    preferred_magnitude = property(getPreferredMagnitude)

    def getPreferredMagnitudeID(self):
        if self.magnitudes:
            return self.magnitudes[0].public_id
        return None

    preferred_magnitude_id = property(getPreferredMagnitudeID)

    def getPreferredOrigin(self):
        if self.origins:
            return self.origins[0]
        return None

    preferred_origin = property(getPreferredOrigin)

    def getPreferredOriginID(self):
        if self.origins:
            return self.origins[0].public_id
        return None

    preferred_origin_id = property(getPreferredOriginID)

    def getPreferredFocalMechanism(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0]
        return None

    preferred_focal_mechanism = property(getPreferredFocalMechanism)

    def getPreferredFocalMechanismID(self):
        if self.focal_mechanism:
            return self.focal_mechanism[0].public_id
        return None

    preferred_focal_mechanism_id = property(getPreferredFocalMechanismID)


class Catalog(object):
    """
    Seismological event catalog containing a list of events.
    """
    def __init__(self, events=None):
        self.events = []
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
            self.extend(event_list.traces)
        else:
            msg = 'Extend only supports a list of Event objects as argument.'
            raise TypeError(msg)

    def write(self, filename, format):
        """
        Exports catalog to file system using given format.
        """
        raise NotImplementedError

    def plot(self, filename, format):
        """
        Creates preview map of all events in current Catalog object.
        """
        raise NotImplementedError


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
