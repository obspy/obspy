# -*- coding: utf-8 -*-
"""
obspy.core.event.event - Event Class
======================================================

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

from obspy.core.event_header import EventType, EventTypeCertainty

from obspy.core.util.base import ENTRY_POINTS

from .base import (_event_type_class_factory,
                   CreationInfo, ResourceIdentifier)

from .radpattern import plot_3drpattern


EVENT_ENTRY_POINTS = ENTRY_POINTS['event']
EVENT_ENTRY_POINTS_WRITE = ENTRY_POINTS['event_write']
ATTRIBUTE_HAS_ERRORS = True


__Event = _event_type_class_factory(
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
                get_referred_object()
        except AttributeError:
            return None

    def preferred_magnitude(self):
        """
        Returns the preferred magnitude
        """
        try:
            return ResourceIdentifier(self.preferred_magnitude_id).\
                get_referred_object()
        except AttributeError:
            return None

    def preferred_focal_mechanism(self):
        """
        Returns the preferred focal mechanism
        """
        try:
            return ResourceIdentifier(self.preferred_focal_mechanism_id).\
                get_referred_object()
        except AttributeError:
            return None

    def plot(self):
        """
        plots the preferred focal mechanism and radiation pattern
        """
        fm = self.preferred_focal_mechanism() or self.focal_mechanisms[0]
        try:
            mtensor = fm.moment_tensor.tensor
            mt = [mtensor.m_rr, mtensor.m_tt, mtensor.m_pp,
                  mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]
            plot_3drpattern(mt, kind='p_sphere')
        except AttributeError as err:
            print(err)
            print('Couldn\'t access event\'s moment tensor')

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

        >>> from obspy import read_events
        >>> event = read_events()[0]  # doctest: +SKIP
        >>> event.write("example.xml", format="QUAKEML")  # doctest: +SKIP
        """
        Catalog(events=[self]).write(filename, format, **kwargs)

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
