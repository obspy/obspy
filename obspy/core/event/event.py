# -*- coding: utf-8 -*-
"""
Provides the Event class

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
from itertools import chain

from obspy.core.event.header import (
    EventType, EventTypeCertainty, EventDescriptionType)
from obspy.core.event.resourceid import ResourceIdentifier
from obspy.core.util.misc import _yield_resource_id_parent_attr
from obspy.imaging.source import plot_radiation_pattern, _setup_figure_and_axes


from .base import _event_type_class_factory, CreationInfo


__Event = _event_type_class_factory(
    "__Event",
    class_attributes=[("resource_id", ResourceIdentifier),
                      ("event_type", EventType),
                      ("event_type_certainty", EventTypeCertainty),
                      ("creation_info", CreationInfo),
                      ("preferred_origin_id", ResourceIdentifier),
                      ("preferred_magnitude_id", ResourceIdentifier),
                      ("preferred_focal_mechanism_id", ResourceIdentifier)],
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

    :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
    :param resource_id: Resource identifier of Event.
    :type force_resource_id: bool, optional
    :param force_resource_id: If set to False, the automatic initialization of
        `resource_id` attribute in case it is not specified will be skipped.
    :type event_type: str, optional
    :param event_type: Describes the type of an event.
        See :class:`~obspy.core.event.header.EventType` for allowed values.
    :type event_type_certainty: str, optional
    :param event_type_certainty: Denotes how certain the information on event
        type is.
        See :class:`~obspy.core.event.header.EventTypeCertainty` for allowed
        values.
    :type creation_info: :class:`~obspy.core.event.base.CreationInfo`, optional
    :param creation_info: Creation information used to describe author,
        version, and creation time.
    :type event_descriptions: list of
        :class:`~obspy.core.event.event.EventDescription`
    :param event_descriptions: Additional event description, like earthquake
        name, Flinn-Engdahl region, etc.
    :type comments: list of :class:`~obspy.core.event.base.Comment`, optional
    :param comments: Additional comments.
    :type picks: list of :class:`~obspy.core.event.origin.Pick`
    :param picks: Picks associated with the event.
    :type amplitudes: list of :class:`~obspy.core.event.magnitude.Amplitude`
    :param amplitudes: Amplitudes associated with the event.
    :type focal_mechanisms: list of
        :class:`~obspy.core.event.source.FocalMechanism`
    :param focal_mechanisms: Focal mechanisms associated with the event
    :type origins: list of :class:`~obspy.core.event.origin.Origin`
    :param origins: Origins associated with the event.
    :type magnitudes: list of :class:`~obspy.core.event.magnitude.Magnitude`
    :param magnitudes: Magnitudes associated with the event.
    :type station_magnitudes: list of
        :class:`~obspy.core.event.magnitude.StationMagnitude`
    :param station_magnitudes: Station magnitudes associated with the event.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """
    do_not_warn_on = ["_format", "extra"]

    def __init__(self, *args, **kwargs):
        super(Event, self).__init__(*args, **kwargs)
        self.scope_resource_ids()

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
            # get lat, lon, time and handle if any are None (#2119)
            lat, lon, time = origin.latitude, origin.longitude, origin.time
            lat_str = '%+7.3f' % lat if lat is not None else 'None'
            lon_str = '%+8.3f' % lon if lon is not None else 'None'
            out += '%s | %s, %s' % (time, lat_str, lon_str)
        if self.magnitudes:
            magnitude = self.preferred_magnitude() or self.magnitudes[0]
            try:
                if round(magnitude.mag, 1) == magnitude.mag:
                    mag_string = '%3.1f ' % magnitude.mag
                else:
                    mag_string = '%4.2f' % magnitude.mag
            except TypeError:
                mag_string = str(magnitude.mag)
            out += ' | %s %-2s' % (mag_string,
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
        if self.preferred_origin_id is None:
            return None
        return self.preferred_origin_id.get_referred_object()

    def preferred_magnitude(self):
        """
        Returns the preferred magnitude
        """
        if self.preferred_magnitude_id is None:
            return None
        return self.preferred_magnitude_id.get_referred_object()

    def preferred_focal_mechanism(self):
        """
        Returns the preferred focal mechanism
        """
        if self.preferred_focal_mechanism_id is None:
            return None
        return self.preferred_focal_mechanism_id.get_referred_object()

    def plot(self, kind=[['ortho', 'beachball'], ['p_sphere', 's_sphere']],
             subplot_size=4.0, show=True, outfile=None, **kwargs):
        """
        Plot event location and/or the preferred focal mechanism
        and radiation pattern.

        :type kind: list[str] or list[list[str]]
        :param kind: A list of strings (for a 1-row plot) or a nested list of
            strings (one list of strings per row), with the following keywords
            to generate a matplotlib figure:

            * ``'ortho'`` (Orthographic plot of event location
              see :meth:`~obspy.core.event.catalog.Catalog.plot`)
            * ``'global'`` (Global plot of event location
              see :meth:`~obspy.core.event.catalog.Catalog.plot`)
            * ``'local'`` (Local plot of event location
              see :meth:`~obspy.core.event.catalog.Catalog.plot`)
            * ``'beachball'`` (Beachball of preferred focal mechanism)
            * ``'p_quiver'`` (quiver plot of p wave farfield)
            * ``'s_quiver'`` (quiver plot of s wave farfield)
            * ``'p_sphere'`` (surface plot of p wave farfield)
            * ``'s_sphere'`` (surface plot of s wave farfield)

        :type subplot_size: float
        :param subplot_size: Width/height of one single subplot cell in inches.
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before
            showing it. Has no effect if `outfile` is specified.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option, image
            will not be displayed interactively. The given path/filename is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.
            The figure is closed after saving it to file.
        :returns: Figure instance with the plot.

        .. rubric:: Examples

        Default plot includes an orthographic map plot, a beachball plot and
        plots of P/S farfield radiation patterns (preferred -- or first --
        focal mechanism has to have a moment tensor set).

        >>> from obspy import read_events
        >>> event = read_events("/path/to/CMTSOLUTION")[0]
        >>> event.plot()  # doctest:+SKIP

        .. plot::

            from obspy import read_events
            event = read_events("/path/to/CMTSOLUTION")[0]
            event.plot()

        Individual subplot parts and the setup of the grid of subplots
        (rows/columns) can be specified by using certain keywords, see `kind`
        parameter description.

        >>> event.plot(kind=[['global'],
        ...                  ['p_sphere', 'p_quiver']])  # doctest:+SKIP

        .. plot::

            from obspy import read_events
            event = read_events("/path/to/CMTSOLUTION")[0]
            event.plot(kind=[['global'], ['p_sphere', 'p_quiver']])
        """
        import matplotlib.pyplot as plt
        from .catalog import Catalog
        try:
            fm = self.preferred_focal_mechanism() or self.focal_mechanisms[0]
            mtensor = fm.moment_tensor.tensor
        except (IndexError, AttributeError) as e:
            msg = "Could not access event's moment tensor ({}).".format(str(e))
            raise ValueError(msg)

        mt = [mtensor.m_rr, mtensor.m_tt, mtensor.m_pp,
              mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]

        if len(kind) == 1:
            kind_ = kind
        else:
            kind_ = list(chain(*kind))

        if any([k_ in ("ortho", "global", "local") for k_ in kind_]):
            cat_ = Catalog([self])
            kwargs["events"] = cat_

        fig, axes, kind_ = _setup_figure_and_axes(kind,
                                                  subplot_size=subplot_size,
                                                  **kwargs)
        for ax, kind__ in zip(axes, kind_):
            if kind__ in ("ortho", "global", "local"):
                ax.stock_img()
                ax.gridlines()
                ax.coastlines()
                cat_.plot(projection=kind__, fig=ax, show=False,
                          **kwargs)
                # shrink plot a bit to avoid it looking oversized compared to
                # 3d axes that have some white space around them
                # if kind__ == "ortho":
                #     scale = 0.8
                #     for getter, setter in zip((ax.get_xlim, ax.get_ylim),
                #                               (ax.set_xlim, ax.set_ylim)):
                #         min_, max_ = getter()
                #         margin = (max_ - min_) * (1 - scale) / 2.0
                #         setter(min_ - margin, max_ + margin)
        plot_radiation_pattern(
            mt, kind=kind, coordinate_system='RTP', fig=fig, show=False)

        # fig.tight_layout(pad=0.1)

        if outfile:
            fig.savefig(outfile)
            plt.close(fig)
        else:
            if show:
                plt.show()

        return fig

    def __deepcopy__(self, memodict=None):
        """
        reset resource_id's object_id after deep copy to allow the
        object specific behavior of get_referred_object
        """
        memodict = memodict or {}
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        result.scope_resource_ids()
        return result

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self.scope_resource_ids()

    def write(self, filename, format, **kwargs):
        """
        Saves event information into a file.

        :type filename: str
        :param filename: The name of the file to write.
        :type format: str
        :param format: The file format to use (e.g. ``"QUAKEML"``). See
            :meth:`obspy.core.event.catalog.Catalog.write()` for a list of
            supported formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            plugin's writer method.

        .. rubric:: Example

        >>> from obspy import read_events
        >>> event = read_events()[0]  # doctest: +SKIP
        >>> event.write("example.xml", format="QUAKEML")  # doctest: +SKIP
        """
        from .catalog import Catalog
        Catalog(events=[self]).write(filename, format, **kwargs)

    def scope_resource_ids(self):
        """
        Ensure all resource_ids in event instance are event-scoped.

        This will ensure the resource_ids refer to objects in the event
        structure when possible.
        """
        gen = _yield_resource_id_parent_attr(self)

        for resource_id, parent, attr in gen:
            if attr == 'resource_id':
                resource_id.set_referred_object(parent, parent=self,
                                                warn=False)
            else:
                resource_id._parent_key = self
                resource_id._object_id = None


__EventDescription = _event_type_class_factory(
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
    :param type: Category of earthquake description.
        See :class:`~obspy.core.event.header.EventDescriptionType` for allowed
        values.

    .. note::

        For handling additional information not covered by the QuakeML
        standard and how to output it to QuakeML see the
        :ref:`ObsPy Tutorial <quakeml-extra>`.
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
