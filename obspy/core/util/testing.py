# -*- coding: utf-8 -*-
"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import difflib
import doctest
import inspect
import io
import os
import re
import warnings

import numpy as np
from lxml import etree

MODULE_TEST_SKIP_CHECKS = {}


def compare_xml_strings(doc1, doc2):
    """
    Simple helper function to compare two XML byte strings.

    :type doc1: bytes
    :type doc2: bytes
    """
    obj1 = etree.fromstring(doc1).getroottree()
    obj2 = etree.fromstring(doc2).getroottree()

    buf = io.BytesIO()
    obj1.write_c14n(buf)
    buf.seek(0, 0)
    str1 = buf.read().decode()
    str1 = [_i.strip() for _i in str1.splitlines()]

    buf = io.BytesIO()
    obj2.write_c14n(buf)
    buf.seek(0, 0)
    str2 = buf.read().decode()
    str2 = [_i.strip() for _i in str2.splitlines()]

    unified_diff = difflib.unified_diff(str1, str2)

    err_msg = "\n".join(unified_diff)
    if err_msg:  # pragma: no cover
        raise AssertionError("Strings are not equal.\n" + err_msg)


def remove_unique_ids(xml_string, remove_creation_time=False):
    """
    Removes unique ID parts of e.g. 'publicID="..."' attributes from xml
    strings.

    :type xml_string: str
    :param xml_string: xml string to process
    :type remove_creation_time: bool
    :param xml_string: controls whether to remove 'creationTime' tags or not.
    :rtype: str
    """
    prefixes = ["id", "publicID", "pickID", "originID", "preferredOriginID",
                "preferredMagnitudeID", "preferredFocalMechanismID",
                "referenceSystemID", "methodID", "earthModelID",
                "triggeringOriginID", "derivedOriginID", "momentMagnitudeID",
                "greensFunctionID", "filterID", "amplitudeID",
                "stationMagnitudeID", "earthModelID", "slownessMethodID",
                "pickReference", "amplitudeReference"]
    if remove_creation_time:
        prefixes.append("creationTime")
    for prefix in prefixes:
        xml_string = re.sub("%s='.*?'" % prefix, '%s=""' % prefix, xml_string)
        xml_string = re.sub('%s=".*?"' % prefix, '%s=""' % prefix, xml_string)
        xml_string = re.sub("<%s>.*?</%s>" % (prefix, prefix),
                            '<%s/>' % prefix, xml_string)
    return xml_string


def get_all_py_files():
    """
    Return a list with full absolute paths to all .py files in ObsPy file tree.

    :rtype: list[str]
    """
    util_dir = os.path.abspath(inspect.getfile(inspect.currentframe()))
    obspy_dir = os.path.dirname(os.path.dirname(os.path.dirname(util_dir)))
    py_files = set()
    # Walk the obspy directory structure
    for dirpath, _, filenames in os.walk(obspy_dir):
        py_files.update([os.path.abspath(os.path.join(dirpath, i)) for i in
                         filenames if i.endswith(".py")])
    return sorted(py_files)


class WarningsCapture(object):
    """
    Try hard to capture all warnings.

    Aims to be a reliable drop-in replacement for built-in
    warnings.catch_warnings() context manager.

    Based on pytest's _DeprecatedCallContext context manager.
    """
    def __enter__(self):
        self.captured_warnings = []
        self._old_warn = warnings.warn
        self._old_warn_explicit = warnings.warn_explicit
        warnings.warn_explicit = self._warn_explicit
        warnings.warn = self._warn
        return self

    def _warn_explicit(self, message, category, *args, **kwargs):
        self.captured_warnings.append(
            warnings.WarningMessage(message=category(message),
                                    category=category,
                                    filename="", lineno=0))

    def _warn(self, message, category=Warning, *args, **kwargs):
        if isinstance(message, Warning):
            self.captured_warnings.append(
                warnings.WarningMessage(
                    message=category(message), category=category or Warning,
                    filename="", lineno=0))
        else:
            self.captured_warnings.append(
                warnings.WarningMessage(
                    message=category(message), category=category,
                    filename="", lineno=0))

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.warn_explicit = self._old_warn_explicit
        warnings.warn = self._old_warn

    def __len__(self):
        return len(self.captured_warnings)

    def __getitem__(self, key):
        return self.captured_warnings[key]


def create_diverse_catalog():
    """
    Create a catalog with a single event that has many features.

    Uses most the event related classes.
    """

    # imports are here in order to avoid circular import issues
    import obspy.core.event as ev
    from obspy import UTCDateTime, Catalog
    # local dict for storing state
    state = dict(time=UTCDateTime('2016-05-04T12:00:01'))

    def _create_event():
        event = ev.Event(
            event_type='mining explosion',
            event_descriptions=[_get_event_description()],
            picks=[_create_pick()],
            origins=[_create_origins()],
            station_magnitudes=[_get_station_mag()],
            magnitudes=[_create_magnitudes()],
            amplitudes=[_get_amplitudes()],
            focal_mechanisms=[_get_focal_mechanisms()],
        )
        # set preferred origin, focal mechanism, magnitude
        preferred_objects = dict(
            origin=event.origins[-1].resource_id,
            focal_mechanism=event.focal_mechanisms[-1].resource_id,
            magnitude=event.magnitudes[-1].resource_id,
        )
        for item, value in preferred_objects.items():
            setattr(event, 'preferred_' + item + '_id', value)

        event.scope_resource_ids()
        return event

    def _create_pick():
        # setup some of the classes
        creation = ev.CreationInfo(
            agency='SwanCo',
            author='Indago',
            creation_time=UTCDateTime(),
            version='10.10',
            author_url=ev.ResourceIdentifier('smi:local/me.com'),
        )

        pick = ev.Pick(
            time=state['time'],
            comments=[ev.Comment(x) for x in 'BOB'],
            evaluation_mode='manual',
            evaluation_status='final',
            creation_info=creation,
            phase_hint='P',
            polarity='positive',
            onset='emergent',
            back_azimith_errors={"uncertainty": 10},
            slowness_method_id=ev.ResourceIdentifier('smi:local/slow'),
            backazimuth=122.1,
            horizontal_slowness=12,
            method_id=ev.ResourceIdentifier(),
            horizontal_slowness_errors={'uncertainty': 12},
            filter_id=ev.ResourceIdentifier(),
            waveform_id=ev.WaveformStreamID('UU', 'FOO', '--', 'HHZ'),
        )
        state['pick_id'] = pick.resource_id
        return pick

    def _create_origins():
        ori = ev.Origin(
            resource_id=ev.ResourceIdentifier('smi:local/First'),
            time=UTCDateTime('2016-05-04T12:00:00'),
            time_errors={'uncertainty': .01},
            longitude=-111.12525,
            longitude_errors={'uncertainty': .020},
            latitude=47.48589325,
            latitude_errors={'uncertainty': .021},
            depth=2.123,
            depth_errors={'uncertainty': 1.22},
            depth_type='from location',
            time_fixed=False,
            epicenter_fixed=False,
            reference_system_id=ev.ResourceIdentifier(),
            method_id=ev.ResourceIdentifier(),
            earth_model_id=ev.ResourceIdentifier(),
            arrivals=[_get_arrival()],
            composite_times=[_get_composite_times()],
            quality=_get_origin_quality(),
            origin_type='hypocenter',
            origin_uncertainty=_get_origin_uncertainty(),
            region='US',
            evaluation_mode='manual',
            evaluation_status='final',
        )
        state['origin_id'] = ori.resource_id
        return ori

    def _get_arrival():
        return ev.Arrival(
            resource_id=ev.ResourceIdentifier('smi:local/Ar1'),
            pick_id=state['pick_id'],
            phase='P',
            time_correction=.2,
            azimuth=12,
            distance=10,
            takeoff_angle=15,
            takeoff_angle_errors={'uncertainty': 10.2},
            time_residual=.02,
            horizontal_slowness_residual=12.2,
            backazimuth_residual=12.2,
            time_weight=.23,
            horizontal_slowness_weight=12,
            backazimuth_weight=12,
            earth_model_id=ev.ResourceIdentifier(),
            commens=[ev.Comment(x) for x in 'Nothing'],
        )

    def _get_composite_times():
        return ev.CompositeTime(
            year=2016,
            year_errors={'uncertainty': 0},
            month=5,
            month_errors={'uncertainty': 0},
            day=4,
            day_errors={'uncertainty': 0},
            hour=0,
            hour_errors={'uncertainty': 0},
            minute=0,
            minute_errors={'uncertainty': 0},
            second=0,
            second_errors={'uncertainty': .01}
        )

    def _get_origin_quality():
        return ev.OriginQuality(
            associate_phase_count=1,
            used_phase_count=1,
            associated_station_count=1,
            used_station_count=1,
            depth_phase_count=1,
            standard_error=.02,
            azimuthal_gap=.12,
            ground_truth_level='GT0',
        )

    def _get_origin_uncertainty():
        return ev.OriginUncertainty(
            horizontal_uncertainty=1.2,
            min_horizontal_uncertainty=.12,
            max_horizontal_uncertainty=2.2,
            confidence_ellipsoid=_get_confidence_ellipsoid(),
            preferred_description="uncertainty ellipse",
        )

    def _get_confidence_ellipsoid():
        return ev.ConfidenceEllipsoid(
            semi_major_axis_length=12,
            semi_minor_axis_length=12,
            major_axis_plunge=12,
            major_axis_rotation=12,
        )

    def _create_magnitudes():
        return ev.Magnitude(
            resource_id=ev.ResourceIdentifier(),
            mag=5.5,
            mag_errors={'uncertainty': .01},
            magnitude_type='Mw',
            origin_id=state['origin_id'],
            station_count=1,
            station_magnitude_contributions=[_get_station_mag_contrib()],
        )

    def _get_station_mag():
        station_mag = ev.StationMagnitude(
            mag=2.24,
        )
        state['station_mag_id'] = station_mag.resource_id
        return station_mag

    def _get_station_mag_contrib():
        return ev.StationMagnitudeContribution(
            station_magnitude_id=state['station_mag_id'],
        )

    def _get_event_description():
        return ev.EventDescription(
            text='some text about the EQ',
            type='earthquake name',
        )

    def _get_amplitudes():
        return ev.Amplitude(
            generic_amplitude=.0012,
            type='A',
            unit='m',
            period=1,
            time_window=_get_timewindow(),
            pick_id=state['pick_id'],
            scalling_time=state['time'],
            mangitude_hint='ML',
            scaling_time_errors=ev.QuantityError(uncertainty=42.0),
        )

    def _get_timewindow():
        return ev.TimeWindow(
            begin=1.2,
            end=2.2,
            reference=UTCDateTime('2016-05-04T12:00:00'),
        )

    def _get_focal_mechanisms():
        return ev.FocalMechanism(
            nodal_planes=_get_nodal_planes(),
            principal_axis=_get_principal_axis(),
            azimuthal_gap=12,
            station_polarity_count=12,
            misfit=.12,
            station_distribution_ratio=.12,
            moment_tensor=_get_moment_tensor(),
        )

    def _get_nodal_planes():
        return ev.NodalPlanes(
            nodal_plane_1=ev.NodalPlane(strike=12, dip=2, rake=12),
            nodal_plane_2=ev.NodalPlane(strike=12, dip=2, rake=12),
            preferred_plane=2,
        )

    def _get_principal_axis():
        return ev.PrincipalAxes(
            t_axis=15,
            p_axis=15,
            n_axis=15,
        )

    def _get_moment_tensor():
        return ev.MomentTensor(
            scalar_moment=12213,
            tensor=_get_tensor(),
            variance=12.23,
            variance_reduction=98,
            double_couple=.22,
            clvd=.55,
            iso=.33,
            source_time_function=_get_source_time_function(),
            data_used=[_get_data_used()],
            method_id=ev.ResourceIdentifier(),
            inversion_type='general',
        )

    def _get_tensor():
        return ev.Tensor(
            m_rr=12,
            m_rr_errors={'uncertainty': .01},
            m_tt=12,
            m_pp=12,
            m_rt=12,
            m_rp=12,
            m_tp=12,
        )

    def _get_source_time_function():
        return ev.SourceTimeFunction(
            type='triangle',
            duration=.12,
            rise_time=.33,
            decay_time=.23,
        )

    def _get_data_used():
        return ev.DataUsed(
            wave_type='body waves',
            station_count=12,
            component_count=12,
            shortest_period=1,
            longest_period=20,
        )

    events = [_create_event()]
    return Catalog(events=events)


def setup_context_testcase(test_case, cm):
    """
    Use a contextmanager to set up a unittest test case.

    Inspired by Ned Batchelder's recipe found here: goo.gl/8TBJ7s.

    :param test_case:
        An instance of unittest.TestCase
    :param cm:
        Any instances which implements the context manager protocol,
        ie its class definition implements __enter__ and __exit__ methods.
    """
    val = cm.__enter__()
    test_case.addCleanup(cm.__exit__, None, None, None)
    return val


def streams_almost_equal(st1, st2, default_stats=True, rtol=1e-05, atol=1e-08,
                         equal_nan=True):
    """
    Return True if two streams are almost equal.

    :param st1: The first :class:`~obspy.core.stream.Stream` object.
    :param st2: The second :class:`~obspy.core.stream.Stream` object.
    :param default_stats:
        If True only compare the default stats on the traces, such as seed
        identification codes, start/end times, sampling_rates, etc. If
        False also compare extra stats attributes such as processing and
        format specific information.
    :param rtol: The relative tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param atol: The absolute tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param equal_nan:
        If ``True`` NaNs are evaluated equal when comparing the time
        series.
    :return: bool

    .. rubric:: Example

    1) Changes to the non-default parameters of the
        :class:`~obspy.core.trace.Stats` objects of the stream's contained
        :class:`~obspy.core.trace.Trace` objects will cause the streams to
        be considered unequal, but they will be considered almost equal.

        >>> from obspy import read
        >>> st1 = read()
        >>> st2 = read()
        >>> # The traces should, of course, be equal.
        >>> assert st1 == st2
        >>> # Perform detrending on st1 twice so processing stats differ.
        >>> st1 = st1.detrend('linear')
        >>> st1 = st1.detrend('linear')
        >>> st2 = st2.detrend('linear')
        >>> # The traces are no longer equal, but are almost equal.
        >>> assert st1 != st2
        >>> assert streams_almost_equal(st1, st2)


    2) Slight differences in each trace's data will cause the streams
        to be considered unequal, but they will be almost equal if the
        differences don't exceed the limits set by the ``rtol`` and
        ``atol`` parameters.

        >>> from obspy import read
        >>> st1 = read()
        >>> st2 = read()
        >>> # Perturb the trace data in st2 slightly.
        >>> for tr in st2:
        ...     tr.data *= (1 + 1e-6)
        >>> # The streams are no longer equal.
        >>> assert st1 != st2
        >>> # But they are almost equal.
        >>> assert streams_almost_equal(st1, st2)
        >>> # Unless, of course, there is a large change.
        >>> st1[0].data *= 10
        >>> assert not streams_almost_equal(st1, st2)
    """
    from obspy.core.stream import Stream
    # Return False if both objects are not streams or not the same length.
    are_streams = isinstance(st1, Stream) and isinstance(st2, Stream)
    if not are_streams or not len(st1) == len(st2):
        return False
    # Kwargs to pass trace_almost_equal.
    tr_kwargs = dict(default_stats=default_stats, rtol=rtol, atol=atol,
                     equal_nan=equal_nan)
    # Ensure the streams are sorted (as done with the __equal__ method)
    st1_sorted = st1.select()
    st1_sorted.sort()
    st2_sorted = st2.select()
    st2_sorted.sort()
    # Iterate over sorted trace pairs and determine if they are almost equal.
    for tr1, tr2 in zip(st1_sorted, st2_sorted):
        if not traces_almost_equal(tr1, tr2, **tr_kwargs):
            return False  # If any are not almost equal return None.
    return True


def traces_almost_equal(tr1, tr2, default_stats=True, rtol=1e-05, atol=1e-08,
                        equal_nan=True):
    """
    Return True if the two traces are almost equal.

    :param tr1: The first :class:`~obspy.core.trace.Trace` object.
    :param tr2: The second :class:`~obspy.core.trace.Trace` object.
    :param default_stats:
        If True only compare the default stats on the traces, such as seed
        identification codes, start/end times, sampling_rates, etc. If
        False also compare extra stats attributes such as processing and
        format specific information.
    :param rtol: The relative tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param atol: The absolute tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param equal_nan:
        If ``True`` NaNs are evaluated equal when comparing the time
        series.
    :return: bool
    """
    from obspy.core.trace import Trace
    # If other isnt  a trace, or data is not the same len return False.
    if not isinstance(tr2, Trace) or len(tr1.data) != len(tr2.data):
        return False
    # First compare the array values
    try:  # Use equal_nan if available
        all_close = np.allclose(tr1.data, tr2.data, rtol=rtol,
                                atol=atol, equal_nan=equal_nan)
    except TypeError:
        # This happens on very old versions of numpy. Essentially
        # we just need to handle NaN detection on our own, if equal_nan.
        is_close = np.isclose(tr1.data, tr2.data, rtol=rtol, atol=atol)
        if equal_nan:
            isnan = np.isnan(tr1.data) & np.isnan(tr2.data)
        else:
            isnan = np.zeros(tr1.data.shape).astype(bool)
        all_close = np.all(isnan | is_close)
    # Then compare the stats objects
    stats1 = _make_stats_dict(tr1, default_stats)
    stats2 = _make_stats_dict(tr2, default_stats)
    return all_close and stats1 == stats2


def _make_stats_dict(tr, default_stats):
    """
    Return a dict of stats from trace optionally including processing.
    """
    from obspy.core.trace import Stats
    if not default_stats:
        return dict(tr.stats)
    return {i: tr.stats[i] for i in Stats.defaults}


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
