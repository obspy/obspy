# -*- coding: utf-8 -*-
import io
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from obspy import UTCDateTime, read_events
from obspy.core.event import (Catalog, Comment, CreationInfo, Event,
                              FocalMechanism, Magnitude, Origin, Pick,
                              ResourceIdentifier, WaveformStreamID)
from obspy.core.event.source import farfield
from obspy.core.util import CARTOPY_VERSION
from obspy.core.util.base import _get_entry_points
from obspy.core.util.misc import MatplotlibBackend
from obspy.core.util.testing import WarningsCapture
from obspy.core.event.base import QuantityError


if CARTOPY_VERSION and CARTOPY_VERSION >= [0, 12, 0]:
    HAS_CARTOPY = True
else:
    HAS_CARTOPY = False


class TestEvent:
    """
    Test suite for obspy.core.event.event.Event
    """

    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = read_events()[1]
        s = event.short_str()
        expected = ("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" +
                    " | 4.3  ML | manual")
        assert s == expected

    def test_str_empty_origin(self):
        """
        Ensure an event with an empty origin returns a str without raising a
        TypeError (#2119).
        """
        event = Event(origins=[Origin()])
        out = event.short_str()
        assert isinstance(out, str)
        assert out == 'None | None, None'

    def test_eq(self):
        """
        Testing the __eq__ method of the Event object.
        """
        # events are equal if the have the same public_id
        # Catch warnings about the same different objects with the same
        # resource id so they do not clutter the test output.
        with warnings.catch_warnings() as _:  # NOQA
            warnings.simplefilter("ignore")
            ev1 = Event(resource_id='id1')
            ev2 = Event(resource_id='id1')
            ev3 = Event(resource_id='id2')
        assert ev1 == ev2
        assert ev2 == ev1
        assert ev1 != ev3
        assert ev3 != ev1
        # comparing with other objects fails
        assert ev1 != 1
        assert ev2 != "id1"

    def test_clear_method_resets_objects(self):
        """
        Tests that the clear() method properly resets all objects. Test for
        #449.
        """
        # Test with basic event object.
        e = Event(force_resource_id=False)
        e.comments.append(Comment(text="test"))
        e.event_type = "explosion"
        assert len(e.comments) == 1
        assert e.event_type == "explosion"
        e.clear()
        assert e == Event(force_resource_id=False)
        assert len(e.comments) == 0
        assert e.event_type is None
        # Test with pick object. Does not really fit in the event test case but
        # it tests the same thing...
        p = Pick()
        p.comments.append(Comment(text="test"))
        p.phase_hint = "p"
        assert len(p.comments) == 1
        assert p.phase_hint == "p"
        # Add some more random attributes. These should disappear upon
        # cleaning.
        with WarningsCapture() as w:
            p.test_1 = "a"
            p.test_2 = "b"
            # two warnings should have been issued by setting non-default keys
            assert len(w) == 2
        assert p.test_1 == "a"
        assert p.test_2 == "b"
        p.clear()
        assert len(p.comments) == 0
        assert p.phase_hint is None
        assert not hasattr(p, "test_1")
        assert not hasattr(p, "test_2")

    @pytest.mark.skipif(not (CARTOPY_VERSION and CARTOPY_VERSION >=
                             [0, 12, 0]),
                        reason='cartopy not installed')
    def test_plot_farfield_without_quiver_with_maps(self, image_path):
        """
        Tests to plot P/S wave farfield radiation pattern, also with beachball
        and some map plots.
        """
        ev = read_events("/path/to/CMTSOLUTION", format="CMTSOLUTION")[0]
        ev.plot(kind=[['global'], ['ortho', 'beachball'],
                      ['p_sphere', 's_sphere']], outfile=image_path)

    def test_farfield_2xn_input(self):
        """
        Tests to compute P/S wave farfield radiation pattern using (theta,phi)
        pairs as input
        """
        # Peru 2001/6/23 20:34:23:
        #  RTP system: [2.245, -0.547, -1.698, 1.339, -3.728, 1.444]
        #  NED system: [-0.547, -1.698, 2.245, -1.444, 1.339, 3.728]
        mt = [-0.547, -1.698, 2.245, -1.444, 1.339, 3.728]
        theta = np.arange(0, 360, 60)
        phi = np.zeros(len(theta))
        rays = np.array([theta, phi]) * np.pi / 180.0
        result = farfield(mt, rays, 'P')
        ref = np.array([[0., 1.13501984, -0.873480164, 2.749332e-16,
                        -1.13501984, 0.873480164], [0, 0, -0, 0, -0, 0],
                        [2.245, 0.655304008, 0.504304008, -2.245,
                         -0.655304008, -0.504304008]])
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-8)


class TestOrigin:
    """
    Test suite for obspy.core.event.Origin
    """
    def test_creation_info(self):
        # 1 - empty Origin class will set creation_info to None
        orig = Origin()
        assert orig.creation_info is None
        # 2 - preset via dict or existing CreationInfo object
        orig = Origin(creation_info={})
        assert isinstance(orig.creation_info, CreationInfo)
        orig = Origin(creation_info=CreationInfo(author='test2'))
        assert isinstance(orig.creation_info, CreationInfo)
        assert orig.creation_info.author == 'test2'
        # 3 - check set values
        orig = Origin(creation_info={'author': 'test'})
        assert orig.creation_info == orig['creation_info']
        assert orig.creation_info.author == 'test'
        assert orig['creation_info']['author'] == 'test'
        orig.creation_info.agency_id = "muh"
        assert orig.creation_info == orig['creation_info']
        assert orig.creation_info.agency_id == 'muh'
        assert orig['creation_info']['agency_id'] == 'muh'

    def test_multiple_origins(self):
        """
        Parameters of multiple origins should not interfere with each other.
        """
        origin = Origin()
        origin.resource_id = 'smi:ch.ethz.sed/origin/37465'
        origin.time = UTCDateTime(0)
        origin.latitude = 12
        origin.latitude_errors.confidence_level = 95
        origin.longitude = 42
        origin.depth_type = 'from location'
        assert origin.resource_id == \
            ResourceIdentifier(id='smi:ch.ethz.sed/origin/37465')
        assert origin.latitude == 12
        assert origin.latitude_errors.confidence_level == 95
        assert origin.latitude_errors.uncertainty is None
        assert origin.longitude == 42
        origin2 = Origin(force_resource_id=False)
        origin2.latitude = 13.4
        assert origin2.depth_type is None
        assert origin2.resource_id is None
        assert origin2.latitude == 13.4
        assert origin2.latitude_errors.confidence_level is None
        assert origin2.longitude is None


class TestCatalog:
    """
    Test suite for obspy.core.event.Catalog
    """
    path = os.path.join(os.path.dirname(__file__), 'data')
    image_dir = os.path.join(os.path.dirname(__file__), 'images')
    iris_xml = os.path.join(path, 'iris_events.xml')
    neries_xml = os.path.join(path, 'neries_events.xml')

    def test_read_invalid_filename(self):
        """
        Tests that we get a sane error message when calling read_events()
        with a filename that doesn't exist
        """
        doesnt_exist = 'dsfhjkfs'
        for i in range(10):
            if os.path.exists(doesnt_exist):
                doesnt_exist += doesnt_exist
                continue
            break
        else:
            self.fail('unable to get invalid file path')

        exception_msg = "No such file or directory"

        formats = _get_entry_points(
            'obspy.plugin.catalog', 'readFormat').keys()
        # try read_inventory() with invalid filename for all registered read
        # plugins and also for filetype autodiscovery
        formats = [None] + list(formats)
        for format in formats:
            with pytest.raises(FileNotFoundError, match=exception_msg):
                read_events(doesnt_exist, format=format)

    def test_creation_info(self):
        cat = Catalog()
        cat.creation_info = CreationInfo(author='test2')
        assert isinstance(cat.creation_info, CreationInfo)
        assert cat.creation_info.author == 'test2'

    def test_read_events_without_parameters(self):
        """
        Calling read_events w/o any parameter will create an example catalog.
        """
        catalog = read_events()
        assert len(catalog) == 3

    def test_str(self):
        """
        Testing the __str__ method of the Catalog object.
        """
        catalog = read_events()
        assert catalog.__str__().startswith("3 Event(s) in Catalog:")
        assert catalog.__str__().endswith(
            "37.736 | 3.0  ML | manual")

    def test_read_events(self):
        """
        Tests the read_events() function using entry points.
        """
        # iris
        catalog = read_events(self.iris_xml)
        assert len(catalog) == 2
        assert catalog[0]._format == 'QUAKEML'
        assert catalog[1]._format == 'QUAKEML'
        # neries
        catalog = read_events(self.neries_xml)
        assert len(catalog) == 3
        assert catalog[0]._format == 'QUAKEML'
        assert catalog[1]._format == 'QUAKEML'
        assert catalog[2]._format == 'QUAKEML'

    def test_read_events_with_wildcard(self):
        """
        Tests the read_events() function with a filename wild card.
        """
        # without wildcard..
        expected = read_events(self.iris_xml)
        expected += read_events(self.neries_xml)
        # with wildcard
        got = read_events(os.path.join(self.path, "*_events.xml"))
        assert expected == got

    def test_append(self):
        """
        Tests the append method of the Catalog object.
        """
        # 1 - create catalog and add a few events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        assert len(catalog) == 0
        catalog.append(event1)
        assert len(catalog) == 1
        assert catalog.events == [event1]
        catalog.append(event2)
        assert len(catalog) == 2
        assert catalog.events == [event1, event2]
        # 2 - adding objects other as Event should fails
        with pytest.raises(TypeError):
            catalog.append(str)
        with pytest.raises(TypeError):
            catalog.append(Catalog)
        with pytest.raises(TypeError):
            catalog.append([event1])

    def test_extend(self):
        """
        Tests the extend method of the Catalog object.
        """
        # 1 - create catalog and extend it with list of events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        assert len(catalog) == 0
        catalog.extend([event1, event2])
        assert len(catalog) == 2
        assert catalog.events == [event1, event2]
        # 2 - extend it with other catalog
        event3 = Event()
        event4 = Event()
        catalog2 = Catalog([event3, event4])
        assert len(catalog) == 2
        catalog.extend(catalog2)
        assert len(catalog) == 4
        assert catalog.events == [event1, event2, event3, event4]
        # adding objects other as Catalog or list should fails
        with pytest.raises(TypeError):
            catalog.extend(str)
        with pytest.raises(TypeError):
            catalog.extend(event1)
        with pytest.raises(TypeError):
            catalog.extend((event1, event2))

    def test_iadd(self):
        """
        Tests the __iadd__ method of the Catalog object.
        """
        # 1 - create catalog and add it with another catalog
        event1 = Event()
        event2 = Event()
        event3 = Event()
        catalog = Catalog([event1])
        catalog2 = Catalog([event2, event3])
        assert len(catalog) == 1
        catalog += catalog2
        assert len(catalog) == 3
        assert catalog.events == [event1, event2, event3]
        # 3 - extend it with another Event
        event4 = Event()
        assert len(catalog) == 3
        catalog += event4
        assert len(catalog) == 4
        assert catalog.events == [event1, event2, event3, event4]
        # adding objects other as Catalog or Event should fails
        with pytest.raises(TypeError):
            catalog.__iadd__(str)
        with pytest.raises(TypeError):
            catalog.__iadd__((event1, event2))
        with pytest.raises(TypeError):
            catalog.__iadd__([event1, event2])

    def test_count_and_len(self):
        """
        Tests the count and __len__ methods of the Catalog object.
        """
        # empty catalog without events
        catalog = Catalog()
        assert len(catalog) == 0
        assert catalog.count() == 0
        # catalog with events
        catalog = read_events()
        assert len(catalog) == 3
        assert catalog.count() == 3

    def test_get_item(self):
        """
        Tests the __getitem__ method of the Catalog object.
        """
        catalog = read_events()
        assert catalog[0] == catalog.events[0]
        assert catalog[-1] == catalog.events[-1]
        assert catalog[2] == catalog.events[2]
        # out of index should fail
        with pytest.raises(IndexError):
            catalog.__getitem__(3)
        with pytest.raises(IndexError):
            catalog.__getitem__(-99)

    def test_slicing(self):
        """
        Tests the __getslice__ method of the Catalog object.
        """
        catalog = read_events()
        assert catalog[0:] == catalog[0:]
        assert catalog[:2] == catalog[:2]
        assert catalog[:] == catalog[:]
        assert len(catalog) == 3
        new_catalog = catalog[1:3]
        assert isinstance(new_catalog, Catalog)
        assert len(new_catalog) == 2

    def test_slicing_with_step(self):
        """
        Tests the __getslice__ method of the Catalog object with step.
        """
        ev1 = Event()
        ev2 = Event()
        ev3 = Event()
        ev4 = Event()
        ev5 = Event()
        catalog = Catalog([ev1, ev2, ev3, ev4, ev5])
        assert catalog[0:6].events == [ev1, ev2, ev3, ev4, ev5]
        assert catalog[0:6:1].events == [ev1, ev2, ev3, ev4, ev5]
        assert catalog[0:6:2].events == [ev1, ev3, ev5]
        assert catalog[1:6:2].events == [ev2, ev4]
        assert catalog[1:6:6].events == [ev2]

    def test_copy(self):
        """
        Testing the copy method of the Catalog object.
        """
        cat = read_events()
        cat2 = cat.copy()
        assert cat == cat2
        assert cat2 == cat
        assert not (cat is cat2)
        assert not (cat2 is cat)
        assert cat.events[0] == cat2.events[0]
        assert not (cat.events[0] is cat2.events[0])

    def test_filter(self):
        """
        Testing the filter method of the Catalog object.
        """
        def getattrs(event, attr):
            if attr == 'magnitude':
                obj = event.magnitudes[0]
                attr = 'mag'
            else:
                obj = event.origins[0]
            for a in attr.split('.'):
                obj = getattr(obj, a)
            return obj
        cat = read_events()
        assert all(event.magnitudes[0].mag < 4.
                   for event in cat.filter('magnitude < 4.'))
        attrs = ('magnitude', 'latitude', 'longitude', 'depth', 'time',
                 'quality.standard_error', 'quality.azimuthal_gap',
                 'quality.used_station_count', 'quality.used_phase_count')
        values = (4., 40., 50., 10., UTCDateTime('2012-04-04 14:20:00'),
                  1., 50, 40, 20)
        for attr, value in zip(attrs, values):
            attr_filter = attr.split('.')[-1]
            cat_smaller = cat.filter('%s < %s' % (attr_filter, value))
            cat_bigger = cat.filter('%s >= %s' % (attr_filter, value))
            assert all(True if a is None else a < value
                       for event in cat_smaller
                       for a in [getattrs(event, attr)])
            assert all(False if a is None else a >= value
                       for event in cat_bigger
                       for a in [getattrs(event, attr)])
            assert all(event in cat for event in (cat_smaller + cat_bigger))
            cat_smaller_inverse = cat.filter(
                '%s < %s' % (attr_filter, value), inverse=True)
            assert all(event in cat_bigger for event in cat_smaller_inverse)
            cat_bigger_inverse = cat.filter(
                '%s >= %s' % (attr_filter, value), inverse=True)
            assert all(event in cat_smaller for event in cat_bigger_inverse)

    def test_catalog_resource_id(self):
        """
        See #662
        """
        cat = read_events(self.neries_xml)
        assert str(cat.resource_id) == r"smi://eu.emsc/unid"

    def test_can_pickle(self):
        """
        Ensure a catalog can be pickled and unpickled and that the results are
        equal.
        """
        cat = read_events()
        cat_bytes = pickle.dumps(cat)
        cat2 = pickle.loads(cat_bytes)
        assert cat == cat2

    def test_issue_2173(self):
        """
        Ensure events with empty origins are equal after round-trip to disk.
        See #2173.
        """
        # create event and save to disk
        origin = Origin(time=UTCDateTime('2016-01-01'))
        event1 = Event(origins=[origin])
        bio = io.BytesIO()
        event1.write(bio, 'quakeml')
        # rewind bytes stream
        bio.seek(0)
        # read from disk
        event2 = read_events(bio)[0]
        # saved and loaded event should be equal
        assert event1 == event2

    def test_read_path(self):
        """
        Ensure read_events works with pathlib.Path objects.
        """
        path = Path(self.iris_xml)
        cat = read_events(path)
        assert cat == read_events(self.iris_xml)


@pytest.mark.skipif(not HAS_CARTOPY,
                    reason='Cartopy not installed or too old')
class CatalogCartopyTestCase:
    """
    Test suite for obspy.core.event.Catalog.plot using Cartopy
    """
    def test_catalog_plot_global(self, image_path):
        """
        Tests the catalog preview plot, default parameters, using Cartopy.
        """
        cat = read_events()
        cat.plot(method='cartopy', outfile=image_path)

    def test_catalog_plot_ortho(self, image_path):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Cartopy.
        """
        cat = read_events()
        cat.plot(method='cartopy', outfile=image_path, projection='ortho',
                 resolution='c', water_fill_color='#98b7e2', label=None,
                 color='date')

    def test_catalog_plot_ortho_longitude_wrap(self, image_path):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Cartopy, with longitudes that need the mean to be
        computed in a circular fashion.
        """
        cat = read_events('/path/to/events_longitude_wrap.zmap', format='ZMAP')
        cat.plot(method='cartopy', outfile=image_path, projection='ortho',
                 resolution='c', label=None, title='', colorbar=False,
                 water_fill_color='b')

    def test_catalog_plot_local(self, image_path):
        """
        Tests the catalog preview plot, local projection, some more non-default
        parameters, using Cartopy.
        """
        cat = read_events()
        cat.plot(method='cartopy', outfile=image_path, projection='local',
                 resolution='50m', continent_fill_color='0.3',
                 color='date', colormap='gist_heat')

    def test_plot_catalog_before_1900(self):
        """
        Tests plotting events with origin times before 1900
        """
        cat = read_events()
        cat[1].origins[0].time = UTCDateTime(813, 2, 4, 14, 13)

        # just checking this runs without error is fine, no need to check
        # content
        with MatplotlibBackend("AGG", sloppy=True):
            cat.plot(outfile=io.BytesIO())
            # also test with just a single event
            cat.events = [cat[1]]
            cat.plot(outfile=io.BytesIO())


class TestWaveformStreamID:
    """
    Test suite for obspy.core.event.WaveformStreamID.
    """
    def test_initialization(self):
        """
        Test the different initialization methods.
        """
        # Default init.
        waveform_id = WaveformStreamID()
        assert waveform_id.network_code is None
        assert waveform_id.station_code is None
        assert waveform_id.location_code is None
        assert waveform_id.channel_code is None
        # With seed string.
        waveform_id = WaveformStreamID(seed_string="BW.FUR.01.EHZ")
        assert waveform_id.network_code == "BW"
        assert waveform_id.station_code == "FUR"
        assert waveform_id.location_code == "01"
        assert waveform_id.channel_code == "EHZ"
        # As soon as any other argument is set, the seed_string will not be
        # used and the default values will be used for any unset arguments.
        waveform_id = WaveformStreamID(location_code="02",
                                       seed_string="BW.FUR.01.EHZ")
        assert waveform_id.network_code is None
        assert waveform_id.station_code is None
        assert waveform_id.location_code == "02"
        assert waveform_id.channel_code is None

    def test_initialization_with_invalid_seed_string(self):
        """
        Test initialization with an invalid seed string. Should raise a
        warning.
        """
        # An invalid SEED string will issue a warning and fill the object with
        # the default values.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            with pytest.raises(UserWarning):
                WaveformStreamID(seed_string="Invalid SEED string")
            # Now ignore the warnings and test the default values.
            warnings.simplefilter('ignore', UserWarning)
            waveform_id = WaveformStreamID(seed_string="Invalid Seed String")
            assert waveform_id.network_code is None
            assert waveform_id.station_code is None
            assert waveform_id.location_code is None
            assert waveform_id.channel_code is None

    def test_id_property(self):
        """
        Enure the `id` property of WaveformStreamID returns the same as
        `get_seed_string`"
        """
        waveform_id = WaveformStreamID(seed_string="BW.FUR.01.EHZ")
        assert waveform_id.id == waveform_id.get_seed_string()


class TestBase:
    """
    Test suite for obspy.core.event.base.
    """
    def test_quantity_error_warn_on_non_default_key(self):
        """
        """
        err = QuantityError()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            err.uncertainty = 0.01
            err.lower_uncertainty = 0.1
            err.upper_uncertainty = 0.02
            err.confidence_level = 80
            assert len(w) == 0
            # setting a typoed or custom field should warn!
            err.confidence_levle = 80
            assert len(w) == 1

    def test_quantity_error_equality(self):
        """
        Comparisons between empty quantity errors and None should return True.
        Non-empty quantity errors should return False.
        """
        err1 = QuantityError()
        assert err1 == None  # NOQA needs to be ==
        err2 = QuantityError(uncertainty=10)
        assert err2 is not None
        assert err2 != err1
        err3 = QuantityError(uncertainty=10)
        assert err3 == err2

    def test_event_type_objects_warn_on_non_default_key(self):
        """
        """
        for cls in (Event, Origin, Pick, Magnitude, FocalMechanism):
            obj = cls()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # setting a typoed or custom field should warn!
                obj.some_custom_non_default_crazy_key = "my_text_here"
                assert len(w) == 1

    def test_setting_nans_or_inf_fails(self):
        """
        Tests that settings NaNs or infs as floating point values fails.
        """
        o = Origin()

        with pytest.raises(ValueError, match='is not a finite'):
            o.latitude = float('nan')

        with pytest.raises(ValueError, match='is not a finite'):
            o.latitude = float('inf')

        with pytest.raises(ValueError, match='is not a finite'):
            o.latitude = float('-inf')

    def test_issue3105(self):
        evs = read_events()
        evs[0].magnitudes[0].mag = 0
        assert len(evs) == 3
        assert len(evs.filter('magnitude < 3.5')) == 2
