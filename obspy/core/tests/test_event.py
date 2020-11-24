# -*- coding: utf-8 -*-
import io
import os
import pickle
import unittest
import warnings
from pathlib import Path

from matplotlib import rcParams
import numpy as np

from obspy import UTCDateTime, read_events
from obspy.core.event import (Catalog, Comment, CreationInfo, Event,
                              FocalMechanism, Magnitude, Origin, Pick,
                              ResourceIdentifier, WaveformStreamID)
from obspy.core.event.source import farfield
from obspy.core.util import (
    BASEMAP_VERSION, CARTOPY_VERSION, PROJ4_VERSION, MATPLOTLIB_VERSION)
from obspy.core.util.base import _get_entry_points
from obspy.core.util.misc import MatplotlibBackend
from obspy.core.util.testing import ImageComparison
from obspy.core.event.base import QuantityError


if CARTOPY_VERSION and CARTOPY_VERSION >= [0, 12, 0]:
    HAS_CARTOPY = True
else:
    HAS_CARTOPY = False


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Event
    """
    def setUp(self):
        """
        Setup code to run before each test. Temporary replaces the state on
        the ResourceIdentifier class level to reset the ResourceID mechanisms
        before each run.
        """
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')

    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = read_events()[1]
        s = event.short_str()
        self.assertEqual("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" +
                         " | 4.3  ML | manual", s)

    def test_str_empty_origin(self):
        """
        Ensure an event with an empty origin returns a str without raising a
        TypeError (#2119).
        """
        event = Event(origins=[Origin()])
        out = event.short_str()
        self.assertIsInstance(out, str)
        self.assertEqual(out, 'None | None, None')

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
        self.assertEqual(ev1, ev2)
        self.assertEqual(ev2, ev1)
        self.assertFalse(ev1 == ev3)
        self.assertFalse(ev3 == ev1)
        # comparing with other objects fails
        self.assertFalse(ev1 == 1)
        self.assertFalse(ev2 == "id1")

    def test_clear_method_resets_objects(self):
        """
        Tests that the clear() method properly resets all objects. Test for
        #449.
        """
        # Test with basic event object.
        e = Event(force_resource_id=False)
        e.comments.append(Comment(text="test"))
        e.event_type = "explosion"
        self.assertEqual(len(e.comments), 1)
        self.assertEqual(e.event_type, "explosion")
        e.clear()
        self.assertEqual(e, Event(force_resource_id=False))
        self.assertEqual(len(e.comments), 0)
        self.assertEqual(e.event_type, None)
        # Test with pick object. Does not really fit in the event test case but
        # it tests the same thing...
        p = Pick()
        p.comments.append(Comment(text="test"))
        p.phase_hint = "p"
        self.assertEqual(len(p.comments), 1)
        self.assertEqual(p.phase_hint, "p")
        # Add some more random attributes. These should disappear upon
        # cleaning.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p.test_1 = "a"
            p.test_2 = "b"
            # two warnings should have been issued by setting non-default keys
            self.assertEqual(len(w), 2)
        self.assertEqual(p.test_1, "a")
        self.assertEqual(p.test_2, "b")
        p.clear()
        self.assertEqual(len(p.comments), 0)
        self.assertEqual(p.phase_hint, None)
        self.assertFalse(hasattr(p, "test_1"))
        self.assertFalse(hasattr(p, "test_2"))

    @unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
    @unittest.skipIf(
        BASEMAP_VERSION or [] >= [1, 1, 0] and MATPLOTLIB_VERSION == [3, 0, 1],
        'matplotlib 3.0.1 is not compatible with basemap')
    @unittest.skipIf(PROJ4_VERSION and PROJ4_VERSION[0] == 5,
                     'unsupported proj4 library')
    def test_plot_farfield_without_quiver_with_maps(self):
        """
        Tests to plot P/S wave farfield radiation pattern, also with beachball
        and some map plots.
        """
        ev = read_events("/path/to/CMTSOLUTION", format="CMTSOLUTION")[0]
        with ImageComparison(self.image_dir, 'event.png') as ic:
            ev.plot(kind=[['global'], ['ortho', 'beachball'],
                          ['p_sphere', 's_sphere']], outfile=ic.name)

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


class OriginTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Origin
    """
    def test_creation_info(self):
        # 1 - empty Origin class will set creation_info to None
        orig = Origin()
        self.assertEqual(orig.creation_info, None)
        # 2 - preset via dict or existing CreationInfo object
        orig = Origin(creation_info={})
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        orig = Origin(creation_info=CreationInfo(author='test2'))
        self.assertTrue(isinstance(orig.creation_info, CreationInfo))
        self.assertEqual(orig.creation_info.author, 'test2')
        # 3 - check set values
        orig = Origin(creation_info={'author': 'test'})
        self.assertEqual(orig.creation_info, orig['creation_info'])
        self.assertEqual(orig.creation_info.author, 'test')
        self.assertEqual(orig['creation_info']['author'], 'test')
        orig.creation_info.agency_id = "muh"
        self.assertEqual(orig.creation_info, orig['creation_info'])
        self.assertEqual(orig.creation_info.agency_id, 'muh')
        self.assertEqual(orig['creation_info']['agency_id'], 'muh')

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
        self.assertEqual(
            origin.resource_id,
            ResourceIdentifier(id='smi:ch.ethz.sed/origin/37465'))
        self.assertEqual(origin.latitude, 12)
        self.assertEqual(origin.latitude_errors.confidence_level, 95)
        self.assertEqual(origin.latitude_errors.uncertainty, None)
        self.assertEqual(origin.longitude, 42)
        origin2 = Origin(force_resource_id=False)
        origin2.latitude = 13.4
        self.assertEqual(origin2.depth_type, None)
        self.assertEqual(origin2.resource_id, None)
        self.assertEqual(origin2.latitude, 13.4)
        self.assertEqual(origin2.latitude_errors.confidence_level, None)
        self.assertEqual(origin2.longitude, None)


class CatalogTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')

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

        exception_msg = "[Errno 2] No such file or directory: '{}'"

        formats = _get_entry_points(
            'obspy.plugin.catalog', 'readFormat').keys()
        # try read_inventory() with invalid filename for all registered read
        # plugins and also for filetype autodiscovery
        formats = [None] + list(formats)
        for format in formats:
            with self.assertRaises(FileNotFoundError) as e:
                read_events(doesnt_exist, format=format)
            self.assertEqual(
                str(e.exception), exception_msg.format(doesnt_exist))

    def test_creation_info(self):
        cat = Catalog()
        cat.creation_info = CreationInfo(author='test2')
        self.assertTrue(isinstance(cat.creation_info, CreationInfo))
        self.assertEqual(cat.creation_info.author, 'test2')

    def test_read_events_without_parameters(self):
        """
        Calling read_events w/o any parameter will create an example catalog.
        """
        catalog = read_events()
        self.assertEqual(len(catalog), 3)

    def test_str(self):
        """
        Testing the __str__ method of the Catalog object.
        """
        catalog = read_events()
        self.assertTrue(catalog.__str__().startswith("3 Event(s) in Catalog:"))
        self.assertTrue(catalog.__str__().endswith(
            "37.736 | 3.0  ML | manual"))

    def test_read_events(self):
        """
        Tests the read_events() function using entry points.
        """
        # iris
        catalog = read_events(self.iris_xml)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0]._format, 'QUAKEML')
        self.assertEqual(catalog[1]._format, 'QUAKEML')
        # neries
        catalog = read_events(self.neries_xml)
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog[0]._format, 'QUAKEML')
        self.assertEqual(catalog[1]._format, 'QUAKEML')
        self.assertEqual(catalog[2]._format, 'QUAKEML')

    def test_read_events_with_wildcard(self):
        """
        Tests the read_events() function with a filename wild card.
        """
        # without wildcard..
        expected = read_events(self.iris_xml)
        expected += read_events(self.neries_xml)
        # with wildcard
        got = read_events(os.path.join(self.path, "*_events.xml"))
        self.assertEqual(expected, got)

    def test_append(self):
        """
        Tests the append method of the Catalog object.
        """
        # 1 - create catalog and add a few events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        self.assertEqual(len(catalog), 0)
        catalog.append(event1)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog.events, [event1])
        catalog.append(event2)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog.events, [event1, event2])
        # 2 - adding objects other as Event should fails
        self.assertRaises(TypeError, catalog.append, str)
        self.assertRaises(TypeError, catalog.append, Catalog)
        self.assertRaises(TypeError, catalog.append, [event1])

    def test_extend(self):
        """
        Tests the extend method of the Catalog object.
        """
        # 1 - create catalog and extend it with list of events
        catalog = Catalog()
        event1 = Event()
        event2 = Event()
        self.assertEqual(len(catalog), 0)
        catalog.extend([event1, event2])
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog.events, [event1, event2])
        # 2 - extend it with other catalog
        event3 = Event()
        event4 = Event()
        catalog2 = Catalog([event3, event4])
        self.assertEqual(len(catalog), 2)
        catalog.extend(catalog2)
        self.assertEqual(len(catalog), 4)
        self.assertEqual(catalog.events, [event1, event2, event3, event4])
        # adding objects other as Catalog or list should fails
        self.assertRaises(TypeError, catalog.extend, str)
        self.assertRaises(TypeError, catalog.extend, event1)
        self.assertRaises(TypeError, catalog.extend, (event1, event2))

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
        self.assertEqual(len(catalog), 1)
        catalog += catalog2
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog.events, [event1, event2, event3])
        # 3 - extend it with another Event
        event4 = Event()
        self.assertEqual(len(catalog), 3)
        catalog += event4
        self.assertEqual(len(catalog), 4)
        self.assertEqual(catalog.events, [event1, event2, event3, event4])
        # adding objects other as Catalog or Event should fails
        self.assertRaises(TypeError, catalog.__iadd__, str)
        self.assertRaises(TypeError, catalog.__iadd__, (event1, event2))
        self.assertRaises(TypeError, catalog.__iadd__, [event1, event2])

    def test_count_and_len(self):
        """
        Tests the count and __len__ methods of the Catalog object.
        """
        # empty catalog without events
        catalog = Catalog()
        self.assertEqual(len(catalog), 0)
        self.assertEqual(catalog.count(), 0)
        # catalog with events
        catalog = read_events()
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog.count(), 3)

    def test_get_item(self):
        """
        Tests the __getitem__ method of the Catalog object.
        """
        catalog = read_events()
        self.assertEqual(catalog[0], catalog.events[0])
        self.assertEqual(catalog[-1], catalog.events[-1])
        self.assertEqual(catalog[2], catalog.events[2])
        # out of index should fail
        self.assertRaises(IndexError, catalog.__getitem__, 3)
        self.assertRaises(IndexError, catalog.__getitem__, -99)

    def test_slicing(self):
        """
        Tests the __getslice__ method of the Catalog object.
        """
        catalog = read_events()
        self.assertEqual(catalog[0:], catalog[0:])
        self.assertEqual(catalog[:2], catalog[:2])
        self.assertEqual(catalog[:], catalog[:])
        self.assertEqual(len(catalog), 3)
        new_catalog = catalog[1:3]
        self.assertTrue(isinstance(new_catalog, Catalog))
        self.assertEqual(len(new_catalog), 2)

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
        self.assertEqual(catalog[0:6].events, [ev1, ev2, ev3, ev4, ev5])
        self.assertEqual(catalog[0:6:1].events, [ev1, ev2, ev3, ev4, ev5])
        self.assertEqual(catalog[0:6:2].events, [ev1, ev3, ev5])
        self.assertEqual(catalog[1:6:2].events, [ev2, ev4])
        self.assertEqual(catalog[1:6:6].events, [ev2])

    def test_copy(self):
        """
        Testing the copy method of the Catalog object.
        """
        cat = read_events()
        cat2 = cat.copy()
        self.assertEqual(cat, cat2)
        self.assertEqual(cat2, cat)
        self.assertFalse(cat is cat2)
        self.assertFalse(cat2 is cat)
        self.assertEqual(cat.events[0], cat2.events[0])
        self.assertFalse(cat.events[0] is cat2.events[0])

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
        self.assertTrue(all(event.magnitudes[0].mag < 4.
                            for event in cat.filter('magnitude < 4.')))
        attrs = ('magnitude', 'latitude', 'longitude', 'depth', 'time',
                 'quality.standard_error', 'quality.azimuthal_gap',
                 'quality.used_station_count', 'quality.used_phase_count')
        values = (4., 40., 50., 10., UTCDateTime('2012-04-04 14:20:00'),
                  1., 50, 40, 20)
        for attr, value in zip(attrs, values):
            attr_filter = attr.split('.')[-1]
            cat_smaller = cat.filter('%s < %s' % (attr_filter, value))
            cat_bigger = cat.filter('%s >= %s' % (attr_filter, value))
            self.assertTrue(all(True if a is None else a < value
                                for event in cat_smaller
                                for a in [getattrs(event, attr)]))
            self.assertTrue(all(False if a is None else a >= value
                                for event in cat_bigger
                                for a in [getattrs(event, attr)]))
            self.assertTrue(all(event in cat
                                for event in (cat_smaller + cat_bigger)))
            cat_smaller_inverse = cat.filter(
                '%s < %s' % (attr_filter, value), inverse=True)
            self.assertTrue(all(event in cat_bigger
                                for event in cat_smaller_inverse))
            cat_bigger_inverse = cat.filter(
                '%s >= %s' % (attr_filter, value), inverse=True)
            self.assertTrue(all(event in cat_smaller
                                for event in cat_bigger_inverse))

    def test_catalog_resource_id(self):
        """
        See #662
        """
        cat = read_events(self.neries_xml)
        self.assertEqual(str(cat.resource_id), r"smi://eu.emsc/unid")

    def test_can_pickle(self):
        """
        Ensure a catalog can be pickled and unpickled and that the results are
        equal.
        """
        cat = read_events()
        cat_bytes = pickle.dumps(cat)
        cat2 = pickle.loads(cat_bytes)
        self.assertEqual(cat, cat2)

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
        # read from disk
        event2 = read_events(bio)[0]
        # saved and loaded event should be equal
        self.assertEqual(event1, event2)

    def test_read_path(self):
        """
        Ensure read_events works with pathlib.Path objects.
        """
        path = Path(self.iris_xml)
        cat = read_events(path)
        self.assertEqual(cat, read_events(self.iris_xml))


@unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
@unittest.skipIf(
    BASEMAP_VERSION or [] >= [1, 1, 0] and MATPLOTLIB_VERSION == [3, 0, 1],
    'matplotlib 3.0.1 is not compatible with basemap')
class CatalogBasemapTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog.plot with Basemap
    """
    def setUp(self):
        # directory where the test files are located
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')

    @unittest.skipIf(PROJ4_VERSION and PROJ4_VERSION[0] == 5,
                     'unsupported proj4 library')
    def test_catalog_plot_global(self):
        """
        Tests the catalog preview plot, default parameters, using Basemap.
        """
        cat = read_events()
        reltol = 1
        if BASEMAP_VERSION < [1, 0, 7]:
            reltol = 3
        with ImageComparison(self.image_dir, 'catalog-basemap1.png',
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='basemap', outfile=ic.name)

    def test_catalog_plot_ortho(self):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Basemap.
        """
        cat = read_events()
        with ImageComparison(self.image_dir, 'catalog-basemap2.png',
                             reltol=1.3) as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='basemap', outfile=ic.name, projection='ortho',
                     resolution='c', water_fill_color='#98b7e2', label=None,
                     color='date')

    def test_catalog_plot_ortho_longitude_wrap(self):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Basemap, with longitudes that need the mean to be
        computed in a circular fashion.
        """
        cat = read_events('/path/to/events_longitude_wrap.zmap', format='ZMAP')
        with ImageComparison(self.image_dir, 'catalog-basemap_long-wrap.png',
                             reltol=1.1) as ic:
            rcParams['savefig.dpi'] = 40
            cat.plot(method='basemap', outfile=ic.name, projection='ortho',
                     resolution='c', label=None, title='', colorbar=False,
                     water_fill_color='b')

    def test_catalog_plot_local(self):
        """
        Tests the catalog preview plot, local projection, some more non-default
        parameters, using Basemap.
        """
        cat = read_events()
        reltol = 1.5
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100
        with ImageComparison(self.image_dir, "catalog-basemap3.png",
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='basemap', outfile=ic.name, projection='local',
                     resolution='l', continent_fill_color='0.3',
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
            cat.plot(outfile=io.BytesIO(), method='basemap')
            # also test with just a single event
            cat.events = [cat[1]]
            cat.plot(outfile=io.BytesIO(), method='basemap')


@unittest.skipIf(not HAS_CARTOPY, 'Cartopy not installed or too old')
class CatalogCartopyTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog.plot using Cartopy
    """
    def setUp(self):
        # directory where the test files are located
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')

    def test_catalog_plot_global(self):
        """
        Tests the catalog preview plot, default parameters, using Cartopy.
        """
        cat = read_events()
        with ImageComparison(self.image_dir, 'catalog-cartopy1.png') as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='cartopy', outfile=ic.name)

    def test_catalog_plot_ortho(self):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Cartopy.
        """
        cat = read_events()
        with ImageComparison(self.image_dir, 'catalog-cartopy2.png') as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='cartopy', outfile=ic.name, projection='ortho',
                     resolution='c', water_fill_color='#98b7e2', label=None,
                     color='date')

    def test_catalog_plot_ortho_longitude_wrap(self):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters, using Cartopy, with longitudes that need the mean to be
        computed in a circular fashion.
        """
        cat = read_events('/path/to/events_longitude_wrap.zmap', format='ZMAP')
        with ImageComparison(self.image_dir,
                             'catalog-cartopy_long-wrap.png') as ic:
            rcParams['savefig.dpi'] = 40
            cat.plot(method='cartopy', outfile=ic.name, projection='ortho',
                     resolution='c', label=None, title='', colorbar=False,
                     water_fill_color='b')

    def test_catalog_plot_local(self):
        """
        Tests the catalog preview plot, local projection, some more non-default
        parameters, using Cartopy.
        """
        cat = read_events()
        with ImageComparison(self.image_dir, 'catalog-cartopy3.png') as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(method='cartopy', outfile=ic.name, projection='local',
                     resolution='50m', continent_fill_color='0.3',
                     color='date', colormap='gist_heat')


class WaveformStreamIDTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.WaveformStreamID.
    """
    def test_initialization(self):
        """
        Test the different initialization methods.
        """
        # Default init.
        waveform_id = WaveformStreamID()
        self.assertEqual(waveform_id.network_code, None)
        self.assertEqual(waveform_id.station_code, None)
        self.assertEqual(waveform_id.location_code, None)
        self.assertEqual(waveform_id.channel_code, None)
        # With seed string.
        waveform_id = WaveformStreamID(seed_string="BW.FUR.01.EHZ")
        self.assertEqual(waveform_id.network_code, "BW")
        self.assertEqual(waveform_id.station_code, "FUR")
        self.assertEqual(waveform_id.location_code, "01")
        self.assertEqual(waveform_id.channel_code, "EHZ")
        # As soon as any other argument is set, the seed_string will not be
        # used and the default values will be used for any unset arguments.
        waveform_id = WaveformStreamID(location_code="02",
                                       seed_string="BW.FUR.01.EHZ")
        self.assertEqual(waveform_id.network_code, None)
        self.assertEqual(waveform_id.station_code, None)
        self.assertEqual(waveform_id.location_code, "02")
        self.assertEqual(waveform_id.channel_code, None)

    def test_initialization_with_invalid_seed_string(self):
        """
        Test initialization with an invalid seed string. Should raise a
        warning.
        """
        # An invalid SEED string will issue a warning and fill the object with
        # the default values.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, WaveformStreamID,
                              seed_string="Invalid SEED string")
            # Now ignore the warnings and test the default values.
            warnings.simplefilter('ignore', UserWarning)
            waveform_id = WaveformStreamID(seed_string="Invalid Seed String")
            self.assertEqual(waveform_id.network_code, None)
            self.assertEqual(waveform_id.station_code, None)
            self.assertEqual(waveform_id.location_code, None)
            self.assertEqual(waveform_id.channel_code, None)

    def test_id_property(self):
        """
        Enure the `id` property of WaveformStreamID returns the same as
        `get_seed_string`"
        """
        waveform_id = WaveformStreamID(seed_string="BW.FUR.01.EHZ")
        self.assertEqual(waveform_id.id, waveform_id.get_seed_string())


class BaseTestCase(unittest.TestCase):
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
            self.assertEqual(len(w), 0)
            # setting a typoed or custom field should warn!
            err.confidence_levle = 80
            self.assertEqual(len(w), 1)

    def test_quantity_error_equality(self):
        """
        Comparisons between empty quantity errors and None should return True.
        Non-empty quantity errors should return False.
        """
        err1 = QuantityError()
        self.assertEqual(err1, None)
        err2 = QuantityError(uncertainty=10)
        self.assertNotEqual(err2, None)
        self.assertNotEqual(err2, err1)
        err3 = QuantityError(uncertainty=10)
        self.assertEqual(err3, err2)

    def test_event_type_objects_warn_on_non_default_key(self):
        """
        """
        for cls in (Event, Origin, Pick, Magnitude, FocalMechanism):
            obj = cls()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # setting a typoed or custom field should warn!
                obj.some_custom_non_default_crazy_key = "my_text_here"
                self.assertEqual(len(w), 1)

    def test_setting_nans_or_inf_fails(self):
        """
        Tests that settings NaNs or infs as floating point values fails.
        """
        o = Origin()

        with self.assertRaises(ValueError) as e:
            o.latitude = float('nan')
        self.assertEqual(
            e.exception.args[0],
            "On Origin object: Value 'nan' for 'latitude' is not a finite "
            "floating point value.")

        with self.assertRaises(ValueError) as e:
            o.latitude = float('inf')
        self.assertEqual(
            e.exception.args[0],
            "On Origin object: Value 'inf' for 'latitude' is not a finite "
            "floating point value.")

        with self.assertRaises(ValueError) as e:
            o.latitude = float('-inf')
        self.assertEqual(
            e.exception.args[0],
            "On Origin object: Value '-inf' for 'latitude' is "
            "not a finite floating point value.")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogTestCase, 'test'))
    suite.addTest(unittest.makeSuite(CatalogBasemapTestCase, 'test'))
    suite.addTest(unittest.makeSuite(CatalogCartopyTestCase, 'test'))
    suite.addTest(unittest.makeSuite(EventTestCase, 'test'))
    suite.addTest(unittest.makeSuite(OriginTestCase, 'test'))
    suite.addTest(unittest.makeSuite(WaveformStreamIDTestCase, 'test'))
    suite.addTest(unittest.makeSuite(BaseTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
