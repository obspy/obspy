# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import copy
import os
import sys
import unittest
import warnings
import tempfile

from matplotlib import rcParams
import numpy as np

from obspy.core.event import (Catalog, Comment, CreationInfo, Event, Origin,
                              Pick, ResourceIdentifier, WaveformStreamID,
                              read_events, Magnitude, FocalMechanism, Arrival)
from obspy.core.event.source import farfield
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import BASEMAP_VERSION, CARTOPY_VERSION
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
        # directory where the test files are located
        path = os.path.join(os.path.dirname(__file__), 'data')
        self.path = path
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = read_events()[1]
        s = event.short_str()
        self.assertEqual("2012-04-04T14:18:37.000000Z | +39.342,  +41.044" +
                         " | 4.3 ML | manual", s)

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

    def test_event_copying_does_not_raise_duplicate_resource_id_warning(self):
        """
        Tests that copying an event does not raise a duplicate resource id
        warning.
        """
        ev = read_events()[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev2 = copy.copy(ev)
            self.assertEqual(len(w), 0)
            ev3 = copy.deepcopy(ev)
            self.assertEqual(len(w), 0)
        # The two events should compare equal.
        self.assertEqual(ev, ev2)
        self.assertEqual(ev, ev3)
        # get resource_ids and referred objects from each of the events
        rid1 = ev.resource_id
        rid2 = ev2.resource_id
        rid3 = ev3.resource_id
        rob1 = rid1.get_referred_object()
        rob2 = rid2.get_referred_object()
        rob3 = rid3.get_referred_object()
        # A shallow copy should just use the exact same resource identifier,
        # while a deep copy should not, although they should be qual.
        self.assertIs(rid1, rid2)
        self.assertIsNot(rid1, rid3)
        self.assertEqual(rid1, rid3)
        # make sure the object_id on the resource_ids are not the same
        self.assertEqual(rid1._object_id, rid2._object_id)
        self.assertNotEqual(rid1._object_id, rid3._object_id)
        # copy should point to the same object, deep copy should not
        self.assertIs(rob1, rob2)
        self.assertIsNot(rob1, rob3)
        # although the referred objects should be equal
        self.assertEqual(rob1, rob3)

    @unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
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
    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

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
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

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
        self.assertTrue(catalog.__str__().endswith("37.736 | 3.0 ML | manual"))

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

    def test_latest_in_scope_object_returned(self):
        """
        Test that the most recently defined object with the same resource_id,
        that is still in scope, is returned from the get_referred_object
        method
        """
        cat1 = read_events()
        # The resource_id attached to the first event is self-pointing
        self.assertIs(cat1[0], cat1[0].resource_id.get_referred_object())
        # make a copy and re-read catalog
        cat2 = cat1.copy()
        cat3 = read_events()
        # the resource_id on the new catalogs point to their attached objects
        self.assertIs(cat1[0], cat1[0].resource_id.get_referred_object())
        self.assertIs(cat2[0], cat2[0].resource_id.get_referred_object())
        self.assertIs(cat3[0], cat3[0].resource_id.get_referred_object())
        # now delete cat1 and make sure cat2 and cat3 still work
        del cat1
        self.assertIs(cat2[0], cat2[0].resource_id.get_referred_object())
        self.assertIs(cat3[0], cat3[0].resource_id.get_referred_object())
        # create a resource_id with the same id as the last defined object
        # with the same resource id (that is still in scope) is returned
        new_id = cat2[0].resource_id.id
        rid = ResourceIdentifier(new_id)
        self.assertIs(rid.get_referred_object(), cat3[0])
        del cat3
        # raises UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertIs(rid.get_referred_object(), cat2[0])
        del cat2
        self.assertIs(rid.get_referred_object(), None)


@unittest.skipIf(not BASEMAP_VERSION, 'basemap not installed')
class CatalogBasemapTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog.plot with Basemap
    """
    def setUp(self):
        # directory where the test files are located
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

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


@unittest.skipIf(not HAS_CARTOPY, 'Cartopy not installed or too old')
class CatalogCartopyTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Catalog.plot using Cartopy
    """
    def setUp(self):
        # directory where the test files are located
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

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


class ResourceIdentifierTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.ResourceIdentifier.
    """
    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

    def test_same_resource_id_different_referred_object(self):
        """
        Tests the handling of the case that different ResourceIdentifier
        instances are created that have the same resource id but different
        objects. The referred objects should still return the same objects
        used in the ResourceIdentifier construction or set_referred_object
        call. However, if an object is set to a resource_id that is not
        equal to the last object set it should issue a warning.
        """
        warnings.simplefilter('default')
        object_a = UTCDateTime(1000)
        object_b = UTCDateTime(1000)
        object_c = UTCDateTime(1001)
        self.assertFalse(object_a is object_b)
        id = 'obspy.org/tests/test_resource'
        res_a = ResourceIdentifier(id=id, referred_object=object_a)
        # Now create a new resource with the same id but a different object.
        # This should not raise a warning as the object a and b are equal.
        with warnings.catch_warnings(record=True) as w:
            res_b = ResourceIdentifier(id=id, referred_object=object_b)
            self.assertEqual(len(w), 0)
        # if the set object is not equal to the last object set to the same
        # resource_id, however, a warning should be issued.
        with warnings.catch_warnings(record=True) as w:
            res_c = ResourceIdentifier(id=id, referred_object=object_c)
            self.assertEqual(len(w), 1)
            expected_text = 'which is not equal to the last object bound'
            self.assertIn(expected_text, str(w[0]))
        # even though the resource_id are the same, the referred objects
        # should point to the original (different) objects
        self.assertIs(object_a, res_a.get_referred_object())
        self.assertIs(object_b, res_b.get_referred_object())
        self.assertIs(object_c, res_c.get_referred_object())

    def test_objects_garbage_collection(self):
        """
        Test that the ResourceIdentifier class does not mess with the garbage
        collection of the attached objects.
        """
        object_a = UTCDateTime()
        ref_count = sys.getrefcount(object_a)
        _res_id = ResourceIdentifier(referred_object=object_a)
        self.assertEqual(sys.getrefcount(object_a), ref_count)
        self.assertTrue(bool(_res_id))

    def test_id_without_reference_not_in_global_list(self):
        """
        This tests some internal workings of the ResourceIdentifier class.
        NEVER modify the __resource_id_weak_dict!

        Only those ResourceIdentifiers that have a reference to an object that
        is referred to somewhere else should stay in the dictionary.
        """
        r_dict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        _r1 = ResourceIdentifier()  # NOQA
        self.assertEqual(len(list(r_dict.keys())), 0)
        # Adding a ResourceIdentifier with an object that does not have a
        # reference will result in a dict that contains None, but that will
        # get removed when the resource_id goes out of scope
        _r2 = ResourceIdentifier(referred_object=UTCDateTime())  # NOQA
        # raises UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(_r2.get_referred_object(), None)
        del _r2  # delete rid to get its id out of r_dict keys
        # Give it a reference and it will stick around.
        obj = UTCDateTime()
        _r3 = ResourceIdentifier(referred_object=obj)  # NOQA
        self.assertEqual(len(list(r_dict.keys())), 1)

    def test_adding_a_referred_object_after_creation(self):
        """
        Check that the referred objects can also be made available after the
        ResourceIdentifier instances have been created.
        """
        obj = UTCDateTime()
        res_id = "obspy.org/time/test"
        ref_a = ResourceIdentifier(res_id)
        ref_b = ResourceIdentifier(res_id)
        ref_c = ResourceIdentifier(res_id)
        # All three will have no resource attached.
        self.assertEqual(ref_a.get_referred_object(), None)
        self.assertEqual(ref_b.get_referred_object(), None)
        self.assertEqual(ref_c.get_referred_object(), None)
        # Setting the object for one will make it available to all other
        # instances, provided they weren't bound to specific objects.
        ref_b.set_referred_object(obj)
        self.assertIs(ref_a.get_referred_object(), obj)
        self.assertIs(ref_b.get_referred_object(), obj)
        self.assertIs(ref_c.get_referred_object(), obj)

    def test_getting_gc_no_shared_resource_id(self):
        """
        Test that calling get_referred_object on a resource id whose object
        has been garbage collected, and whose resource_id is unique,
        returns None
        """
        obj1 = UTCDateTime()
        rid1 = ResourceIdentifier(referred_object=obj1)
        # delete obj1, make sure rid1 return None
        del obj1
        # raises UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertIs(rid1.get_referred_object(), None)

    def test_getting_gc_with_shared_resource_id(self):
        """
        Test that calling get_referred_object on a resource id whose object
        has been garbage collected, but that has another object that shares
        the same resource_id, returns the other object with the same resource
        id and issues a warning
        """
        uri = 'testuri'
        obj1 = UTCDateTime(1000)
        obj2 = UTCDateTime(1000)
        rid1 = ResourceIdentifier(uri, referred_object=obj1)
        rid2 = ResourceIdentifier(uri, referred_object=obj2)
        self.assertFalse(rid1.get_referred_object() is
                         rid2.get_referred_object())
        self.assertNotEqual(rid1._object_id, rid2._object_id)
        del obj1
        warnings.simplefilter('default')
        with warnings.catch_warnings(record=True) as w:
            rid1.get_referred_object()
            self.assertEqual(len(w), 1)
            self.assertIn('The object with identity', str(w[0]))
        # now both rids should return the same object
        self.assertIs(rid1.get_referred_object(), rid2.get_referred_object())
        # the object id should now be bound to obj2
        self.assertEqual(rid1._object_id, rid2._object_id)

    def test_resources_in_global_dict_get_garbage_collected(self):
        """
        Tests that the ResourceIdentifiers in the class level resource dict get
        deleted if they have no other reference and the object they refer to
        goes out of scope.
        """
        obj_a = UTCDateTime()
        obj_b = UTCDateTime()
        res1 = ResourceIdentifier(referred_object=obj_a)
        res2 = ResourceIdentifier(referred_object=obj_b)
        # Now two keys should be in the global dict.
        rdict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        self.assertEqual(len(list(rdict.keys())), 2)
        del obj_a, obj_b
        # raises UserWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertIs(res1.get_referred_object(), None)
            self.assertIs(res2.get_referred_object(), None)

    def test_quakeml_regex(self):
        """
        Tests that regex used to check for QuakeML validatity actually works.
        """
        # This one contains all valid characters. It should pass the
        # validation.
        res_id = (
            "smi:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "1234567890-.*()_~'/abcdefghijklmnopqrstuvwxyzABCDEFGHIKLMNOPQR"
            "STUVWXYZ0123456789-.*()_~'+?=,;&")
        res = ResourceIdentifier(res_id)
        self.assertEqual(res_id, res.get_quakeml_uri())
        # The id has to valid from start to end. Due to the spaces this cannot
        # automatically be converted to a correct one.
        res_id = ("something_before smi:local/something  something_after")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri)
        # A colon is an invalid character.
        res_id = ("smi:local/hello:yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri)
        # Space as well
        res_id = ("smi:local/hello yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri)
        # Dots are fine
        res_id = ("smi:local/hello....yea")
        res = ResourceIdentifier(res_id)
        self.assertEqual(res_id, res.get_quakeml_uri())
        # Hats not
        res_id = ("smi:local/hello^^yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri)

    def test_resource_id_valid_quakemluri(self):
        """
        Test that a resource identifier per default (i.e. no arguments to
        __init__()) gets set up with a QUAKEML conform ID.
        """
        rid = ResourceIdentifier()
        self.assertEqual(rid.id, rid.get_quakeml_uri())

    def test_resource_id_tracking(self):
        """
        The class keeps track of all instances.
        """
        # Create a couple of lightweight objects for testing purposes.
        t1 = UTCDateTime(2013, 1, 1)
        t2 = UTCDateTime(2013, 1, 2)
        t3 = UTCDateTime(2013, 1, 3)

        # First assert, that all ResourceIds are tracked correctly.
        r1 = ResourceIdentifier("a", referred_object=t1)
        r2 = ResourceIdentifier("b", referred_object=t2)
        r3 = ResourceIdentifier("c", referred_object=t3)

        self.assertEqual(
            ResourceIdentifier._ResourceIdentifier__resource_id_tracker,
            {"a": 1, "b": 1, "c": 1})

        # Create a new instance, similar to the first one.
        r4 = ResourceIdentifier("a", referred_object=t1)
        self.assertEqual(
            ResourceIdentifier._ResourceIdentifier__resource_id_tracker,
            {"a": 2, "b": 1, "c": 1})

        # Now delete r2 and r4. They should not be tracked anymore.
        del r2
        del r4
        self.assertEqual(
            ResourceIdentifier._ResourceIdentifier__resource_id_tracker,
            {"a": 1, "c": 1})

        # Delete the two others. Nothing should be tracked any more.
        del r1
        del r3
        self.assertEqual(
            ResourceIdentifier._ResourceIdentifier__resource_id_tracker, {})

    def test_automatic_dereferring_if_resource_id_goes_out_of_scope(self):
        """
        Tests that objects that have no more referrer are no longer stored in
        the reference dictionary.
        """
        t1 = UTCDateTime(2010, 1, 1)  # test object
        r_dict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        rid = 'a'  # test resource id

        # Create object and assert the reference has been created.
        r1 = ResourceIdentifier(rid, referred_object=t1)
        self.assertEqual(r1.get_referred_object(), t1)
        self.assertTrue(rid in r_dict)
        # Deleting the object should remove the reference.
        del r1
        self.assertFalse(rid in r_dict)
        # Now create two equal references.
        r1 = ResourceIdentifier(rid, referred_object=t1)
        r2 = ResourceIdentifier(rid, referred_object=t1)
        self.assertEqual(r1.get_referred_object(), t1)
        # Deleting one should not remove the reference.
        del r1
        self.assertEqual(r2.get_referred_object(), t1)
        self.assertTrue(rid in r_dict)
        # Deleting the second one should
        del r2
        self.assertFalse(rid in r_dict)

    def test_initialize_with_resource_identifier(self):
        """
        Test initializing an ResourceIdentifier with an ResourceIdentifier.
        """
        rid = ResourceIdentifier()
        rid2 = ResourceIdentifier(str(rid))
        rid3 = ResourceIdentifier(rid)
        self.assertEqual(rid, rid2)
        self.assertEqual(rid, rid3)

    def test_error_message_for_failing_quakeml_id_conversion(self):
        """
        Converting an id to a QuakeML compatible id might fail. Test the
        error message.
        """
        invalid_id = "http://example.org"
        rid = ResourceIdentifier(invalid_id)
        with self.assertRaises(ValueError) as e:
            rid.get_quakeml_uri()
        self.assertEqual(
            e.exception.args[0],
            "The id 'http://example.org' is not a valid QuakeML resource "
            "identifier. ObsPy tried modifying it to "
            "'smi:local/http://example.org' but it is still not valid. Please "
            "make sure all resource ids are either valid or can be made valid "
            "by prefixing them with 'smi:<authority_id>/'. Valid ids are "
            "specified in the QuakeML manual section 3.1 and in particular "
            "exclude colons for the final part.")


class ResourceIDEventScopeTestCase(unittest.TestCase):
    """
    Test suit for ensuring event scoping of objects bound to
    ResourceIdentifier instances
    """

    def make_test_catalog(self):
        """
        Make a test catalog with fixed resource IDs some of which reference
        other objects belonging to the event (eg arrivals -> picks)
        """
        pick_rid = ResourceIdentifier(id='obspy.org/tests/test_pick')
        origin_rid = ResourceIdentifier(id='obspy.org/tests/test_origin')
        arrival_rid = ResourceIdentifier(id='obspy.org/tests/test_arrival')
        ar_pick_rid = ResourceIdentifier(id='obspy.org/tests/test_pick')
        catatlog_rid = ResourceIdentifier(id='obspy.org/tests/test_catalog')

        picks = [Pick(time=UTCDateTime(), resource_id=pick_rid)]
        arrivals = [Arrival(resource_id=arrival_rid, pick_id=ar_pick_rid)]
        origins = [Origin(arrivals=arrivals, resource_id=origin_rid)]
        events = [Event(picks=picks, origins=origins)]
        events[0].preferred_origin_id = str(origin_rid.id)
        catalog = Catalog(events=events, resource_id=catatlog_rid)
        # next bind all unbound resource_ids to the current event scope
        catalog.resource_id.bind_resource_ids()
        return catalog

    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()
        # set the test catalog as an attr for test access
        self.catalog = self.make_test_catalog()
        # save the catalog to a temp file for testing reading in the catalog
        self.catalog_path = tempfile.mkstemp()[1]
        self.catalog.write(self.catalog_path, 'quakeml')
        # create a list of equal catalogs/events created with read and copy
        self.event_list = [
            self.catalog[0],
            read_events(self.catalog_path)[0],
            read_events(self.catalog_path)[0],
            self.catalog.copy()[0],
            self.catalog.copy()[0],
        ]

    def test_preferred_origins(self):
        """
        test that the objects bound to the preferred origins are event scoped
        """
        for ev in self.event_list:
            self.assertIs(ev.preferred_origin(), ev.origins[0])

    def test_arrivals_refer_to_picks_in_same_event(self):
        """
        ensure the pick_ids of the arrivals refer to the pick belonging
        to the same event
        """
        for ev in self.event_list:
            pick_id = ev.picks[0].resource_id
            arrival_pick_id = ev.origins[0].arrivals[0].pick_id
            self.assertEqual(pick_id, arrival_pick_id)
            pick = ev.picks[0]
            arrival_pick = arrival_pick_id.get_referred_object()
            self.assertIs(pick, arrival_pick)

    def tearDown(self):
        # remove the temp file
        os.remove(self.catalog_path)


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

    def test_resource_ids_refer_to_newest_object(self):
        """
        Tests that resource ids which are assigned multiple times but point to
        identical objects always point to the newest object. This prevents some
        odd behaviour.
        """
        t1 = UTCDateTime(2010, 1, 1)
        t2 = UTCDateTime(2010, 1, 1)

        rid = ResourceIdentifier("a", referred_object=t1)  # @UnusedVariable
        rid = ResourceIdentifier("a", referred_object=t2)

        del t1

        self.assertEqual(rid.get_referred_object(), t2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogTestCase, 'test'))
    suite.addTest(unittest.makeSuite(CatalogBasemapTestCase, 'test'))
    suite.addTest(unittest.makeSuite(CatalogCartopyTestCase, 'test'))
    suite.addTest(unittest.makeSuite(EventTestCase, 'test'))
    suite.addTest(unittest.makeSuite(OriginTestCase, 'test'))
    suite.addTest(unittest.makeSuite(WaveformStreamIDTestCase, 'test'))
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    suite.addTest(unittest.makeSuite(BaseTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
