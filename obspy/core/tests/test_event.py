# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
from obspy.core.event import readEvents, Catalog, Event, WaveformStreamID, \
    Origin, CreationInfo, ResourceIdentifier, Comment, Pick
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import getBasemapVersion
from obspy.core.util.testing import ImageComparison
from obspy.core.util.decorator import skipIf
import os
import sys
import unittest
import warnings


BASEMAP_VERSION = getBasemapVersion()
if BASEMAP_VERSION:
    from matplotlib import rcParams


class EventTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.Event
    """
    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

    def test_str(self):
        """
        Testing the __str__ method of the Event object.
        """
        event = readEvents()[1]
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
        self.assertTrue(ev1 == ev2)
        self.assertTrue(ev2 == ev1)
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
        self.assertTrue(e == Event(force_resource_id=False))
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
        p.test_1 = "a"
        p.test_2 = "b"
        self.assertEqual(p.test_1, "a")
        self.assertEqual(p.test_2, "b")
        p.clear()
        self.assertEqual(len(p.comments), 0)
        self.assertEqual(p.phase_hint, None)
        self.assertFalse(hasattr(p, "test_1"))
        self.assertFalse(hasattr(p, "test_2"))

    def test_event_copying_does_not_raise_duplicate_resource_id_warnings(self):
        """
        Tests that copying an event does not raise a duplicate resource id
        warning.
        """
        ev = readEvents()[0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev2 = copy.copy(ev)
            self.assertEqual(len(w), 0)
            ev3 = copy.deepcopy(ev)
            self.assertEqual(len(w), 0)

        # The two events should compare equal.
        self.assertEqual(ev, ev2)
        self.assertEqual(ev, ev3)

        # A shallow copy should just use the exact same resource identifier,
        # while a deep copy should not.
        self.assertTrue(ev.resource_id is ev2.resource_id)
        self.assertTrue(ev.resource_id is not ev3.resource_id)
        self.assertTrue(ev.resource_id == ev3.resource_id)

        # But all should point to the same object.
        self.assertTrue(ev.resource_id.getReferredObject() is
                        ev2.resource_id.getReferredObject())
        self.assertTrue(ev.resource_id.getReferredObject() is
                        ev3.resource_id.getReferredObject())


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

    def test_creationInfo(self):
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

    def test_multipleOrigins(self):
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
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.iris_xml = os.path.join(path, 'iris_events.xml')
        self.neries_xml = os.path.join(path, 'neries_events.xml')
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()

    def test_creationInfo(self):
        cat = Catalog()
        cat.creation_info = CreationInfo(author='test2')
        self.assertTrue(isinstance(cat.creation_info, CreationInfo))
        self.assertEqual(cat.creation_info.author, 'test2')

    def test_readEventsWithoutParameters(self):
        """
        Calling readEvents w/o any parameter will create an example catalog.
        """
        catalog = readEvents()
        self.assertEqual(len(catalog), 3)

    def test_str(self):
        """
        Testing the __str__ method of the Catalog object.
        """
        catalog = readEvents()
        self.assertTrue(catalog.__str__().startswith("3 Event(s) in Catalog:"))
        self.assertTrue(catalog.__str__().endswith("37.736 | 3.0 ML | manual"))

    def test_readEvents(self):
        """
        Tests the readEvents function using entry points.
        """
        # iris
        catalog = readEvents(self.iris_xml)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog[0]._format, 'QUAKEML')
        self.assertEqual(catalog[1]._format, 'QUAKEML')
        # neries
        catalog = readEvents(self.neries_xml)
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog[0]._format, 'QUAKEML')
        self.assertEqual(catalog[1]._format, 'QUAKEML')
        self.assertEqual(catalog[2]._format, 'QUAKEML')

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

    def test_countAndLen(self):
        """
        Tests the count and __len__ methods of the Catalog object.
        """
        # empty catalog without events
        catalog = Catalog()
        self.assertEqual(len(catalog), 0)
        self.assertEqual(catalog.count(), 0)
        # catalog with events
        catalog = readEvents()
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog.count(), 3)

    def test_getitem(self):
        """
        Tests the __getitem__ method of the Catalog object.
        """
        catalog = readEvents()
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
        catalog = readEvents()
        self.assertEqual(catalog[0:], catalog[0:])
        self.assertEqual(catalog[:2], catalog[:2])
        self.assertEqual(catalog[:], catalog[:])
        self.assertEqual(len(catalog), 3)
        new_catalog = catalog[1:3]
        self.assertTrue(isinstance(new_catalog, Catalog))
        self.assertEqual(len(new_catalog), 2)

    def test_slicingWithStep(self):
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
        cat = readEvents()
        cat2 = cat.copy()
        self.assertTrue(cat == cat2)
        self.assertTrue(cat2 == cat)
        self.assertFalse(cat is cat2)
        self.assertFalse(cat2 is cat)
        self.assertTrue(cat.events[0] == cat2.events[0])
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
        cat = readEvents()
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
        cat = readEvents(self.neries_xml)
        self.assertEqual(str(cat.resource_id), r"smi://eu.emsc/unid")

    @skipIf(not BASEMAP_VERSION, 'basemap not installed')
    def test_catalog_plot_cylindrical(self):
        """
        Tests the catalog preview plot, default parameters.
        """
        cat = readEvents()
        with ImageComparison(self.image_dir, "catalog1.png") as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(outfile=ic.name)

    @skipIf(not BASEMAP_VERSION, 'basemap not installed')
    def test_catalog_plot_ortho(self):
        """
        Tests the catalog preview plot, ortho projection, some non-default
        parameters.
        """
        cat = readEvents()
        with ImageComparison(self.image_dir, "catalog2.png") as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(outfile=ic.name, projection="ortho",
                     resolution="c",
                     water_fill_color="b", label=None)

    @skipIf(not BASEMAP_VERSION, 'basemap not installed')
    def test_catalog_plot_local(self):
        """
        Tests the catalog preview plot, local projection, some more non-default
        parameters.
        """
        cat = readEvents()
        reltol = 1.5
        # Basemap smaller 1.0.4 has a serious issue with plotting. Thus the
        # tolerance must be much higher.
        if BASEMAP_VERSION < [1, 0, 4]:
            reltol = 100
        with ImageComparison(self.image_dir, "catalog3.png",
                             reltol=reltol) as ic:
            rcParams['savefig.dpi'] = 72
            cat.plot(outfile=ic.name, projection="local",
                     resolution="i", continent_fill_color="0.3",
                     color="date", colormap="gist_heat")


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

        Skipped for Python 2.5 because it does not have the catch_warnings
        context manager.
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
        objects. This should not happen and thus a warning should be emitted.
        """
        object_a = UTCDateTime(1000)
        object_b = UTCDateTime(1001)
        self.assertEqual(object_a is object_b, False)
        id = 'obspy.org/tests/test_resource'
        res_a = ResourceIdentifier(id=id,
                                   referred_object=object_a)
        # Now create a new resource with the same id but a different object.
        # This will raise a warning.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('error', UserWarning)
            self.assertRaises(UserWarning, ResourceIdentifier,
                              id=id,
                              referred_object=object_b)
            # Now ignore the warning and actually create the new
            # ResourceIdentifier.
            warnings.simplefilter('ignore', UserWarning)
            res_b = ResourceIdentifier(id=id,
                                       referred_object=object_b)
        # Object b was the last to added, thus all resource identifiers will
        # now point to it.
        self.assertEqual(object_b is res_a.getReferredObject(), True)
        self.assertEqual(object_b is res_b.getReferredObject(), True)

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
        # Adding a ResourceIdentifier with an object that has a reference
        # somewhere will have no effect because it gets garbage collected
        # pretty much immediately.
        _r2 = ResourceIdentifier(referred_object=UTCDateTime())  # NOQA
        self.assertEqual(len(list(r_dict.keys())), 0)
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
        obj_id = id(obj)
        res_id = "obspy.org/time/test"
        ref_a = ResourceIdentifier(res_id)
        ref_b = ResourceIdentifier(res_id)
        ref_c = ResourceIdentifier(res_id)
        # All three will have no resource attached.
        self.assertEqual(ref_a.getReferredObject(), None)
        self.assertEqual(ref_b.getReferredObject(), None)
        self.assertEqual(ref_c.getReferredObject(), None)
        # Setting the object for one will make it available to all other
        # instances.
        ref_b.setReferredObject(obj)
        self.assertEqual(id(ref_a.getReferredObject()), obj_id)
        self.assertEqual(id(ref_b.getReferredObject()), obj_id)
        self.assertEqual(id(ref_c.getReferredObject()), obj_id)

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
        # Deleting the objects should also remove the from the dictionary.
        del obj_a, obj_b
        self.assertEqual(len(list(rdict.keys())), 0)
        # references are still around but no longer have associates objects.
        self.assertEqual(res1.getReferredObject(), None)
        self.assertEqual(res2.getReferredObject(), None)

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
        self.assertEqual(res_id, res.getQuakeMLURI())
        # The id has to valid from start to end. Due to the spaces this cannot
        # automatically be converted to a correct one.
        res_id = ("something_before smi:local/something  something_after")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.getQuakeMLURI)
        # A colon is an invalid character.
        res_id = ("smi:local/hello:yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.getQuakeMLURI)
        # Space as well
        res_id = ("smi:local/hello yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.getQuakeMLURI)
        # Dots are fine
        res_id = ("smi:local/hello....yea")
        res = ResourceIdentifier(res_id)
        self.assertEqual(res_id, res.getQuakeMLURI())
        # Hats not
        res_id = ("smi:local/hello^^yea")
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.getQuakeMLURI)

    def test_resource_id_valid_quakemluri(self):
        """
        Test that a resource identifier per default (i.e. no arguments to
        __init__()) gets set up with a QUAKEML conform ID.
        """
        rid = ResourceIdentifier()
        self.assertEqual(rid.id, rid.getQuakeMLURI())

    def test_resource_id_init_deprecation(self):
        """
        Test that a resource identifier initialized with deprecated
        "resource_id" gets initialized correctly and that a warning is shown.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            rid = ResourceIdentifier(resource_id="blablup")
        self.assertEqual(rid.id, "blablup")
        self.assertEqual(len(w), 1)
        w = w[0]
        self.assertEqual(w.category, DeprecationWarning)
        self.assertTrue(
            str(w.message).startswith("Deprecated keyword resource_id "))

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
        t1 = UTCDateTime(2010, 1, 1)

        # Create object and assert the reference has been created.
        r1 = ResourceIdentifier("a", referred_object=t1)
        self.assertEqual(
            dict(
                ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict),
            {"a": t1})
        # Deleting the object should remove the reference.
        del r1
        self.assertEqual(
            dict(
                ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict),
            {})

        # Now create two equal references.
        r1 = ResourceIdentifier("a", referred_object=t1)
        r2 = ResourceIdentifier("a", referred_object=t1)
        self.assertEqual(
            dict(
                ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict),
            {"a": t1})
        # Deleting one should not remove the reference.
        del r1
        self.assertEqual(
            dict(
                ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict),
            {"a": t1})
        # Deleting the second one should
        del r2
        self.assertEqual(
            dict(
                ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict),
            {})


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CatalogTestCase, 'test'))
    suite.addTest(unittest.makeSuite(EventTestCase, 'test'))
    suite.addTest(unittest.makeSuite(OriginTestCase, 'test'))
    suite.addTest(unittest.makeSuite(WaveformStreamIDTestCase, 'test'))
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
