"""
Tests for the resource identifier
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import gc
import itertools
import os
import sys
import tempfile
import unittest
import warnings

from future.builtins import str, list, object, dict

import obspy.core.event as ev
from obspy import UTCDateTime, read_events, Catalog
from obspy.core.event import ResourceIdentifier


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

    def test_dubug_class_state(self):
        """
        Ensure the debug_class_state method juggles the state within context
        manager.
        """

        def current_rdict():
            """ return the current __resource_id_weak_dict """
            RI = ResourceIdentifier
            return RI._ResourceIdentifier__resource_id_weak_dict

        def make_resouce_id():
            some_obj = ev.Event()
            return ResourceIdentifier(referred_object=some_obj)

        rdict1 = current_rdict()
        rid1 = make_resouce_id()

        self.assertIn(rid1.id, rdict1)

        with ResourceIdentifier._debug_class_state():
            # rdict state should have been replaced
            self.assertNotIn(rid1.id, current_rdict())
            rid2 = make_resouce_id()
            self.assertIn(rid2.id, current_rdict())

            with ResourceIdentifier._debug_class_state():
                self.assertNotIn(rid1.id, current_rdict())
                self.assertNotIn(rid2.id, current_rdict())
                rid3 = make_resouce_id()
                self.assertIn(rid3.id, current_rdict())

            self.assertNotIn(rid3.id, current_rdict())
            self.assertIn(rid2.id, current_rdict())

        self.assertNotIn(rid2.id, current_rdict())
        self.assertIn(rid1.id, current_rdict())


class ResourceIDEventScopeTestCase(unittest.TestCase):
    """
    Test suit for ensuring event scoping of objects bound to
    ResourceIdentifier instances
    """

    def run(self, result=None):
        with ResourceIdentifier._debug_class_state() as state:
            self.state = state
            super(ResourceIDEventScopeTestCase, self).run(result)

    def setUp(self):
        # Clear the Resource Identifier dict for the tests. NEVER do this
        # otherwise.
        # context = ResourceIdentifier._debug_class_state
        # self.state = setup_with_context_manager(self, context)
        # import pdb; pdb.set_trace()
        # ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict.clear()
        # Also clear the tracker.
        # ResourceIdentifier._ResourceIdentifier__resource_id_tracker.clear()
        # set the test catalog as an attr for test access
        self.catalog = MegaCat().catalog
        # save the catalog to a temp file for testing reading in the catalog
        self.catalog_path = tempfile.mkstemp()[1]
        self.catalog.write(self.catalog_path, 'quakeml')
        # copying then deleting seems to affect other copied catalogs

        r_dict = ResourceIdentifier._ResourceIdentifier__resource_id_weak_dict
        count = ResourceIdentifier._ResourceIdentifier__resource_id_tracker
        unbound = ResourceIdentifier._ResourceIdentifier__unbound_resource_id

        from pprint import pprint
        import pdb; pdb.set_trace()

        pprint(r_dict)
        pprint(count)
        pprint(unbound)

        old = self.catalog.copy()

        pprint(r_dict)
        pprint(count)
        pprint(unbound)

        del old

        pprint(r_dict)
        pprint(count)
        pprint(unbound)

        # create a list of equal catalogs/events created with read and copy
        self.event_list = [
            self.catalog[0],
            read_events(self.catalog_path)[0],
            read_events(self.catalog_path)[0],
            self.catalog.copy()[0],
            self.catalog.copy()[0],
        ]

    def test_preferred_origin(self):
        """
        Test preferred_origin is set and event scoped.
        """
        for ev in self.event_list:
            preferred_origin = ev.preferred_origin()
            self.assertIsNotNone(preferred_origin)
            self.assertIs(preferred_origin, ev.origins[-1])

    def test_preferred_magnitude(self):
        """
        Test preferred_magnitude is set and event scoped.
        """
        for ev in self.event_list:
            preferred_magnitude = ev.preferred_magnitude()
            self.assertIsNotNone(preferred_magnitude)
            self.assertIs(preferred_magnitude, ev.magnitudes[-1])

    def test_preferred_focal_mechanism(self):
        """
        Test preferred_focal_mechanism is set and event scoped.
        """
        for ev in self.event_list:
            preferred_focal_mech = ev.preferred_focal_mechanism()
            self.assertIsNotNone(preferred_focal_mech)
            self.assertIs(preferred_focal_mech, ev.focal_mechanisms[-1])

    def test_arrivals_refer_to_picks_in_same_event(self):
        """
        Ensure the pick_ids of the arrivals refer to the pick belonging
        to the same event.
        """
        for ev in self.event_list:
            pick_id = ev.picks[0].resource_id
            arrival_pick_id = ev.origins[0].arrivals[0].pick_id
            self.assertEqual(pick_id, arrival_pick_id)
            pick = ev.picks[0]
            arrival_pick = arrival_pick_id.get_referred_object()
            self.assertIs(pick, arrival_pick)

    def test_all_referred_objects_in_event(self):
        """
        All the referred object should be members of the current event.
        """
        for ev in self.event_list:
            self.assertTrue(all_resource_ids_in_event(ev))

    def test_no_overlapping_objects(self):
        """
        Each event should share no objects (only copies) with other events.
        """
        # get a dict of all ids found in objects in each event
        ids = [get_object_id_dict(ev) for ev in self.event_list]
        # iterate all combinations and make sure there is no overlap
        for id_set1, id_set2 in itertools.combinations(ids, 2):
            shared_objects = set(id_set1) & set(id_set2)
            self.assertEqual(len(shared_objects), 0)

    def tearDown(self):
        # remove the temp file
        os.remove(self.catalog_path)


def get_instances(obj, cls=None):
    """
    Recurse obj return a list of all instances of cls.

    If cls is None, return all python instances (that have a __dict__ attr)
    """
    instance_cache = {}

    def _get_instances(obj, cls, ids=None):
        out = []
        ids = ids or set()
        if id(obj) in ids:
            return []
        if (id(obj), cls) in instance_cache:
            return instance_cache[(id(obj), cls)]
        ids.add(id(obj))
        dict_or_slots = hasattr(obj, '__dict__') or hasattr(obj, '__slots__')
        if cls is None or isinstance(obj, cls):
            if dict_or_slots and cls is None:
                out.append(obj)
        if hasattr(obj, '__dict__'):
            for item, val in obj.__dict__.items():
                out += _get_instances(val, cls, ids)
        if isinstance(obj, (list, tuple)):
            for val in obj:
                out += _get_instances(val, cls, ids)
        instance_cache[(id(obj), cls)] = out
        return out

    return _get_instances(obj, cls)


def setup_with_context_manager(testcase, cm):
    """Use a contextmanager to setUp a test case."""
    val = cm.__enter__()
    testcase.addCleanup(cm.__exit__, None, None, None)
    return val


def get_object_id_dict(obj):
    """
    Recurse an object and return a dict of id: object
    """
    return {id(x): x for x in get_instances(obj)}


def all_resource_ids_in_event(event):
    """
    Return True if all the objects referred to by resource_ids are in
    contained in the event.
    """
    object_ids = {id(x) for x in get_instances(event)}
    # iterate resource ids and ensure referred objects are in event
    for resource_id in get_instances(event, ResourceIdentifier):
        referred = resource_id.get_referred_object()
        if referred is None:
            continue
        if not id(referred) in object_ids:
            return False
    return True


class MegaCat(object):
    """
    Create a catalog with a single event that has many features.
    """

    def __init__(self):
        self.time = UTCDateTime('2016-05-04T12:00:01')
        events = [self._create_event()]
        self.catalog = Catalog(events=events)

    def _create_event(self):
        event = ev.Event(
            event_type='mining explosion',
            event_descriptions=[self._get_event_description()],
            picks=[self._create_pick()],
            origins=[self._create_origins()],
            station_magnitudes=[self._get_station_mag()],
            magnitudes=[self._create_magnitudes()],
            amplitudes=[self._get_amplitudes()],
            focal_mechanisms=[self._get_focal_mechanisms()],
        )
        # set preferred origin, focal mech, magnitude
        preferred_objects = dict(
            origin=event.origins[-1].resource_id,
            focal_mechanism=event.focal_mechanisms[-1].resource_id,
            magnitude=event.magnitudes[-1].resource_id,
        )
        for item, value in preferred_objects.items():
            setattr(event, 'preferred_' + item + '_id', value)

        return event

    def _create_pick(self):
        # setup some of the classes
        creation = ev.CreationInfo(
            agency='SwanCo',
            author='Indago',
            creation_time=UTCDateTime(),
            version='10.10',
            author_url=ResourceIdentifier('me.com'),
        )

        pick = ev.Pick(
            time=self.time,
            comments=[ev.Comment(x) for x in 'BOB'],
            evaluation_mode='manual',
            evaluation_status='final',
            creation_info=creation,
            phase_hint='P',
            polarity='positive',
            onset='emergent',
            back_azimith_errors={"uncertainty": 10},
            slowness_method_id=ResourceIdentifier('something_very_slow'),
            backazimuth=122.1,
            horizontal_slowness=12,
            method_id=ResourceIdentifier('pick.com'),
            horizontal_slowness_errors={'uncertainty': 12},
            filter_id=ResourceIdentifier('bandpass0_10'),
            waveform_id=ev.WaveformStreamID('UU', 'FOO', '--', 'HHZ'),
        )
        self.pick_id = pick.resource_id
        return pick

    def _create_origins(self):
        ori = ev.Origin(
            resource_id=ResourceIdentifier('First'),
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
            reference_system_id=ResourceIdentifier('AWS'),
            method_id=ResourceIdentifier('HYPOINVERSE'),
            earth_model_id=ResourceIdentifier('First'),
            arrivals=[self._get_arrival()],
            composite_times=[self._get_composite_times()],
            quality=self._get_origin_quality(),
            origin_type='hypocenter',
            origin_uncertainty=self._get_origin_uncertainty(),
            region='US',
            evaluation_mode='manual',
            evaluation_status='final',
        )
        self.origin_id = ori.resource_id
        return ori

    def _get_arrival(self):
        return ev.Arrival(
            resource_id=ResourceIdentifier('Ar1'),
            pick_id=self.pick_id,
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
            earth_model_id=ResourceIdentifier('1232'),
            commens=[ev.Comment(x) for x in 'Nothing'],
        )

    def _get_composite_times(self):
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

    def _get_origin_quality(self):
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

    def _get_origin_uncertainty(self):
        return ev.OriginUncertainty(
            horizontal_uncertainty=1.2,
            min_horizontal_uncertainty=.12,
            max_horizontal_uncertainty=2.2,
            confidence_ellipsoid=self._get_confidence_ellipsoid(),
            preferred_description="uncertainty ellipse",
        )

    def _get_confidence_ellipsoid(self):
        return ev.ConfidenceEllipsoid(
            semi_major_axis_length=12,
            semi_minor_axis_length=12,
            major_axis_plunge=12,
            major_axis_rotation=12,
        )

    def _create_magnitudes(self):
        return ev.Magnitude(
            resource_id=ResourceIdentifier(),
            mag=5.5,
            mag_errors={'uncertainty': .01},
            magnitude_type='Mw',
            origin_id=self.origin_id,
            station_count=1,
            station_magnitude_contributions=[self._get_station_mag_contrib()],
        )

    def _get_station_mag(self):
        station_mag = ev.StationMagnitude(
            mag=2.24,
        )
        self.station_mag_id = station_mag.resource_id
        return station_mag

    def _get_station_mag_contrib(self):
        return ev.StationMagnitudeContribution(
            station_magnitude_id=self.station_mag_id,
        )

    def _get_event_description(self):
        return ev.EventDescription(
            text='some text about the EQ',
            type='earthquake name',
        )

    def _get_amplitudes(self):
        return ev.Amplitude(
            generic_amplitude=.0012,
            type='A',
            unit='m',
            period=1,
            time_window=self._get_timewindow(),
            pick_id=self.pick_id,
            scalling_time=self.time,
            mangitude_hint='ML',
        )

    def _get_timewindow(self):
        return ev.TimeWindow(
            begin=1.2,
            end=2.2,
            reference=UTCDateTime('2016-05-04T12:00:00'),
        )

    def _get_focal_mechanisms(self):
        return ev.FocalMechanism(
            nodal_planes=self._get_nodal_planes(),
            principal_axis=self._get_principal_axis(),
            azimuthal_gap=12,
            station_polarity_count=12,
            misfit=.12,
            station_distribution_ratio=.12,
            moment_tensor=self._get_moment_tensor(),
        )

    def _get_nodal_planes(self):
        return ev.NodalPlanes(
            nodal_plane_1=ev.NodalPlane(strike=12, dip=2, rake=12),
            nodal_plane_2=ev.NodalPlane(strike=12, dip=2, rake=12),
            preferred_plane=2,
        )

    def _get_principal_axis(self):
        return ev.PrincipalAxes(
            t_axis=15,
            p_axis=15,
            n_axis=15,
        )

    def _get_moment_tensor(self):
        return ev.MomentTensor(
            scalar_moment=12213,
            tensor=self._get_tensor(),
            variance=12.23,
            variance_reduction=98,
            double_couple=.22,
            clvd=.55,
            iso=.33,
            source_time_function=self._get_source_time_function(),
            data_used=[self._get_data_used()],
            method_id=ResourceIdentifier(),
            inversion_type='general',
        )

    def _get_tensor(self):
        return ev.Tensor(
            m_rr=12,
            m_rr_errors={'uncertainty': .01},
            m_tt=12,
            m_pp=12,
            m_rt=12,
            m_rp=12,
            m_tp=12,
        )

    def _get_source_time_function(self):
        return ev.SourceTimeFunction(
            type='triangle',
            duration=.12,
            rise_time=.33,
            decay_time=.23,
        )

    def _get_data_used(self):
        return ev.DataUsed(
            wave_type='body waves',
            station_count=12,
            component_count=12,
            shortest_period=1,
            longest_period=20,
        )

    def __call__(self):
        return self.catalog


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    suite.addTest(unittest.makeSuite(ResourceIDEventScopeTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
