"""
Tests for the resource identifier

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import copy
import gc
import io
import itertools
import multiprocessing.pool
import pickle
import sys
import unittest
import warnings

from obspy import UTCDateTime, read_events
from obspy.core import event as event
from obspy.core.event.resourceid import ResourceIdentifier, _ResourceKey
from obspy.core.util.misc import _yield_obj_parent_attr
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.testing import (create_diverse_catalog,
                                     setup_context_testcase,
                                     WarningsCapture)


class ResourceIdentifierTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.resourceid.ResourceIdentifier.
    """
    def setUp(self):
        """
        Setup code to run before each test. Temporary replaces the state on
        the ResourceIdentifier class level to reset the ResourceID mechanisms
        before each run.
        """
        # setup temporary id dict for tests in this case and bind to self
        context = ResourceIdentifier._debug_class_state()
        state = setup_context_testcase(self, context)
        self.parent_id_tree = state['parent_id_tree']
        self.id_order = state['id_order']
        self.id_object_map = state['id_object_map']

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
        with WarningsCapture() as w:
            res_b = ResourceIdentifier(id=id, referred_object=object_b)
            self.assertEqual(len(w), 0)
        # if the set object is not equal to the last object set to the same
        # resource_id, however, a warning should be issued.
        with WarningsCapture() as w:
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
        Test that calling get_referred_object() on a resource id whose object
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
        del obj1
        with WarningsCapture() as w:
            rid1.get_referred_object()
            self.assertEqual(len(w), 1)
            self.assertIn('The object with identity', str(w[0]))
        # now both rids should return the same object
        self.assertIs(rid1.get_referred_object(), rid2.get_referred_object())
        # the object id should now be bound to obj2
        self.assertEqual(rid1._object_id, rid2._object_id)

    def test_resource_id_state_cleanup(self):
        """
        Tests that the state in the ResourceIdentifier class gets gets cleaned
        up when resource_ids are garbage collected.
        """
        obj_a = UTCDateTime()
        obj_b = UTCDateTime()
        res1 = ResourceIdentifier(referred_object=obj_a)
        res2 = ResourceIdentifier(referred_object=obj_b)
        # Now two keys should be in the global dict.
        self.assertEqual(len(self.id_order), 2)
        self.assertEqual(len(self.id_object_map), 2)
        del obj_a, obj_b
        # raises UserWarnings
        self.assertEqual(len(self.id_order), 2)
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
        self.assertEqual(res_id, res.get_quakeml_uri_str())
        # The id has to valid from start to end. Due to the spaces this cannot
        # automatically be converted to a correct one.
        res_id = "something_before smi:local/something  something_after"
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri_str)
        # A colon is an invalid character.
        res_id = "smi:local/hello:yea"
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri_str)
        # Space as well
        res_id = "smi:local/hello yea"
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri_str)
        # Dots are fine
        res_id = "smi:local/hello....yea"
        res = ResourceIdentifier(res_id)
        self.assertEqual(res_id, res.get_quakeml_uri_str())
        # Hats not
        res_id = "smi:local/hello^^yea"
        res = ResourceIdentifier(res_id)
        self.assertRaises(ValueError, res.get_quakeml_uri_str)

    def test_resource_id_valid_quakemluri(self):
        """
        Test that a resource identifier per default (i.e. no arguments to
        __init__()) gets set up with a QUAKEML conform ID.
        """
        rid = ResourceIdentifier()
        self.assertEqual(rid.id, rid.get_quakeml_uri_str())

    def test_de_referencing_when_object_goes_out_of_scope(self):
        """
        Tests that objects that have no more referrer are no longer stored in
        the reference dictionary.
        """
        r_dict = self.id_order
        t1 = UTCDateTime(2010, 1, 1)  # test object
        rid = 'a'  # test resource id

        # Create object and assert the reference has been created.
        r1 = ResourceIdentifier(rid, referred_object=t1)
        self.assertEqual(r1.get_referred_object(), t1)
        self.assertTrue(r1._resource_key in r_dict)
        # Deleting the object should remove the reference.
        r_dict_len = len(r_dict)
        del r1
        self.assertEqual(len(r_dict), r_dict_len - 1)
        # Now create two equal references.
        r1 = ResourceIdentifier(rid, referred_object=t1)
        r2 = ResourceIdentifier(rid, referred_object=t1)
        self.assertEqual(r1.get_referred_object(), t1)
        # Deleting one should not remove the reference.
        del r1
        self.assertEqual(r2.get_referred_object(), t1)
        self.assertIn(r2._resource_key, r_dict)
        # Deleting the second one should (r_dict should now be empty)
        del r2
        self.assertEqual(len(r_dict), 0)
        r3 = ResourceIdentifier(rid)
        self.assertNotIn(r3._resource_key, r_dict)

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
            rid.get_quakeml_uri_str()
        self.assertEqual(
            e.exception.args[0],
            "The id 'http://example.org' is not a valid QuakeML resource "
            "identifier. ObsPy tried modifying it to "
            "'smi:local/http://example.org' but it is still not valid. Please "
            "make sure all resource ids are either valid or can be made valid "
            "by prefixing them with 'smi:<authority_id>/'. Valid ids are "
            "specified in the QuakeML manual section 3.1 and in particular "
            "exclude colons for the final part.")

    def test_debug_class_state(self):
        """
        Ensure the debug_class_state method juggles the state within context
        manager.
        """
        def get_id_object_list():
            """ return the current _id_object_list """
            return ResourceIdentifier._id_order

        def make_resouce_id():
            some_obj = event.Event()
            return ResourceIdentifier(referred_object=some_obj)

        rdict1 = get_id_object_list()
        rid1 = make_resouce_id()

        self.assertIn(rid1._resource_key, rdict1)

        with ResourceIdentifier._debug_class_state():
            # rdict state should have been replaced
            self.assertNotIn(rid1._resource_key, get_id_object_list())
            rid2 = make_resouce_id()
            self.assertIn(rid2._resource_key, get_id_object_list())

            with ResourceIdentifier._debug_class_state():
                self.assertNotIn(rid1._resource_key, get_id_object_list())
                self.assertNotIn(rid2._resource_key, get_id_object_list())
                rid3 = make_resouce_id()
                self.assertIn(rid3._resource_key, get_id_object_list())

            self.assertNotIn(rid3._resource_key, get_id_object_list())
            self.assertIn(rid2._resource_key, get_id_object_list())

        self.assertNotIn(rid2._resource_key, get_id_object_list())
        self.assertIn(rid1._resource_key, get_id_object_list())

    def test_preferred_origin(self, catalogs=None):
        """
        Test preferred_origin is set and event scoped.
        """
        for cat in catalogs or make_diverse_catalog_list():
            preferred_origin = cat[0].preferred_origin()
            self.assertIsNotNone(preferred_origin)
            self.assertIs(preferred_origin, cat[0].origins[-1])

    def test_preferred_magnitude(self, catalogs=None):
        """
        Test preferred_magnitude is set and event scoped.
        """
        for cat in catalogs or make_diverse_catalog_list():
            preferred_magnitude = cat[0].preferred_magnitude()
            self.assertIsNotNone(preferred_magnitude)
            self.assertIs(preferred_magnitude, cat[0].magnitudes[-1])

    def test_preferred_focal_mechanism(self, catalogs=None):
        """
        Test preferred_focal_mechanism is set and event scoped.
        """
        for cat in catalogs or make_diverse_catalog_list():
            preferred_focal_mech = cat[0].preferred_focal_mechanism()
            self.assertIsNotNone(preferred_focal_mech)
            self.assertIs(preferred_focal_mech, cat[0].focal_mechanisms[-1])

    def test_arrivals_refer_to_picks_in_same_event(self, catalogs=None):
        """
        Ensure the pick_ids of the arrivals refer to the pick belonging
        to the same event.
        """
        for cat in catalogs or make_diverse_catalog_list():
            pick_id = cat[0].picks[0].resource_id
            arrival_pick_id = cat[0].origins[0].arrivals[0].pick_id
            self.assertEqual(pick_id, arrival_pick_id)
            pick = cat[0].picks[0]
            arrival_pick = arrival_pick_id.get_referred_object()
            self.assertIs(pick, arrival_pick)

    def test_all_referred_objects_in_events(self, catalogs=None):
        """
        All the referred object should be members of the current event.
        """
        for num, cat in enumerate(catalogs or make_diverse_catalog_list()):
            for ev in cat:
                ev_ids = get_object_id_dict(ev)
                gen = _yield_obj_parent_attr(ev, ResourceIdentifier)
                for rid, parent, attr in gen:
                    referred_object = rid.get_referred_object()
                    if referred_object is None:
                        continue
                    self.assertIn(id(referred_object), ev_ids)

    def test_all_resource_id_attrs_are_attached(self, catalogs=None):
        """
        Find all objects that have a resource_id attribute and ensure it
        is an instance of ResourceIdentifier and refers to the object.
        """
        for cat in catalogs or make_diverse_catalog_list():
            for obj in get_instances(cat, has_attr='resource_id'):
                if isinstance(obj, ResourceIdentifier):
                    continue
                rid = obj.resource_id
                self.assertIsInstance(rid, ResourceIdentifier)
                referred_object = rid.get_referred_object()
                self.assertIsNotNone(referred_object)
                # the attached resource id should refer to parent object
                self.assertIs(obj, referred_object)

    def test_no_overlapping_objects(self, catalogs=None):
        """
        Each event should share no objects, except the id_key singletons, or
        built-ins (such as str or int) with copies of the same event.
        """
        catalogs = catalogs or make_diverse_catalog_list()
        for cat1, cat2 in itertools.combinations(catalogs, 2):
            # get a dict of object id: object reference
            ids1 = get_non_built_in_id_dict(cat1)
            ids2 = get_non_built_in_id_dict(cat2)
            # get a dict of all singleton resource keys
            singleton_ids1 = get_object_id_dict(cat1, _ResourceKey)
            singleton_ids2 = get_object_id_dict(cat2, _ResourceKey)
            # get a dict of strings (needed for py2 due to futures module)
            str_ids1 = get_object_id_dict(cat1, str)
            str_ids2 = get_object_id_dict(cat2, str)
            # get a dict of all objects that are not singleton resource keys
            non_singleton1 = set(ids1) - set(singleton_ids1) - set(str_ids1)
            non_singleton2 = set(ids2) - set(singleton_ids2) - set(str_ids2)
            # find any overlap between events that are not resource keys
            overlap = non_singleton1 & non_singleton2
            self.assertEqual(len(overlap), 0)  # assert no overlap

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
        # while a deep copy should not, although they should be equal.
        self.assertIs(rid1, rid2)
        self.assertIsNot(rid1, rid3)
        self.assertEqual(rid1, rid3)
        # copy should point to the same object, deep copy should not
        self.assertIs(rob1, rob2)
        self.assertIsNot(rob1, rob3)
        # although the referred objects should be equal
        self.assertEqual(rob1, rob3)

    def test_set_referred_object_warning(self):
        """
        Setting a referred object equal to an object that is equal to the last
        referred object should not emit a warning.
        """
        def assert_differnt_referred_object_warning_issued(w):
            """ assert the different referred object warning is emitted """
            self.assertEqual(len(w), 1)
            self.assertIn('is not equal', str(w[0].message))

        obj1 = UTCDateTime(10)
        obj2 = UTCDateTime(11)
        obj3 = UTCDateTime(10)

        rid1 = ResourceIdentifier('abc123', referred_object=obj1)
        # this should raise a warning as the new object != old object
        with WarningsCapture() as w:
            rid2 = ResourceIdentifier('abc123', referred_object=obj2)
        assert_differnt_referred_object_warning_issued(w)

        with WarningsCapture() as w:
            rid2.set_referred_object(obj1)
        assert_differnt_referred_object_warning_issued(w)

        # set the referred object back to previous state, this should warn
        with WarningsCapture() as w:
            ResourceIdentifier('abc123', referred_object=obj2)
        assert_differnt_referred_object_warning_issued(w)

        # this should not emit a warning since obj1 == obj3
        with WarningsCapture() as w:
            rid1.set_referred_object(obj3)
        self.assertEqual(len(w), 0)

    def test_catalog_resource_ids(self):
        """
        Basic tests on the catalog resource ids.
        """
        cat1 = read_events()
        # The resource_id attached to the first event is self-pointing
        self.assertIs(cat1[0], cat1[0].resource_id.get_referred_object())
        # make a copy and re-read catalog
        cat2 = cat1.copy()
        cat3 = read_events()
        # the resource_id on the new catalogs point to attached objects
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

        gc.collect()  # Call gc to ensure WeakValueDict works
        # raises UserWarning, suppress to keep std out cleaner
        with WarningsCapture():
            self.assertIs(rid.get_referred_object(), cat2[0])
            del cat2
            self.assertIs(rid.get_referred_object(), None)

    def test_event_scoped_resource_id_many_threads(self):
        """
        Test that event-scoping of resource IDs still works when many
        threads are used to generate catalogs via various methods.
        """
        pool = multiprocessing.pool.ThreadPool()
        # currently it is not possible to avoid warnings with many threads
        with warnings.catch_warnings(record=True):
            nested_catalogs = pool.map(make_diverse_catalog_list, range(5))
        pool.close()
        # get a flat list of catalogs
        catalogs = list(itertools.chain.from_iterable(nested_catalogs))
        # run catalogs through previous tests
        self.test_all_referred_objects_in_events(catalogs)
        self.test_all_resource_id_attrs_are_attached(catalogs)
        self.test_no_overlapping_objects(catalogs)
        self.test_arrivals_refer_to_picks_in_same_event(catalogs)
        self.test_preferred_focal_mechanism(catalogs)
        self.test_preferred_magnitude(catalogs)
        self.test_preferred_origin(catalogs)

    def test_resetting_id_warns_on_default_id(self):
        """
        Because the ResourceIdentifier class hashes on the id attribute, it
        should warn if it is being changed. This tests the case were an id is
        not specified and the default uuid is used.
        """
        rid = ResourceIdentifier()
        with WarningsCapture() as w:
            warnings.simplefilter('default')
            rid.id = 'Another string that will mess up the hash. Bad.'

        self.assertEqual(len(w), 1)
        self.assertIn('overwritting the id attribute', str(w[0]))

    def test_resetting_id_warns_on_set_id(self):
        """
        Because the ResourceIdentifier class hashes on the id attribute, it
        should warn if it is being changed. This tests the case were an id is
        manually specified.
        """
        rid = ResourceIdentifier('a very unique string indeed')
        with WarningsCapture() as w:
            warnings.simplefilter('default')
            rid.id = 'Another string that will mess up the hash. Bad.'

        self.assertEqual(len(w), 1)
        self.assertIn('overwritting the id attribute', str(w[0].message))

    def test_resource_ids_refer_to_newest_object(self):
        """
        Tests that resource ids which are assigned multiple times but point to
        identical objects always point to the newest object.
        """
        t1 = UTCDateTime(2010, 1, 1)
        t2 = UTCDateTime(2010, 1, 1)

        ResourceIdentifier("a", referred_object=t1)
        rid = ResourceIdentifier("a", referred_object=t2)

        del t1

        self.assertIs(rid.get_referred_object(), t2)

    def test_get_pick_from_arrival_on_copied_catalog_doesnt_warn(self):
        """
        Ensure a copied catalog doesn't raise a warning when
        get_referred_object is called on a resource_id.
        """
        cat = create_diverse_catalog().copy()
        arrival = cat[0].origins[0].arrivals[0]
        with WarningsCapture() as w:
            arrival.pick_id.get_referred_object()
        self.assertEqual(len(w), 0)

    def test_issue_2278(self):
        """
        Tests for issue # 2278 which has to do with resource ids returning
        the wrong objects when the bound object has gone out of scope and
        a new object adopts the old object's python id.
        """
        # Create a simple class for resource_ids to refere to.
        class Simple(object):

            def __init__(self, value):
                self.value = value

        parent = Simple('parent1')

        # keep track of objects, resource_ids, and used python ids
        obj_list1, rid_list1, used_ids1 = [], [], set()
        # create a slew of objects and resource_ids
        for _ in range(100):
            obj_list1.append(Simple(1))
            used_ids1.add(id(obj_list1[-1]))
        for obj in obj_list1:
            kwargs = dict(referred_object=obj, parent=parent)
            rid_list1.append(ResourceIdentifier(**kwargs))
        # delete objects and create second set
        del obj_list1

        # create another slew of objects and resource_ids. Some will reuse
        # deleted object ids
        obj_list2, rid_list2, used_ids2 = [], [], set()
        for _ in range(100):
            obj_list2.append(Simple(2))
            used_ids2.add(id(obj_list2[-1]))
        for obj in obj_list2:
            kwargs = dict(referred_object=obj, parent=parent)
            rid_list2.append(ResourceIdentifier(**kwargs))

        # since we cannot control which IDs python uses, skip the test if
        # no overlapping ids were created.
        if not used_ids1 & used_ids2:
            self.skipTest('setup requires reuse of python ids')

        # Iterate over first list of ids. Referred objects should be None
        for rid in rid_list1:
            # should raise a warning
            with WarningsCapture():
                self.assertIs(rid.get_referred_object(), None)

    def test_get_object_hook(self):
        """
        Test that custom logic for getting referred objects can be plugged
        into resource ids.
        """

        class MyClass():
            pass

        new_obj1 = MyClass()
        new_obj2 = MyClass()

        def _get_object_hook(arg):
            if str(arg) == '123':
                return new_obj1
            elif str(arg) == '789':
                return new_obj2
            else:
                return None

        with ResourceIdentifier._debug_class_state():
            ResourceIdentifier.register_get_object_hook(_get_object_hook)
            rid1 = ResourceIdentifier('123')
            rid2 = ResourceIdentifier('456')
            rid3 = ResourceIdentifier('789')
            self.assertIs(rid1.get_referred_object(), new_obj1)
            self.assertIs(rid2.get_referred_object(), None)
            # Now clear the hook, _get_object_hook should no longer be called
            ResourceIdentifier.remove_get_object_hook(_get_object_hook)
            self.assertIs(rid3.get_referred_object(), None)
            # But rid1 should have been bound to new_obj1 (so it no longer
            # needs to call get_object_hook to find it
            self.assertIs(rid1.get_referred_object(), new_obj1)

    def test_mutative_methods_deprecation(self):
        """
        Because Resource ids are hashable they should be immutable. Make
        sure any methods that mutate resource_ids are deprecated. Currently
        there are two:

        1. `convert_id_to_quakeml_uri`
        2. `regnerate_uuid`
        """
        rid = ResourceIdentifier('not_a_valid_quakeml_uri')
        with WarningsCapture() as w:
            rid.convert_id_to_quakeml_uri()
        self.assertGreaterEqual(len(w), 1)
        self.assertTrue([isinstance(x, ObsPyDeprecationWarning) for x in w])

        rid = ResourceIdentifier()
        with WarningsCapture() as w:
            rid.regenerate_uuid()
        self.assertGreaterEqual(len(w), 1)
        self.assertTrue([isinstance(x, ObsPyDeprecationWarning) for x in w])

    def test_get_quakeml_id(self):
        """
        Tests for returning valid quakeml ids using the get_quakeml_id method.
        """
        obj = UTCDateTime('2017-09-17')
        rid1 = ResourceIdentifier('invalid_id', referred_object=obj)
        rid2 = rid1.get_quakeml_id(authority_id='remote')
        # The resource ids should not be equal but should refer to the same
        # object.
        self.assertNotEqual(rid2, rid1)
        self.assertIs(rid1.get_referred_object(), rid2.get_referred_object())
        # A valid resource id should return a resource id that is equal.
        rid3 = rid2.get_quakeml_id()
        self.assertEqual(rid2, rid3)


def get_instances(obj, cls=None, is_attr=None, has_attr=None):
    """
    Recurse object, return a list of instances meeting search criteria.

    :param obj:
        The object to recurse through attributes of lists, tuples, and other
        instances.
    :param cls:
        Only return instances of cls if not None, else return all instances.
    :param is_attr:
        Only return objects stored as attr_name, if None return all.
    :param has_attr:
        Only return objects that have attribute has_attr, if None return all.
    """
    gen = _yield_obj_parent_attr(obj, cls, is_attr=is_attr, has_attr=has_attr)
    return [x[0] for x in gen]


def get_object_id_dict(obj, cls=None):
    """
    Recurse an object and return a dict in the form of {id: object}.
    """
    return {id(x[0]): x[0] for x in _yield_obj_parent_attr(obj, cls)}


def get_non_built_in_id_dict(obj, cls=None):
    """
    Same as get_object_id_dict but exclude built-in data types.
    """

    obj_dict = get_object_id_dict(obj, cls=cls)
    return {item: val for item, val in obj_dict.items()
            if hasattr(val, '__dict__') or hasattr(val, '__slots__')}


def make_diverse_catalog_list(*args):  # NOQA
    """
    Make a list of diverse catalogs.

    Creates several copies of the diverse catalog, which is returned from
    :func:`~obspy.core.util.testing.create_diverse_catalog`. Copies are
    created with the copy method and reading a quakeml representation from
    a byte string.

    The unused args is necessary for thread-pool mapping.
    """
    # create a complex catalog
    cat1 = create_diverse_catalog()
    bytes_io = io.BytesIO()
    cat1.write(bytes_io, 'quakeml')
    # get a few copies from reading from bytes
    cat2 = read_events(bytes_io)
    cat3 = read_events(bytes_io)
    # make more catalogs with copy method
    cat4 = cat1.copy()
    cat5 = cat4.copy()
    # ensure creating a copying and deleting doesnt mess up id tracking
    cat_to_delete = cat2.copy()
    del cat_to_delete
    # pickle and unpickle catalog
    cat_bytes = pickle.dumps(cat4)
    cat6 = pickle.loads(cat_bytes)
    return [cat1, cat2, cat3, cat4, cat5, cat6]


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
