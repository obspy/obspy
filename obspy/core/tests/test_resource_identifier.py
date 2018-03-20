"""
Tests for the resource identifier
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import itertools
import sys
import unittest
import warnings

from obspy import UTCDateTime, read_events
from obspy.core import event as event
from obspy.core.event.resourceid import ResourceIdentifier, _ResourceSingleton
from obspy.core.util.testing import (MegaCatalog, setup_context_testcase,
                                     WarningsCapture)


class ResourceIdentifierTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.event.ResourceIdentifier.
    """

    # setup and utility function for tests

    def setUp(self):
        """
        Setup code to run before each test. Temporary replaces the
        resource_id_weak_dict and unbound_resource_id on the
        ResourceIdentifier class level and binds the new dicts to self with
        keys 'r_dict' and 'unbound'.
        """
        # setup temporary id dict for tests in this case and bind to self
        context = ResourceIdentifier._debug_class_state()
        state = setup_context_testcase(self, context)
        self.r_dict = state['rdict']  # the weak resource dict
        self.unbound = state['unbound']  # the ubound resource ids

    def print_state(self):
        """
        Print the current resource_id state, very useful for debugging
        """
        from pprint import pprint
        print('-' * 79)
        print('resource_dict:')
        pprint(dict(self.r_dict))
        print('-' * 79)
        print('unbound:')
        pprint(dict(self.unbound))
        print('-' * 79)

    # tests

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
        _r1 = ResourceIdentifier()  # NOQA
        self.assertEqual(len(list(self.r_dict.keys())), 0)
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
        self.assertEqual(len(list(self.r_dict.keys())), 1)

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
        self.assertEqual(len(list(self.r_dict.keys())), 2)
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

    def test_automatic_dereferring_if_resource_id_goes_out_of_scope(self):
        """
        Tests that objects that have no more referrer are no longer stored in
        the reference dictionary.
        """
        with ResourceIdentifier._debug_class_state() as id_state:
            r_dict = id_state['rdict']
            t1 = UTCDateTime(2010, 1, 1)  # test object
            rid = 'a'  # test resource id

            # Create object and assert the reference has been created.
            r1 = ResourceIdentifier(rid, referred_object=t1)
            self.assertEqual(r1.get_referred_object(), t1)
            self.assertTrue(r1._id_key in r_dict)
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
            self.assertIn(r2._id_key, r_dict)
            # Deleting the second one should (r_dict should not be empty)
            del r2
            self.assertEqual(len(r_dict), 0)
            r3 = ResourceIdentifier(rid)
            self.assertNotIn(r3._id_key, r_dict)

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
            some_obj = event.Event()
            return ResourceIdentifier(referred_object=some_obj)

        rdict1 = current_rdict()
        rid1 = make_resouce_id()

        self.assertIn(rid1._id_key, rdict1)

        with ResourceIdentifier._debug_class_state():
            # rdict state should have been replaced
            self.assertNotIn(rid1._id_key, current_rdict())
            rid2 = make_resouce_id()
            self.assertIn(rid2._id_key, current_rdict())

            with ResourceIdentifier._debug_class_state():
                self.assertNotIn(rid1._id_key, current_rdict())
                self.assertNotIn(rid2._id_key, current_rdict())
                rid3 = make_resouce_id()
                self.assertIn(rid3._id_key, current_rdict())

            self.assertNotIn(rid3._id_key, current_rdict())
            self.assertIn(rid2._id_key, current_rdict())

        self.assertNotIn(rid2._id_key, current_rdict())
        self.assertIn(rid1._id_key, current_rdict())

    def test_copy_catalog(self):
        """
        Test that the number of the object ids in each resource id dict value
        ticks up once for each copy of the catalog made.
        """
        # generate some catalogs
        cat_list = make_mega_catalog_list()
        # list of dicts containing referred object ids
        id_list = list(self.r_dict.values())
        for id_dict in id_list:
            # get the length of living weak refs in id_dict
            alive_len = len([x for _, x in id_dict.items()
                             if x() is not None])
            self.assertEqual(alive_len, len(cat_list))

    def test_preferred_origin(self):
        """
        Test preferred_origin is set and event scoped.
        """
        for ev in make_mega_catalog_list():
            preferred_origin = ev[0].preferred_origin()
            self.assertIsNotNone(preferred_origin)
            self.assertIs(preferred_origin, ev[0].origins[-1])

    def test_preferred_magnitude(self):
        """
        Test preferred_magnitude is set and event scoped.
        """
        for ev in make_mega_catalog_list():
            preferred_magnitude = ev[0].preferred_magnitude()
            self.assertIsNotNone(preferred_magnitude)
            self.assertIs(preferred_magnitude, ev[0].magnitudes[-1])

    def test_preferred_focal_mechanism(self):
        """
        Test preferred_focal_mechanism is set and event scoped.
        """
        for ev in make_mega_catalog_list():
            preferred_focal_mech = ev[0].preferred_focal_mechanism()
            self.assertIsNotNone(preferred_focal_mech)
            self.assertIs(preferred_focal_mech, ev[0].focal_mechanisms[-1])

    def test_arrivals_refer_to_picks_in_same_event(self):
        """
        Ensure the pick_ids of the arrivals refer to the pick belonging
        to the same event.
        """
        for ev in make_mega_catalog_list():
            pick_id = ev[0].picks[0].resource_id
            arrival_pick_id = ev[0].origins[0].arrivals[0].pick_id
            self.assertEqual(pick_id, arrival_pick_id)
            pick = ev[0].picks[0]
            arrival_pick = arrival_pick_id.get_referred_object()
            self.assertIs(pick, arrival_pick)

    def test_all_referred_objects_in_catalog(self):
        """
        All the referred object should be members of the current one event
        catalog.
        """
        for ev in make_mega_catalog_list():
            ev_ids = get_object_id_dict(ev)  # all ids containe in dict
            for rid in get_instances(ev, ResourceIdentifier):
                referred_object = rid.get_referred_object()
                if referred_object is None:
                    continue
                self.assertIn(id(referred_object), ev_ids)

    def test_all_resource_id_attrs_are_attached(self):
        """
        Find all objects that have a resource_id attribute and ensure it
        is an instance of ResourceIdentifier and refers to the object.
        """
        catalogs = make_mega_catalog_list()
        for cat in catalogs:  # iterate the test catalog
            for obj in get_instances(cat, has_attr='resource_id'):
                if isinstance(obj, ResourceIdentifier):
                    continue
                rid = obj.resource_id
                self.assertIsInstance(rid, ResourceIdentifier)
                referred_object = rid.get_referred_object()
                self.assertIsNotNone(referred_object)
                # the attached resource id should refer to parent object
                self.assertIs(obj, referred_object)

    def test_no_overlapping_objects(self):
        """
        Each event should share no objects, except the id_key singletons, with
        copies of the same event.
        """
        catalogs = make_mega_catalog_list()
        for cat1, cat2 in itertools.combinations(catalogs, 2):
            # get a dict of object id: object reference
            ids1 = get_object_id_dict(cat1)
            ids2 = get_object_id_dict(cat2)
            # get a dict of all singleton resource keys
            singleton_ids1 = get_object_id_dict(cat1, _ResourceSingleton)
            singleton_ids2 = get_object_id_dict(cat2, _ResourceSingleton)
            # get a dict of strings (needed for py2 due to futures module)
            str_ids1 = get_object_id_dict(cat1, str)
            str_ids2 = get_object_id_dict(cat2, str)
            # get a dict of all objects that are not singleton resource keys
            non_singleton1 = set(ids1) - set(singleton_ids1) - set(str_ids1)
            non_singleton2 = set(ids2) - set(singleton_ids2) - set(str_ids2)
            # find any overlap between events that are not resource keys
            overlap = non_singleton1 & non_singleton2
            self.assertEqual(len(overlap), 0)  # assert no overlap

    def test_resetting_id_warns_on_default_id(self):
        """
        Because the ResourceIdentifier class hashes on the id attribute, it
        should warn if it is being changed. This tests the case were an id is
        not specified and the default uuid is used.
        """
        # test
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
        # test
        rid = ResourceIdentifier('a very unique string indeed')
        with WarningsCapture() as w:
            warnings.simplefilter('default')
            rid.id = 'Another string that will mess up the hash. Bad.'

        self.assertEqual(len(w), 1)
        self.assertIn('overwritting the id attribute', str(w[0].message))


def get_instances(obj, cls=None, is_attr=None, has_attr=None):
    """
    Recurse object, return a list of instances of meeting search criteria.

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
    instance_cache = {}

    def _get_instances(obj, cls, ids=None, attr=None):
        out = []
        ids = ids or set()  # id cache to avoid circular references
        if id(obj) in ids:
            return []
        if (id(obj), cls) in instance_cache:
            return instance_cache[(id(obj), cls)]
        ids.add(id(obj))
        if cls is None or isinstance(obj, cls):
            # filter out built-ins by looking for __dict__ or __slots__
            not_bultin = hasattr(obj, '__dict__') or hasattr(obj, '__slots__')
            # check if this object is stored as the desired attribute
            is_attribute = is_attr is None or attr == is_attr
            # check if object has desired attribute
            has_attribute = has_attr is None or hasattr(obj, has_attr)
            if not_bultin and is_attribute and has_attribute:
                out.append(obj)
        if hasattr(obj, '__dict__'):
            for item, val in obj.__dict__.items():
                out += _get_instances(val, cls, ids, attr=item)
        if isinstance(obj, (list, tuple)):
            for val in obj:
                out += _get_instances(val, cls, ids, attr=attr)
        instance_cache[(id(obj), cls)] = out
        return out

    return _get_instances(obj, cls)


def make_mega_catalog_list():
    """
    Make a list of complex catalogs (and copies) and return it.
    """
    # create a complex catalog
    cat1 = MegaCatalog().catalog
    # ResourceIdentifier.bind_resource_ids()
    bytes_io = io.BytesIO()
    cat1.write(bytes_io, 'quakeml')
    # get a few copies from reading from bytes
    cat2 = read_events(bytes_io)
    cat3 = read_events(bytes_io)
    # make more catalogs with copy method
    cat4 = cat1.copy()
    cat5 = cat4.copy()
    # ensure creating a copying and deleting doesnt mess up id tracking
    cat6 = cat2.copy()
    del cat6
    return [cat1, cat2, cat3, cat4, cat5]


def get_object_id_dict(obj, cls=None):
    """
    Recurse an object and return a dict of id: object
    """
    return {id(x): x for x in get_instances(obj, cls)}


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ResourceIdentifierTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
