# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import unittest
import warnings

from obspy.core import AttribDict


class AttribDictTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util.attribdict
    """

    def test_pop(self):
        """
        Tests pop method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # removing via pop
        temp = ad.pop('test')
        self.assertEqual(temp, 1)
        self.assertFalse('test' in ad)
        self.assertIn('test2', ad)
        self.assertFalse('test' in ad.__dict__)
        self.assertIn('test2', ad.__dict__)
        self.assertFalse(hasattr(ad, 'test'))
        self.assertTrue(hasattr(ad, 'test2'))
        # using pop() for not existing element raises a KeyError
        self.assertRaises(KeyError, ad.pop, 'test')

    def test_popitem(self):
        """
        Tests pop method of AttribDict class.
        """
        ad = AttribDict()
        ad['test2'] = 'test'
        # removing via popitem
        temp = ad.popitem()
        self.assertEqual(temp, ('test2', 'test'))
        self.assertFalse('test2' in ad)
        self.assertFalse('test2' in ad.__dict__)
        self.assertFalse(hasattr(ad, 'test2'))
        # popitem for empty AttribDict raises a KeyError
        self.assertRaises(KeyError, ad.popitem)

    def test_delete(self):
        """
        Tests delete method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # deleting test using dictionary
        del ad['test']
        self.assertFalse('test' in ad)
        self.assertIn('test2', ad)
        self.assertFalse('test' in ad.__dict__)
        self.assertIn('test2', ad.__dict__)
        self.assertFalse(hasattr(ad, 'test'))
        self.assertTrue(hasattr(ad, 'test2'))
        # deleting test2 using attribute
        del ad.test2
        self.assertFalse('test2' in ad)
        self.assertFalse('test2' in ad.__dict__)
        self.assertFalse(hasattr(ad, 'test2'))

    def test_init(self):
        """
        Tests initialization of AttribDict class.
        """
        ad = AttribDict({'test': 'NEW'})
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)

    def test_setitem(self):
        """
        Tests __setitem__ method of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad['test'] = 'NEW'
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)
        # 2
        ad = AttribDict()
        ad.__setitem__('test', 'NEW')
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)

    def test_setattr(self):
        """
        Tests __setattr__ method of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad.test = 'NEW'
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)
        # 2
        ad = AttribDict()
        ad.__setattr__('test', 'NEW')
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)

    def test_setdefault(self):
        """
        Tests setdefault method of AttribDict class.
        """
        ad = AttribDict()
        # 1
        default = ad.setdefault('test', 'NEW')
        self.assertEqual(default, 'NEW')
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)
        # 2 - existing key should not be overwritten
        default = ad.setdefault('test', 'SOMETHINGDIFFERENT')
        self.assertEqual(default, 'NEW')
        self.assertEqual(ad['test'], 'NEW')
        self.assertEqual(ad.test, 'NEW')
        self.assertEqual(ad.get('test'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__dict__['test'], 'NEW')
        self.assertEqual(ad.__dict__.get('test'), 'NEW')
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)
        # 3 - default value isNone
        ad = AttribDict()
        default = ad.setdefault('test')
        self.assertEqual(default, None)
        self.assertEqual(ad['test'], None)
        self.assertEqual(ad.test, None)
        self.assertEqual(ad.get('test'), None)
        self.assertEqual(ad.__getattr__('test'), None)
        self.assertEqual(ad.__getitem__('test'), None)
        self.assertEqual(ad.__dict__['test'], None)
        self.assertEqual(ad.__dict__.get('test'), None)
        self.assertIn('test', ad)
        self.assertIn('test', ad.__dict__)

    def test_clear(self):
        """
        Tests clear method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # removing via pop
        ad.clear()
        self.assertFalse('test' in ad)
        self.assertFalse('test2' in ad)
        self.assertFalse('test' in ad.__dict__)
        self.assertFalse('test2' in ad.__dict__)
        self.assertFalse(hasattr(ad, 'test'))
        self.assertFalse(hasattr(ad, 'test2'))
        # class attributes should be still present
        self.assertTrue(hasattr(ad, 'readonly'))
        self.assertTrue(hasattr(ad, 'defaults'))

    def test_init_argument(self):
        """
        Tests initialization of AttribDict with various arguments.
        """
        # one dict works as expected
        ad = AttribDict({'test': 1})
        self.assertEqual(ad.test, 1)
        # multiple dicts results into TypeError
        self.assertRaises(TypeError, AttribDict, {}, {})
        self.assertRaises(TypeError, AttribDict, {}, {}, blah=1)
        # non-dicts results into TypeError
        self.assertRaises(TypeError, AttribDict, 1)
        self.assertRaises(TypeError, AttribDict, object())

    def test_defaults(self):
        """
        Tests default of __getitem__/__getattr__ methods of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad['test'] = 'NEW'
        self.assertEqual(ad.__getitem__('test'), 'NEW')
        self.assertEqual(ad.__getitem__('xxx', 'blub'), 'blub')
        self.assertEqual(ad.__getitem__('test', 'blub'), 'NEW')
        self.assertEqual(ad.__getattr__('test'), 'NEW')
        self.assertEqual(ad.__getattr__('xxx', 'blub'), 'blub')
        self.assertEqual(ad.__getattr__('test', 'blub'), 'NEW')
        # should raise KeyError without default item
        self.assertRaises(KeyError, ad.__getitem__, 'xxx')
        self.assertRaises(AttributeError, ad.__getattr__, 'xxx')
        # 2
        ad2 = AttribDict(defaults={'test2': 'NEW'})
        self.assertEqual(ad2.__getitem__('test2'), 'NEW')
        self.assertRaises(KeyError, ad2.__getitem__, 'xxx')

    def test_set_readonly(self):
        """
        Tests of setting readonly attributes.
        """
        class MyAttribDict(AttribDict):
            readonly = ['test']
            defaults = {'test': 1}

        ad = MyAttribDict()
        self.assertEqual(ad.test, 1)
        self.assertRaises(AttributeError, ad.__setitem__, 'test', 1)

    def test_deepcopy(self):
        """
        Tests __deepcopy__ method of AttribDict.
        """
        class MyAttribDict(AttribDict):
            defaults = {'test': 1}

        ad = MyAttribDict()
        ad.muh = 2
        ad2 = ad.__deepcopy__()
        self.assertEqual(ad2.test, 1)
        self.assertEqual(ad2.muh, 2)

    def test_compare_with_dict(self):
        """
        Checks if AttribDict is still comparable to a dict object.
        """
        adict = {'test': 1}
        ad = AttribDict(adict)
        self.assertEqual(ad, adict)
        self.assertEqual(adict, ad)

    def test_pretty_str(self):
        """
        Test _pretty_str method of AttribDict.
        """
        # 1
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = '           test1: 1\n           test2: 2'
        self.assertEqual(ad._pretty_str(), out)
        # 2
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = '           test2: 2\n           test1: 1'
        self.assertEqual(ad._pretty_str(priorized_keys=['test2']), out)
        # 3
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = ' test1: 1\n test2: 2'
        self.assertEqual(ad._pretty_str(min_label_length=6), out)

    def test_types(self):
        """
        Test that types are enforced with _types attribute
        """
        class AttrOcity(AttribDict):
            _types = {'string': str, 'number': (float, int), 'int': int,
                      'another_number': (float, int)}

        ad = AttrOcity()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('default')
            ad.string = int
            ad.number = '1'
            ad.int = 1.0
            ad.not_type_controlled = 2
            ad.another_number = 1

        self.assertEqual(len(w), 3)
        self.assertIsInstance(ad.string, str)
        self.assertIsInstance(ad.number, float)
        self.assertIsInstance(ad.int, int)
        self.assertIsInstance(ad.another_number, int)


def suite():
    return unittest.makeSuite(AttribDictTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
