# -*- coding: utf-8 -*-

from obspy.core.util import AttribDict
import unittest


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
        self.assertEquals(temp, 1)
        self.assertFalse('test' in ad)
        self.assertTrue('test2' in ad)
        self.assertFalse('test' in ad.__dict__)
        self.assertTrue('test2' in ad.__dict__)
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
        self.assertEquals(temp, ('test2', 'test'))
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
        self.assertTrue('test2' in ad)
        self.assertFalse('test' in ad.__dict__)
        self.assertTrue('test2' in ad.__dict__)
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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)

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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)
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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)

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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)
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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)

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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)
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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)
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
        self.assertTrue('test' in ad)
        self.assertTrue('test' in ad.__dict__)


def suite():
    return unittest.makeSuite(AttribDictTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
