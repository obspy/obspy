# -*- coding: utf-8 -*-

from obspy.core.util import AttribDict
import unittest


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.core.util
    """

    def test_popAttribDict(self):
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

    def test_popitemAttribDict(self):
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

    def test_deleteAttribDict(self):
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


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
