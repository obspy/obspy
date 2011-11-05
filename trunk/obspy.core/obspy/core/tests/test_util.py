# -*- coding: utf-8 -*-

from obspy.core.util import AttribDict, calcVincentyInverse
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

    def test_calcVincentyInverse(self):
        """
        Tests for the Vincenty's Inverse formulae.
        """
        # the following will raise StopIteration exceptions because of two
        # nearly antipodal points
        self.assertRaises(StopIteration, calcVincentyInverse,
                          15.26804251, 2.93007342, -14.80522806, -177.2299081)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.3562106, 72.2382356, -27.55995499, -107.78571981)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calcVincentyInverse,
                          27.4675551, 17.28133229, -27.65771704, -162.65420626)
        self.assertRaises(StopIteration, calcVincentyInverse, 0, 0, 0, 13)
        # working examples
        self.assertAlmostEquals(calcVincentyInverse(0, 0.2, 0, 20),
                                (2204125.9174282863, 90.0, 270.0))
        self.assertAlmostEquals(calcVincentyInverse(0, 0, 0, 10),
                                (1113194.9077920639, 90.0, 270.0))
        self.assertAlmostEquals(calcVincentyInverse(0, 0, 0, 17),
                                (1892431.3432465086, 90.0, 270.0))
        # out of bounds
        self.assertRaises(ValueError, calcVincentyInverse, 91, 0, 0, 0)
        self.assertRaises(ValueError, calcVincentyInverse, -91, 0, 0, 0)
        self.assertRaises(ValueError, calcVincentyInverse, 0, 0, 91, 0)
        self.assertRaises(ValueError, calcVincentyInverse, 0, 0, -91, 0)


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
