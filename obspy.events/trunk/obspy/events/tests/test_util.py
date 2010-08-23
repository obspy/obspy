# -*- coding: utf-8 -*-

from obspy.events.util import UniqueList
import unittest


class UtilTestCase(unittest.TestCase):
    """
    Test suite for obspy.events.util.
    """

    def test_UniqueListInit(self):
        """
        Make sure no duplicate entries are added at the init method.
        """
        # Autoremove duplicate entries.
        ul1 = UniqueList([1, 2, 3, 1, 2, 3])
        ul2 = UniqueList([1, 2, 3])
        self.assertEqual(ul1, ul2)
        # Raise type error if more than one argument.
        self.assertRaises(TypeError, UniqueList.__init__, 1, 2)
        # Init with non list types.
        ul1 = UniqueList((1, 2, 3, 1, 2, 3))
        self.assertEqual(ul1, ul2)
        # Empyt init should also work.
        ul = UniqueList()
        ul.append(1)
        self.assertEqual(ul, UniqueList([1]))

    def test_UniqueListAppend(self):
        """
        Test for appending to UniqueList.
        """
        ul = UniqueList([3, 1])
        ul.append(2)
        ul.append(1)
        self.assertEqual(ul, UniqueList([3, 1, 2]))

    def test_UniqueListExtend(self):
        """
        Test for extending to UniqueList.
        """
        ul = UniqueList([1, 2, 3])
        ul.extend(UniqueList([1, 2, 3, 4]))
        ul.extend([1, 2, 3, 4, 5])
        self.assertEqual(ul, UniqueList([1, 2, 3, 4, 5]))
        # Object need to be iterable.
        self.assertRaises(TypeError, UniqueList.extend, 1)

    def test_UniqueListSetItem(self):
        """
        Test for __setitem__ in UniqueList.
        """
        ul = UniqueList([1, 2, 3])
        ul[0] = 4
        self.assertEqual(ul, UniqueList([4, 2, 3]))
        ul[0] = 2
        self.assertEqual(ul, UniqueList([4, 2, 3]))

    def test_UniqueListSetSlice(self):
        """
        Test for __setslice__ in UniqueList.
        """
        ul = UniqueList([1, 2, 3])
        ul[0:2] = UniqueList([4, 5])
        self.assertEqual(ul, UniqueList([4, 5, 3]))
        ul[0:2] = UniqueList([2, 3])
        self.assertEqual(ul, UniqueList([4, 5, 3]))
        # Slice setting with duplicates.
        ul[0:2] = [7, 7]
        self.assertEqual(ul, UniqueList([4, 5, 3]))

    def test_UniqueListLockAndUnlock(self):
        """
        Test the locking and unlocking of Unique Lists.
        """
        ul = UniqueList([1, 2, 3])
        ul.append(4)
        self.assertEqual(ul, UniqueList([1, 2, 3, 4]))
        # Lock and change stuff.
        ul._lock()
        ul.append(5)
        ul.extend([6, 7])
        ul[0] = 8
        ul[1:3] = [9, 10]
        self.assertEqual(ul, UniqueList([1, 2, 3, 4]))
        # Unlock and do the same things.
        ul._unlock()
        ul.append(5)
        ul.extend([6, 7])
        ul[0] = 8
        ul[1:3] = [9, 10]
        self.assertEqual(ul, UniqueList([8, 9, 10, 4, 5, 6, 7]))


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
