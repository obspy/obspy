"""
The obspy.clients.seedlink.client.slstate test suite.
"""
import unittest

from obspy.clients.seedlink.client.slstate import SLState


class SLStateTestCase(unittest.TestCase):

    def test_issue561(self):
        """
        Assure that different state objects don't share data buffers.
        """
        slstate1 = SLState()
        slstate2 = SLState()

        self.assertNotEqual(id(slstate1.databuf), id(slstate2.databuf))
        self.assertNotEqual(id(slstate1.packed_buf), id(slstate2.packed_buf))


def suite():
    return unittest.makeSuite(SLStateTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
