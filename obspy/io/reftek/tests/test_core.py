#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport


class ReftekTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.reftek
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")
        self.filename = "225051000_00008656"
        self.filepath = os.path.join(self.datapath, self.filename)
        # files "2015282_225051_0ae4c_1_[123].msd" contain miniseed data
        # converted with "rt_mseed" tool of Reftek utilities.

        # information on the data packets in the test file:
        # (omitting leading/trailing EH/ET packet)
        #   >>> for p in packets[1:-1]:
        #   ...     print(
        #   ...         "{:02d}".format(p.packet_sequence),
        #   ...         p.channel_number,
        #   ...         "{:03d}".format(p.number_of_samples),
        #   ...         p.time)
        #   01 0 549 2015-10-09T22:50:51.000000Z
        #   02 1 447 2015-10-09T22:50:51.000000Z
        #   03 2 805 2015-10-09T22:50:51.000000Z
        #   04 0 876 2015-10-09T22:50:53.745000Z
        #   05 1 482 2015-10-09T22:50:53.235000Z
        #   06 1 618 2015-10-09T22:50:55.645000Z
        #   07 2 872 2015-10-09T22:50:55.025000Z
        #   08 0 892 2015-10-09T22:50:58.125000Z
        #   09 1 770 2015-10-09T22:50:58.735000Z
        #   10 2 884 2015-10-09T22:50:59.385000Z
        #   11 0 848 2015-10-09T22:51:02.585000Z
        #   12 1 790 2015-10-09T22:51:02.585000Z
        #   13 2 844 2015-10-09T22:51:03.805000Z
        #   14 0 892 2015-10-09T22:51:06.215000Z
        #   15 1 768 2015-10-09T22:51:05.925000Z
        #   16 2 884 2015-10-09T22:51:08.415000Z
        #   17 1 778 2015-10-09T22:51:10.765000Z
        #   18 0 892 2015-10-09T22:51:11.675000Z
        #   19 2 892 2015-10-09T22:51:12.835000Z
        #   20 1 736 2015-10-09T22:51:14.655000Z
        #   21 0 892 2015-10-09T22:51:16.135000Z
        #   22 2 860 2015-10-09T22:51:17.295000Z
        #   23 1 738 2015-10-09T22:51:18.335000Z
        #   24 0 892 2015-10-09T22:51:20.595000Z
        #   25 1 673 2015-10-09T22:51:22.025000Z
        #   26 2 759 2015-10-09T22:51:21.595000Z
        #   27 0 067 2015-10-09T22:51:25.055000Z


def suite():
    return unittest.makeSuite(ReftekTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
