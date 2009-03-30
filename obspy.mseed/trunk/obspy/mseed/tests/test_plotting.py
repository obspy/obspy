# -*- coding: utf-8 -*-
"""
The obspy.mseed plotting test suite.
"""

from obspy.mseed import libmseed
import inspect
import os
import unittest


class LibMSEEDPlottingTestCase(unittest.TestCase):
    """
    Test cases for Libmseed plotting.
    """
    def setUp(self):
        # Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
        
        outpath = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.outpath = os.path.join(outpath, 'output')
    
    def tearDown(self):
        pass
    
    def test_Plotting(self):
        """
        Creates plotted examples in test/output directory
        """
        mseed = libmseed()
        mseed_file = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
        
        #Full graph with user defined colors and size.
        mseed.graph_create_graph(mseed_file, os.path.join(self.outpath, 
            'full_graph_1111x222px_purple_graph_and_lightgreen_background'),\
                         size = (1111, 222), color = 'purple', \
                         bgcolor = 'lightgreen')
        #Reset memory
        mseed.resetMs_readmsr()
        
        #Graph with user defined start and endtime.
        mstg = mseed.readTraces(mseed_file, dataflag = 0)
        starttime = mstg.contents.traces.contents.starttime
        endtime = mstg.contents.traces.contents.endtime
        stime = mseed.mseedtimestringToDatetime(starttime - 86400 * 1e6)
        etime = mseed.mseedtimestringToDatetime(endtime + 86400 * 1e6)
        #Reset again.
        mseed.resetMs_readmsr()
        mseed.graph_create_graph(mseed_file, os.path.join(self.outpath, 
            'graph_1024x768px_with_one_empty_day_before_and_after_graph'),\
                         timespan = (stime, etime))
        
def suite():
    return unittest.makeSuite(LibMSEEDPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')