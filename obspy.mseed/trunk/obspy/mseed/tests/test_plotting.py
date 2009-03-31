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
    
    def test_LibMSEEDPlotting(self):
        """
        Creates plotted examples in test/output directory
        """
        mseed = libmseed()
        mseed_file = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
        #Calculate full minmaxlist once and use it for caching.
        minmaxlist = mseed.graph_create_min_max_list(mseed_file, 777)
        
        #Full graph with user defined colors and size.
        mseed.graph_create_graph(mseed_file, os.path.join(self.outpath,
            'full_graph_777x222px_purple_and_lightgreen'), size = (777, 222),
            color = 'purple', bgcolor = 'lightgreen', minmaxlist = minmaxlist)
        
        #Same graph as before but returned as a binary imagestring.
        imgstring = mseed.graph_create_graph(mseed_file, format = 'png',
            size = (777, 222), color = 'purple', bgcolor = 'lightgreen',
            minmaxlist = minmaxlist)
        imgfile = open(os.path.join(self.outpath,
                       'full_graph_777x222px_purple_and_lightgreen_imagestr'),
                       'wb')
        imgfile.write(imgstring)
        imgfile.close()
        
        #Same graph as above but red with transparent background
        mseed.graph_create_graph(mseed_file, os.path.join(self.outpath,
            'full_graph_777x222px_red_and_transparent'), size = (777, 222),
            transparent = True, minmaxlist = minmaxlist)
        
        #Graph with user defined start and endtime both outside the graph.
        mstg = mseed.readTraces(mseed_file, dataflag = 0)
        starttime = mstg.contents.traces.contents.starttime
        endtime = mstg.contents.traces.contents.endtime
        #Graph begins one day before the file and ends one day after the file.
        stime = mseed.mseedtimestringToDatetime(starttime - 86400 * 1e6)
        etime = mseed.mseedtimestringToDatetime(endtime + 86400 * 1e6)
        #Create graph
        mseed.graph_create_graph(mseed_file, os.path.join(self.outpath, 
            'graph_1024x768px_with_one_empty_day_before_and_after_graph'),\
                         timespan = (stime, etime))
        
    def test_PlottingOutputFormats(self):
        """
        Test various output formats.
        """
        mseed = libmseed()
        mseed_file = os.path.join(self.path,
                                  'BW.BGLD..EHE.D.2008.001.first_record')
        #Calculate full minmaxlist once and use it for caching.
        minmaxlist = mseed.graph_create_min_max_list(mseed_file, 50)
        # PDF
        data = mseed.graph_create_graph(mseed_file, format = 'pdf',\
                                        size = (50, 50),\
                                        minmaxlist = minmaxlist)
        self.assertEqual(data[0:4], "%PDF")
        # PS
        data = mseed.graph_create_graph(mseed_file, format = 'ps',\
                                        size = (50, 50),\
                                        minmaxlist = minmaxlist)
        self.assertEqual(data[0:4], "%!PS")
        # PNG
        data = mseed.graph_create_graph(mseed_file, format = 'png',\
                                        size = (50, 50),\
                                        minmaxlist = minmaxlist)
        self.assertEqual(data[1:4], "PNG")
        # SVG
        data = mseed.graph_create_graph(mseed_file, format = 'svg',\
                                        size = (50, 50),\
                                        minmaxlist = minmaxlist)
        self.assertEqual(data[0:5], "<?xml")
        
def suite():
    return unittest.makeSuite(LibMSEEDPlottingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')