#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The libgse2 test suite.
"""

from obspy.gse2 import libgse2
import inspect, os, random, unittest


class LibGSE2TestCase(unittest.TestCase):
    """
    Test cases for libgse2.
    """
    def setUp(self):
        # directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass
    
    def test_read(self):
        """
        Compares waveform data read by libgse2 with an ASCII dump.
        
        Checks the first 13 datasamples when reading loc_RJOB20050831023349.z.
        The values are assumed to be correct. The values were created using
        getevents. Only checks relative values.
        """
        gse2_file = os.path.join(self.path, 'loc_RJOB20050831023349.z')
        # list of known data samples
        datalist = [12, -10, 16, 33, 9, 26, 16, 7, 17, 6, 1, 3, -2]
        header, data = libgse2.read(gse2_file)
        self.assertEqual('RJOB ', header['station'])
        self.assertEqual('  Z', header['channel'])
        self.assertEqual(200, header['samp_rate'])
        self.assertEqual('20050831023349.850', "%04d%02d%02d%02d%02d%06.3f" % (
            header['d_year'],
            header['d_mon'],
            header['d_day'],
            header['t_hour'],
            header['t_min'],
            header['t_sec']
            ))
        for i in range(len(datalist)-1):
            self.assertEqual(datalist[i]-datalist[i+1], data[i]-data[i+1])
        #from pylab import plot,array,show;plot(array(data));show()
    
    def test_readAnWrite(self):
        """
        Writes, reads and compares files created via libgse2.
        """
        gse2_file = os.path.join(self.path, 'loc_RNON20040609200559.z')
        header, data = libgse2.read(gse2_file)
        # define test ranges
        filename = 'temp.mseed'
        temp_file = os.path.join(self.path, filename)
        libgse2.write(header, data, temp_file)
        newheader, newdata = libgse2.read(temp_file)
        self.assertEqual(header, newheader)
        self.assertEqual(data, newdata)
        os.remove(temp_file)
    
    #def test_readFirstHeaderInfo(self):
    #    """
    #    Reads and compares header info from the first record.
    #    
    #    The values can be read from the filename.
    #    """
    #    mseed = libmseed()
    #    filename = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
    #    header = mseed.getFirstRecordHeaderInfo(filename)
    #    self.assertEqual(header['location'], '')
    #    self.assertEqual(header['network'], 'BW')
    #    self.assertEqual(header['station'], 'BGLD')
    #    self.assertEqual(header['channel'], 'EHE')
    #
    #def test_getStartAndEndTime(self):
    #    """
    #    Tests getting the start- and end time of a file.
    #    
    #    The values are compared with the readTraces() method which parses the
    #    whole file. This will only work for files with only one trace and with-
    #    out any gaps or overlaps.
    #    """
    #    mseed = libmseed()
    #    filename = os.path.join(self.path, 'BW.BGLD..EHE.D.2008.001')
    #    #mseed.getEndtime(filename)
    #    
    #    # get the start- and end time
    #    times = mseed.getStartAndEndTime(filename)
    #    # reseting the ms_readmsr() method of libmseed
    #    mseed.resetMs_readmsr()
    #    # parse the whole file
    #    mstg = mseed.readTraces(filename, dataflag = 0)
    #    chain = mstg.contents.traces.contents
    #    self.assertEqual(times[0], mseed.MSTime2Datetime(chain.starttime))
    #    self.assertEqual(times[1], mseed.MSTime2Datetime(chain.endtime))


def suite():
    return unittest.makeSuite(LibGSE2TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
