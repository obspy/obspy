# -*- coding: utf-8 -*-

from obspy.mseed import libmseed
import inspect
import os
import unittest


class LibMSEEDTestCase(unittest.TestCase):
    
    def setUp(self):
        #Directory where the test files are located
        path = os.path.dirname(inspect.getsourcefile(self.__class__))
        self.path = os.path.join(path, 'data')
    
    def tearDown(self):
        pass

#    def test_read_using_Traces(self):
#        """
#        Checks about the first 5000 datasamples when reading BW.BGLD..EHE.D.2008.001
#        using traces. The values in BW.BGLD..EHE.D.2008.001_first5000lines.ASCII
#        are assumed to be correct.
#        Only checks relative values.
#        """
#        mseed=libmseed()
#        f=open(os.path.join(self.path,'BW.BGLD..EHE.D.2008.001_first5000lines.ASCII'),'r')
#        datalist=f.readlines()
#        station='BGLD'
#        channel='EHE'
#        frequency=200
#        starttime=1199145599915000
#        datalist[0:7]=[]
#        for i in range(len(datalist)):
#            datalist[i]=int(datalist[i])
#        header, data, numtraces=mseed.read_ms_using_traces(os.path.join(self.path,'BW.BGLD..EHE.D.2008.001'))
#        self.assertEqual(station, header['station'])
#        self.assertEqual(channel, header['channel'])
#        self.assertEqual(frequency, header['samprate'])
#        self.assertEqual(starttime, header['starttime'])
#        self.assertEqual(numtraces, 1)
#        for i in range(len(datalist)-1):
#            self.assertEqual(datalist[i]-datalist[i+1], data[i]-data[i+1])
#    
#    def test_read_an_write_MS_using_Traces(self):
#        """
#        A reencoded SEED file should still have the same values regardless of 
#        the used record length, encoding and byteorder.
#        """
#        # define test ranges
#        record_length_values = [2**i for i in range(8,21)]
#        encoding_values = [1, 3, 10, 11]
#        byteorder_values = [0, 1]
#        
#        mseed=libmseed() 
#        header, data, numtraces=mseed.read_ms_using_traces(os.path.join(self.path,
#                                                           'test.mseed'))
#        # Deletes the dataquality indicators
#        testheader=header.copy()
#        del testheader['dataquality']
#        # loops over all combinations of test values
#        for reclen in record_length_values:
#            for byteorder in byteorder_values:
#                for encoding in encoding_values:
#                    filename = 'temp.%s.%s.%s.mseed' % (reclen, byteorder, 
#                                                        encoding)
#                    temp_file = os.path.join(self.path, filename)
#                    mseed.write_ms(header, data, temp_file,
#                                   numtraces, encoding=encoding, 
#                                   byteorder=byteorder, reclen=reclen)
#                    newheader, newdata, newnumtraces=mseed.read_ms_using_traces(temp_file)
#                    del newheader['dataquality']
#                    self.assertEqual(testheader, newheader)
#                    self.assertEqual(data, newdata)
#                    self.assertEqual(numtraces, newnumtraces)
#                    os.remove(temp_file)

    def test_findgaps(self):
        mseed = libmseed()
        gapslist = mseed.findgaps(os.path.join(self.path,'gaps.mseed'))
        starttime = 1199145599915000
        self.assertEqual(gapslist[0][0], starttime+1970000)
        self.assertEqual(gapslist[1][0], starttime+8150000)
        self.assertEqual(gapslist[2][0], starttime+14330000)
        self.assertEqual(gapslist[0][1], 2065000)
        self.assertEqual(gapslist[1][1], 2065000)
        self.assertEqual(gapslist[2][1], 4125000)



def suite():
    return unittest.makeSuite(LibMSEEDTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
