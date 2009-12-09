#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: sacio.py
#  Purpose: Read & Write Seismograms, Format SAC.
#   Author: Yannik Behr, C. J. Ammon's
#    Email: yannik.behr@vuw.ac.nz
#
# Copyright (C) 2008-2010 Yannik Behr, C. J. Ammon's
#---------------------------------------------------------------------
""" 
An object-oriented version of C. J. Ammon's SAC I/O module.
Here is C. J. Ammon's his introductory comment:

Version 2.0.3, by C.J. Ammon, Penn State
This software is free and is distributed with no guarantees.
For a more complete description, start python and enter,
Suspected limitations: I don't used XY files much - I am
not sure that those access routines are bug free.

Send bug reports (not enhancement/feature requests) to: 
cja12@psu.edu [with PySAC in the subject field]
I don't support this software so don't wait for an answer.
I may not have time...


This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.


The ReadSac class provides the following functions:

Reading:

    ReadSacFile       - read binary SAC file
    ReadSacXY         - read XY SAC file    
    ReadSacHeader     - read SAC header
    GetHvalue         - extract information from header

Writing:

    WriteSacHeader    - write SAC header
    WriteSacBinary    - write binary SAC file
    WriteSacXY        - write ascii SAC file
    SetHvalue         - set SAC header variable

Convenience:

    IsSACfile         - test if valid binary SAC file
    IsXYSACfile       - test if valid XY SAC file
    ListStdValues     - print common header values
    GetHvalueFromFile - access to specific header item in specified file
    SetHvalueInFile   - change specific header item in specified file
    IsValidSacFile    - test for valid binary SAC file (wraps 'IsSACfile')

    
#################### TESTS ########################################
    
"""    

import os
import time, copy
from obspy.core import UTCDateTime
import numpy as np


class SacError(Exception):
    pass


class SacIOError(Exception):
    pass

    
class ReadSac(object):
    """ Class for SAC file IO
    initialise with: t=ReadSac()"""

    def __init__(self,filen=False,headonly=False,alpha=False):
        self.fdict = {'delta':0, 'depmin':1, 'depmax':2, 'scale':3,   \
                      'odelta':4, 'b':5, 'e':6, 'o':7, 'a':8,'int1':9,'t0':10,\
                      't1':11,'t2':12,'t3':13,'t4':14,'t5':15,'t6':16,\
                      't7':17,'t8':18,'t9':19,'f':20,'stla':31,'stlo':32,    \
                      'stel':33,'stdp':34,'evla':35,'evlo':36,'evdp':38,'mag':39, \
                      'user0':40,'user1':41,'user2':42,'user3':43,\
                      'user4':44,'user5':45,'user6':46,'user7':47,\
                      'user8':48,'user9':49,'dist':50,'az':51,'baz':52,\
                      'gcarc':53,'depmen':56,'cmpaz':57,'cmpinc':58}

        self.idict = {'nzyear':0, 'nzjday':1, 'nzhour':2, 'nzmin':3, \
                      'nzsec':4, 'nzmsec':5, 'nvhdr':6, 'norid':7, \
                      'nevid':8,'npts':9, 'nwfid':11, \
                      'iftype':15,'idep':16,'iztype':17,'iinst':19,\
                      'istreg':20,'ievreg':21,'ievtype':22,'iqual':23,\
                      'isynth':24,'imagtyp':25,'imagsrc':26, \
                      'leven':35,'lpspol':36,'lovrok':37,\
                      'lcalda':38}

        self.sdict = {'kstnm':0,'kevnm':1,'khole':2, 'ko':3,'ka':4,\
                      'kt0':5,'kt1':6,'kt2':7,'kt3':8,'kt4':9,\
                      'kt5':10,'kt6':11,'kt7':12,'kt8':13,\
                      'kt9':14,'kf':15,'kuser0':16,'kuser1':17,\
                      'kuser2':18,'kcmpnm':19,'knetwk':20,\
                      'kdatrd':21,'kinst':22}
        self.byteorder = 'little'
        self.InitArrays()
        self.headonly = headonly
        if filen:
            if alpha:
                self.__call__(filen,alpha=True)
            else:
                self.__call__(filen)


    def __call__(self,filename,alpha=False):
        if alpha:
            self.ReadSacXY(filename)
        elif self.headonly:
            self.ReadSacHeader(filename)
        else:
            self.ReadSacFile(filename)


    def InitArrays(self):
        """
        Function to initialize the floating, character and integer
        header arrays (self.hf, self.hs, self.hi) with dummy values. This
        function is usefull for writing sac files from artificial data,
        thus the header arrays are not filled by a read method
        beforehand

        @return: Nothing
        """
        # The sac header has 70 floats, then 40 integers, then 192 bytes
        # in strings. Store them in array (an convert the char to a
        # list). That's a total of 632 bytes.
        #
        # allocate the array for header floats
        #self.hf = array.array('f',[-12345.0])*70
        self.hf = np.ndarray(70,dtype='<f4')
        self.hf[:] = -12345.0
        #
        # allocate the array for header ints
        #self.hi = array.array('i',[-12345])*40
        self.hi = np.ndarray(40,dtype='<i4')
        self.hi[:] = -12345
        #
        # allocate the array for header characters
        self.hs = np.ndarray(24,dtype='|S8')
        self.hs[:] = '-12345   ' # setting default value
        # allocate the array for the points
        self.seis = np.ndarray([],dtype='<f4')


    def fromarray(self,trace,begin=0.0,delta=1.0,distkm=0):
        """create a sac-file from an array.array instance
        >>> t=ReadSac()
        >>> b = np.arange(10)
        >>> t.fromarray(b)
        >>> t.GetHvalue('npts')
        10
        """
        if not isinstance(trace,np.ndarray):
            raise SacError("input needs to be of instance numpy.ndarray")
        else:
            self.seis = copy.copy(trace)
            self.seis = np.require(self.seis,'<f4')
        ### set a few values that are required to create a valid SAC-file
        self.SetHvalue('int1',2)
        self.SetHvalue('cmpaz',0)
        self.SetHvalue('cmpinc',0)
        self.SetHvalue('nvhdr',6)
        self.SetHvalue('leven',1)
        self.SetHvalue('lpspol',1)
        self.SetHvalue('lcalda',0)
        self.SetHvalue('nzyear',1970)
        self.SetHvalue('nzjday',1)
        self.SetHvalue('nzhour',0)
        self.SetHvalue('nzmin',0)
        self.SetHvalue('nzsec',0)
        self.SetHvalue('kcmpnm','Z')
        self.SetHvalue('evla',0)
        self.SetHvalue('evlo',0)
        self.SetHvalue('iftype',1)
        
        self.SetHvalue('npts',len(trace))
        self.SetHvalue('delta',delta)
        self.SetHvalue('b',begin)
        self.SetHvalue('dist',distkm)

        
    def GetHvalue(self,item):
        """Get a header value using the header arrays: GetHvalue("npts")
        Return value is 1 if no problems occurred, zero otherwise."""
        key = item.lower() # convert the item to lower case

        if self.fdict.has_key(key):
            index = self.fdict[key]
            return(self.hf[index])
        elif self.idict.has_key(key):
            index = self.idict[key]
            return(self.hi[index])
        elif self.sdict.has_key(key):
            index = self.sdict[key]
            #length = 8
            if index == 0:
                #myarray = self.hs[0:8]
                myarray = self.hs[0]
            elif index == 1:
                #myarray = self.hs[8:24]
                myarray = self.hs[1] + self.hs[2]
            else:
                #start = 8 + index*8  # the extra 8 is from item #2
                #end   = start + 8
                #myarray = self.hs[start:end]
                myarray = self.hs[index+1] # extra 1 is from item #2
            #return(myarray.tostring())
            return myarray
        else:
            raise SacError("Cannot find header entry for: ",item)
        

    def SetHvalue(self,item,value):
        """Set a header value using the header arrays: SetHvalue("npts",2048)
        >>> t = ReadSac()
        >>> t.SetHvalue("kstnm","spiff")
        >>> t.GetHvalue('kstnm')
        'spiff   '
        """
        key = item.lower() # convert the item to lower case
        #
        if self.fdict.has_key(key):
                index = self.fdict[key]
                self.hf[index] = float(value)
        elif self.idict.has_key(key):
                index = self.idict[key]
                self.hi[index] = int(value)
        elif self.sdict.has_key(key):
                index = self.sdict[key]
                value = '%-8s' % value
                if index == 0:
                        self.hs[0] = value
                elif index == 1:
                    value1 = '%-8s'%value[0:8]
                    value2 = '%-8s'%value[8:16]
                    self.hs[1] = value1
                    self.hs[2] = value2
                else:
                        self.hs[index+1] = value
        else:
            raise SacError("Cannot find header entry for: ",item)


    def IsSACfile(self, name, fsize=True, lenchk=False):
        """Test for a valid SAC file using arrays: IsSACfile(path)
        Return value is a one if valid, zero if not.
        """
        npts = self.GetHvalue('npts')
        if lenchk:
            if npts != len(self.seis):
                raise SacError("Number of points in header and length of trace inconsistent!")
        if fsize:
            st = os.stat(name) #file's size = st[6] 
            sizecheck = st[6] - (632 + 4 * npts)
            # size check info
            if sizecheck != 0:
                raise SacError("File-size and theoretical size inconsistent: %s\nCheck that header values are consistent with timeseries."%name)
        # get the SAC file version number
        version = self.GetHvalue('nvhdr')
        if version < 0 or version > 20:
            raise SacError("Unknown header version!")
        if self.GetHvalue('delta') <= 0:
            raise SacError("Delta < 0 is not a valid header entry!")
        

    def ReadSacHeader(self,fname):
        """\nRead a header value into the header arrays 
        The header is split into three arrays - floats, ints, and strings
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac()
        >>> t.ReadSacHeader(file)
        >>> t.GetHvalue('npts')
        100
        """
        
        #### check if file exists
        try:
            #### open the file
            f = open(fname,'r')
        except IOError:
            raise SacIOError("No such file:"+fname)
        else:
            try:
                #--------------------------------------------------------------
                # parse the header
                #
                # The sac header has 70 floats, then 40 integers, then 192 bytes
                #    in strings. Store them in array (an convert the char to a
                #    list). That's a total of 632 bytes.
                #--------------------------------------------------------------
                self.hf = np.fromfile(f,dtype='<f4',count=70)
                self.hi = np.fromfile(f,dtype='<i4',count=40)
                self.hs = np.fromfile(f,dtype='|S8', count=24)    # read in the char values
            except EOFError, e:
                self.hf = self.hi = self.hs = None
                f.close()
                raise SacIOError("Cannot read all header values: ",e)
            else:
                try:
                    self.IsSACfile(fname)
                except SacError, e:
                    try:
                        ### if it is not a valid SAC-file try with big endian byte order
                        f.seek(0,0)
                        self.hf = np.fromfile(f,dtype='>f4',count=70)
                        self.hi = np.fromfile(f,dtype='>i4',count=40)
                        self.hs = np.fromfile(f,dtype='|S8', count=24)    # read in the char values
                        self.IsSACfile(fname)
                        self.byteorder = 'big'
                    except SacError, e:
                        self.hf = self.hi = self.hs = None
                        f.close()
                        raise SacError(e)
                    else:
                        try:
                            self._get_date_()
                        except SacError:
                            pass


    def WriteSacHeader(self,fname):
        """\nWrite a header value to the disk 
        \tok = WriteSacHeader(thePath)
        The header is split into three arrays - floats, ints, and strings
        The "ok" value is one if no problems occurred, zero otherwise.\n
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac(file)
        >>> t.WriteSacBinary('test2.sac')
        >>> u = ReadSac()
        >>> u.ReadSacHeader('test2.sac')
        >>> u.SetHvalue('kstnm','spoff   ')
        >>> u.WriteSacHeader('test2.sac')
        >>> u.GetHvalueFromFile('test2.sac',"kstnm")
        'spoff   '
        >>> os.remove('test2.sac')
        """
        #--------------------------------------------------------------
        # open the file
        #
        try:
            os.path.exists(fname)
        except IOError:
            print "No such file:"+fname
        else:
            f = open(fname,'r+') # open file for modification
            f.seek(0,0) # set pointer to the file beginning
            try:
                # write the header
                self.hf.tofile(f)
                self.hi.tofile(f)
                self.hs.tofile(f)
            except Exception, e:
                f.close()
                raise SacError("Cannot write header to file: ",fname,'  ',e)


    def ReadSacFile(self,fname):
        """\nRead read in the header and data in a SAC file 
        The header is split into three arrays - floats, ints, and strings and the
        data points are returned in the array seis
        >>> t=ReadSac()
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t.ReadSacFile(file)
        >>> t.GetHvalue('npts')
        100
        """
        try:
            #### open the file
            f = open(fname,'rb')
        except IOError:
            raise SacIOError("No such file:"+fname)
        else:
            try:
                #--------------------------------------------------------------
                # parse the header
                #
                # The sac header has 70 floats, then 40 integers, then 192 bytes
                #    in strings. Store them in array (an convert the char to a
                #    list). That's a total of 632 bytes.
                #--------------------------------------------------------------
                self.hf = np.fromfile(f,dtype='<f4',count=70)
                self.hi = np.fromfile(f,dtype='<i4',count=40)
                self.hs = np.fromfile(f,dtype='|S8', count=24)    # read in the char values
            except EOFError, e:
                raise SacIOError("Cannot read any or no header values: ",e)
            else:
                ##### only continue if it is a SAC file
                try:
                    self.IsSACfile(fname)
                except SacError:
                    try:
                        ### if it is not a valid SAC-file try with big endian byte order
                        f.seek(0,0)
                        self.hf = np.fromfile(f,dtype='>f4',count=70)
                        self.hi = np.fromfile(f,dtype='>i4',count=40)
                        self.hs = np.fromfile(f,dtype='|S8', count=24)    # read in the char values
                        self.IsSACfile(fname)
                        self.byteorder = 'big'
                    except SacError, e:
                        raise SacError(e)
                #--------------------------------------------------------------
                # read in the seismogram points
                #--------------------------------------------------------------
                npts = self.hi[9]  # you just have to know it's in the 10th place
                #                  # actually, it's in the SAC manual
                try:
                    if self.byteorder == 'big':
                        self.seis = np.fromfile(f,dtype='>f4',count=npts)
                    else:
                        self.seis = np.fromfile(f,dtype='<f4',count=npts)
                except EOFError, e:
                    self.hf = self.hi = self.hs = self.seis = None
                    f.close()
                    raise SacIOError("Cannot read any or only some data points: ",e)
                else:
                    try:
                        self._get_date_()
                    except SacError:
                        pass



    def ReadSacXY(self,fname):
        """\nRead SAC XY files (ascii)
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','testxy.sac')
        >>> t = ReadSac(file,alpha=True)
        >>> t.GetHvalue('npts')
        100
        >>> t.WriteSacBinary('testbin.sac')
        >>> os.path.exists('testbin.sac')
        True
        >>> os.remove('testbin.sac')
        """
        ###### open the file
        try:
            f = open(fname,'r')
        except IOError:
            raise SacIOError("No such file:"+fname)
        else:
            try:
                #--------------------------------------------------------------
                # parse the header
                #
                # The sac header has 70 floats, then 40 integers, then 192 bytes
                #    in strings. Store them in array (an convert the char to a
                #    list). That's a total of 632 bytes.
                #--------------------------------------------------------------
                # read in the float values
                self.hf = np.fromfile(f,dtype='<f4',count=70,sep=" ")
                # read in the int values
                self.hi = np.fromfile(f,dtype='<i4',count=40,sep=" ")
                # reading in the string part is a bit more complicated
                # because every string field has to be 8 characters long
                # apart from the second field which is 16 characters long
                # resulting in a total length of 192 characters
                for i in xrange(0,24,3):
                    self.hs[i:i+3] = np.fromfile(f,dtype='|S8',count=3)
                    f.readline() # strip the newline
                #--------------------------------------------------------------
                # read in the seismogram points
                #--------------------------------------------------------------
                self.seis = np.loadtxt(f,dtype='<f4').ravel()
            except IOError, e:
                self.hf = self.hs = self.hi = self.seis = None
                f.close()
                raise SacIOError("%s is not a valid SAC file:"%fname, e)
            try:
                self.IsSACfile(fname,fsize=False,lenchk=True)
            except SacError,e:
                f.close()
                raise SacError(e)
            else:
                try:
                    self._get_date_()
                except SacError:
                    pass


    def WriteSacXY(self,ofname):
        """ Write SAC XY file (ascii)
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac(file)
        >>> t.WriteSacXY('tmp3.sac')
        >>> d = ReadSac('tmp3.sac',alpha=True)
        >>> d.WriteSacBinary('tmp4.sac')
        >>> os.stat('tmp4.sac')[6] == os.stat(file)[6]
        True
        >>> os.remove('tmp3.sac')
        >>> os.remove('tmp4.sac')
        """
        try:
            f = open(ofname,'w')
        except IOError:
            raise SacIOError("Can't open file:"+ofname)
        else:
            try:
                np.savetxt(f,np.reshape(self.hf,(14,5)),fmt="%-8.6g %-8.6g %-8.6g %-8.6g %-8.6g")
                np.savetxt(f,np.reshape(self.hi,(8,5)),fmt="%-8.6g %-8.6g %-8.6g %-8.6g %-8.6g")
                for i in xrange(0,24,3):
                    self.hs[i:i+3].tofile(f)
                    f.write('\n')
            except:
                raise SacIOError("Can't write header values:"+ofname)
            else:
                try:
                    npts = self.GetHvalue('npts')
                    rows = npts/5
                    np.savetxt(f,np.reshape(self.seis[0:5*rows],(rows,5)),fmt="%15.7g\t%15.7g\t%15.7g\t%15.7g\t%15.7g")
                    np.savetxt(f,self.seis[5*rows:],delimiter='\t')
                except:
                    raise SacIOError("Can't write trace values:"+ofname)


    def WriteSacBinary(self,ofname):
        """\nWrite a SAC file using the head arrays and array seis 
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t=ReadSac(file)
        >>> t.WriteSacBinary('test2.sac')
        >>> os.stat('test2.sac')[6] == os.stat(file)[6]
        True
        >>> os.remove('test2.sac')
        """
        try:
            f = open(ofname,'wb+')
        except IOError:
            raise SacIOError("Cannot open file: ",ofname)
        else:
            try:
                self._chck_header_()
                self.hf.tofile(f)
                self.hi.tofile(f)
                self.hs.tofile(f)
                self.seis.tofile(f)
            except Exception, e:
                f.close()
                raise SacIOError("Cannot write SAC-buffer to file: ",ofname,e)

        
    def PrintIValue(self, label='=', value=-12345):
        """Convenience function for printing undefined integer header values"""
        if value != -12345:
            print label, value


    def PrintFValue(self, label='=', value=-12345.0):
        """Convenience function for printing undefined float header values"""
        if value != -12345.0:
            print '%s %.8g' % (label, value)


    def PrintSValue(self, label='=', value='-12345'):
        """Convenience function for printing undefined string header values"""
        if value != '-12345':
            print label, value


    def ListStdValues(self): # h is a header list, s is a float list
        """ Convenience function for printing common header values
        ListStdValues()"""
        #
        # Seismogram Info:
        #
        nzyear = self.GetHvalue('nzyear')
        nzjday = self.GetHvalue('nzjday')
        month = time.strptime(`nzyear`+" "+`nzjday`,"%Y %j").tm_mon
        date = time.strptime(`nzyear`+" "+`nzjday`,"%Y %j").tm_mday
        print '%s %2.2d/%2.2d/%d (%d) %d:%d:%d.%d' % ('\nReference Time = ',    \
                                                      month, date, \
                                                      self.GetHvalue('nzyear'), \
                                                      self.GetHvalue('nzjday'), \
                                                      self.GetHvalue('nzhour'), \
                                                      self.GetHvalue('nzmin'),  \
                                                      self.GetHvalue('nzsec'),  \
                                                      self.GetHvalue('nzmsec'))
        self.PrintIValue('Npts  = ',self.GetHvalue('npts'))
        self.PrintFValue('Delta = ',  self.GetHvalue('delta')  )
        self.PrintFValue('Begin = ',  self.GetHvalue('b')  )
        self.PrintFValue('End   = ',  self.GetHvalue('e')  )
        self.PrintFValue('Min   = ',  self.GetHvalue('depmin')  )
        self.PrintFValue('Mean  = ',  self.GetHvalue('depmen')  )
        self.PrintFValue('Max   = ',  self.GetHvalue('depmax')  )
        #
        self.PrintIValue('Header Version = ',self.GetHvalue('nvhdr'))
        #
        # station Info:
        #
        self.PrintSValue('Station = ',     self.GetHvalue('kstnm'))
        self.PrintSValue('Channel = ',     self.GetHvalue('kcmpnm'))
        self.PrintFValue('Station Lat  = ',self.GetHvalue('stla'))
        self.PrintFValue('Station Lon  = ',self.GetHvalue('stlo'))
        self.PrintFValue('Station Elev = ',self.GetHvalue('stel'))
        #
        # Event Info:
        #
        self.PrintSValue('Event       = ',self.GetHvalue('kevnm'))
        self.PrintFValue('Event Lat   = ',self.GetHvalue('evla'))
        self.PrintFValue('Event Lon   = ',self.GetHvalue('evlo'))
        self.PrintFValue('Event Depth = ',self.GetHvalue('evdp'))
        self.PrintFValue('Origin Time = ',self.GetHvalue('o'))
        #
        self.PrintFValue('Azimuth        = ',self.GetHvalue('az'))
        self.PrintFValue('Back Azimuth   = ',self.GetHvalue('baz'))
        self.PrintFValue('Distance (km)  = ',self.GetHvalue('dist'))
        self.PrintFValue('Distance (deg) = ',self.GetHvalue('gcarc'))
        


    def GetHvalueFromFile(self, thePath,theItem):
        """\nQuick access to a specific header item in a specified file.
        GetHvalueFromFile(thePath,theItem)
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac(file)
        >>> t.WriteSacBinary('test2.sac')
        >>> u = ReadSac()
        >>> u.SetHvalueInFile('test2.sac','kstnm','heinz   ')
        >>> u.GetHvalueFromFile('test2.sac','kstnm')
        'heinz   '
        >>> os.remove('test2.sac')
        """
        #
        #  Read in the Header
        #
        self.ReadSacHeader(thePath)
        #
        return(self.GetHvalue(theItem))


    def SetHvalueInFile(self, thePath,theItem,theValue):
        """\nQuick access to change a specific header item in a specified file.
        SetHvalueFromFile(thePath,theItem, theValue)
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac(file)
        >>> t.WriteSacBinary('test2.sac')
        >>> u = ReadSac()
        >>> u.SetHvalueInFile('test2.sac','kstnm','heinz   ')
        >>> u.GetHvalueFromFile('test2.sac','kstnm')
        'heinz   '
        >>> os.remove('test2.sac')
        """
        #
        #  Read in the Header
        #
        self.ReadSacHeader(thePath)
        #
        self.SetHvalue(theItem,theValue)
        self.WriteSacHeader(thePath)        


    def IsValidSacFile(self, thePath):
        """\nQuick test for a valid SAC binary file file.
        IsValidSACFile(thePath)
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        #
        #  Read in the Header
        #
        self.ReadSacHeader(thePath)
        #
        self.IsSACfile(thePath)



    def _get_date_(self):
        """if date header values are set calculate date in julian seconds
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> t = ReadSac(file)
        >>> t.starttime.timestamp
        269596800.0
        >>> t.endtime.timestamp - t.starttime.timestamp
        100.0
        """
        ### if any of the time-header values are still set to
        ### -12345 then UTCDateTime raises an exception and
        ### starttime is set to 0.0
        try:
            self.starttime = UTCDateTime(year=self.GetHvalue('nzyear'),
                                         julday=self.GetHvalue('nzjday'),
                                         hour=self.GetHvalue('nzhour'),
                                         minute=self.GetHvalue('nzmin'),
                                         second=self.GetHvalue('nzsec'),
                                         microsecond=self.GetHvalue('nzmsec') * 1000)
            self.endtime = self.starttime + \
                           self.GetHvalue('npts')*float(self.GetHvalue('delta'))
        except:
            try:
                self.starttime = UTCDateTime(0.0)
                self.endtime = self.starttime + \
                               self.GetHvalue('npts')*float(self.GetHvalue('delta'))
            except:
                raise SacError("Cannot calculate date")


    def _chck_header_(self):
        """
        if trace changed since read, adapt header values
        """
        self.seis = np.require(self.seis,'<f4')
        self.SetHvalue('npts',len(self.seis))
        self.SetHvalue('depmin',self.seis.min())
        self.SetHvalue('depmax',self.seis.max())
        self.SetHvalue('depmen',self.seis.mean())


    def swap_byte_order(self):
        """
        swap byte order of SAC-file in memory
        >>> file = os.path.join(os.path.dirname(__file__),'tests','data','test.sac')
        >>> fileswap = os.path.join(os.path.dirname(__file__),'tests','data','test.sac.swap')
        >>> x = ReadSac(fileswap)
        >>> x.swap_byte_order()
        >>> x.WriteSacBinary('tmp_swap.sac')
        >>> tr1 = ReadSac('tmp_swap.sac')
        >>> tr2 = ReadSac(file)
        >>> np.testing.assert_array_equal(tr1.seis,tr2.seis)
        >>> os.remove('tmp_swap.sac')
        """

        if self.byteorder == 'big':
            bs = 'L'
        elif self.byteorder == 'little':
            bs = 'B'
        self.seis.byteswap(True)
        self.hf.byteswap(True)
        self.hi.byteswap(True)
        self.seis = self.seis.newbyteorder(bs)
        self.hf   = self.hf.newbyteorder(bs)
        self.hi   = self.hi.newbyteorder(bs)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import os.path
    if os.path.isfile('test2.sac'):
        os.remove('test2.sac')
        
