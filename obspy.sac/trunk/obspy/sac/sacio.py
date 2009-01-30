""" An object-oriented version of C. J. Ammon's SAC I/O module.
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



The ReadSac class provides the following functions:

Reading:

    ReadSacFile       - read binary SAC file
    ReadXYSacFile     - read XY SAC file    
    ReadSacHeader     - read SAC header
    GetHvalue         - extract information from header

Writing:

    WriteSacHeader    - write SAC header
    WriteSacBinary    - write binary SAC file


Convenience:

    IsSACfile         - test if valid binary SAC file
    IsXYSACfile       - test if valid XY SAC file
    PrintIValue       - print integer header values
    PrintFValue       - print float header values
    PrintSValue       - print integer header values
    ListStdValues     - print common header values
    GetHvalueFromFile - access to specific header item in specified file
    SetHvalueInFile   - change specific header item in specified file
    IsValidSacFile    - test for valid binary SAC file (wraps 'IsSACfile')

date handling:

    is_leap_year      - decide whether leap year
    ndaysinyear       - calculate number of days in year
    doy               - calculate yearday
    monthday          - calculate day of month
    yd2seconds        - calculate number of seconds since 1970
    dt2seconds        - calculate number of seconds since 1970

#################### TESTS ########################################
    
>>> t=ReadSac()
>>> t.ReadSacFile('test.sac')
1
>>> t.GetHvalue('npts')
100
>>> t.SetHvalue("kstnm","spiff")
1
>>> t.WriteSacBinary('test2.sac')
1
>>> t.ReadSacHeader('test2.sac')
1
>>> t.SetHvalue("kstnm","spoff")
1
>>> t.WriteSacHeader('test2.sac')
1
>>> t.SetHvalueInFile('test2.sac',"kcmpnm",'Z')
1
>>> t.GetHvalueFromFile('test2.sac',"kcmpnm")
'Z       '
>>> t.IsValidSacFile('test2.sac')
1



"""    

import array,os,string
from sacutil import *


class ReadSac(PyTutil):
    """ Class for SAC file IO
    initialise with: t=ReadSac()"""


    def __init__(self):
        self.fdict = {'delta':0, 'depmin':1, 'depmax':2, 'scale':3,   \
                      'odelta':4, 'b':5, 'e':6, 'o':7, 'a':8, 't0':10,\
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


    def GetHvalue(self,item):
        """Get a header value using the header arrays: GetHvalue("npts")
        Return value is 1 if no problems occurred, zero otherwise."""
        key = string.lower(item) # convert the item to lower case

        if self.fdict.has_key(key):
            index = self.fdict[key]
            return(self.hf[index])
        elif self.idict.has_key(key):
            index = self.idict[key]
            return(self.hi[index])
        elif self.sdict.has_key(key):
            index = self.sdict[key]
            length = 8

            if index == 0:
                myarray = self.hs[0:8]
            elif index == 1:
                myarray = self.hs[8:24]
            else:
                start = 8 + index*8  # the extra 8 is from item #2
                end   = start + 8
                myarray = self.hs[start:end]
                return(myarray.tostring())
        else:
            return('NULL')

    def SetHvalue(self,item,value):
	"""Set a header value using the header arrays: SetHvalue("npts",2048)
	Return value is 1 if no problems occurred, zero otherwise."""
	#
	# it's trivial to search each dictionary with the key and return
	#   the value that matches the key
	#
	key = string.lower(item) # convert the item to lower case
	#
	ok = 0
	if self.fdict.has_key(key):
		index = self.fdict[key]
		self.hf[index] = float(value)
		ok = 1
	elif self.idict.has_key(key):
		index = self.idict[key]
		self.hi[index] = int(value)
		ok = 1
	elif self.sdict.has_key(key):
		index = self.sdict[key]
		vlen = len(value)
		if index == 0:
			if vlen > 8:
				vlen = 8
			for i in range(0,8):
				self.hs[i] = ' '
			for i in range(0,vlen):
				self.hs[i] = value[i]
		elif index == 1:
			start = 8
			if vlen > 16:
				vlen =16 
			for i in range(0,16):
				self.hs[i+start] = ' '
			for i in range(0,vlen):
				self.hs[i+start] = value[i]
		else:
			#
			# if you are here, then the index > 2
			#
			if vlen > 8:
				vlen = 8
			start  = 8 + index*8 
			for i in range(0,8):
				self.hs[i+start] = ' '
			for i in range(0,vlen):
				self.hs[i+start] = value[i]
		ok = 1
	
	return(ok)


    def IsSACfile(self,name):
        """Test for a valid SAC file using arrays: IsSACfile(path)
        Return value is a one if valid, zero if not.
        """
        #
        ok = 1
        #
        # size check info
        #
        npts = self.GetHvalue('npts')
        st = os.stat(name) #file's size = st[6] 
        sizecheck = st[6] - (632 + 4 * npts)
        #
        # get the SAC file version number
        #
        version = self.GetHvalue('nvhdr')
        #
        # if any of these conditions are true,
        #   the file is NOT a SAC file
        #
        if self.GetHvalue('delta') <= 0:
            ok = 0
        elif sizecheck != 0:
            ok = 0
        elif ((version < 0) or (version > 20) ):
            ok = 0
        return(ok)


    def IsXYSACfile(self,name):
	"""Test for a valid SAC file using arrays: IsSACfile(path)
	Return value is a one if valid, zero if not."""
	#
	ok = 1
	#
	# size check info
	#
	npts = self.GetHvalue('npts')
	st = os.stat(name) #file's size = st[6] 
	sizecheck = st[6] - (632 + 2 * 4 * npts)
	#
	# get the SAC file version number
	#
	version = self.GetHvalue('nvhdr',hf,hi,hs)
	#
	# if any of these conditions are true,
	#   the file is NOT a SAC file
	#
	if sizecheck != 0:
		ok = 0
	elif ((version < 0) or (version > 20) ):
		ok = 0
	#
	return(ok)


    def ReadSacHeader(self,fname):
	"""\nRead a header value into the header arrays 
        \tok = ReadSacHeader(thePath)
        The header is split into three arrays - floats, ints, and strings
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
	#
	self.hf = array.array('f') # allocate the array for header floats
	self.hi = array.array('l') # allocate the array for header ints
	self.hs = array.array('c') # allocate the array for header characters
	#--------------------------------------------------------------
	# open the file
	#
	try:
            f = open(fname,'r')
            #--------------------------------------------------------------
            # parse the header
            #
            # The sac header has 70 floats, then 40 integers, then 192 bytes
            #    in strings. Store them in array (an convert the char to a
            #    list). That's a total of 632 bytes.
            #--------------------------------------------------------------
            self.hf.fromfile(f,70)     # read in the float values
            self.hi.fromfile(f,40)     # read in the int values
            self.hs.fromfile(f,192)    # read in the char values
            #
            f.close()
            #
            ok = self.IsSACfile(fname)
            #
            #--------------------------------------------------------------
	except:
            # make sure we close the file
            ok = 0
            #if f.close() == 0: # zero means the file is open
            #	f.close()
        return(ok)


    def WriteSacHeader(self,fname):
        """\nWrite a header value to the disk 
        \tok = WriteSacHeader(thePath)
        The header is split into three arrays - floats, ints, and strings
        The "ok" value is one if no problems occurred, zero otherwise.\n
        """
        #--------------------------------------------------------------
        # open the file
        #
        ok = 1
        try:
            f = open(fname,'r+') # open file for modification
            f.seek(0,0) # set pointer to the file beginning
            # write the header
            self.hf.tofile(f)
            self.hi.tofile(f)
            self.hs.tofile(f)
            f.close()
            #
            #--------------------------------------------------------------
        except:
            # make sure we close the file
            #if f.closed == 0: # zero means the file is open
            #	f.close()
            ok = 0
        return(ok)


    def ReadSacFile(self,fname):
        """\nRead read in the header and data in a SAC file 
        \tok = ReadSacFile(thePath)
        The header is split into three arrays - floats, ints, and strings and the
        data points are returned in the array seis
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        self.seis = array.array('f') # allocate the array for the points
        self.hf = array.array('f') # allocate the array for header floats
        self.hi = array.array('l') # allocate the array for header ints
        self.hs = array.array('c') # allocate the array for header characters
        #--------------------------------------------------------------
        # open the file
        #
        try:
            f = open(fname,'rb')
            #--------------------------------------------------------------
            # parse the header
            #
            # The sac header has 70 floats, then 40 integers, then 192 bytes
            #    in strings. Store them in array (an convert the char to a
            #    list). That's a total of 632 bytes.
            #--------------------------------------------------------------
            self.hf.fromfile(f,70)     # read in the float values
            self.hi.fromfile(f,40)     # read in the int values
            self.hs.fromfile(f,192)    # read in the char values
            #
            # only continue if it is a SAC file
            #
            ok = self.IsSACfile(fname)
            if ok:
                #--------------------------------------------------------------
                # read in the seismogram points
                #--------------------------------------------------------------
                npts = self.hi[9]  # you just have to know it's in the 10th place
                #             # actually, it's in the SAC manual
                #
                mBytes = npts * 4
                #
                self.seis = array.array('f')
                self.seis.fromfile(f,npts) # the data are now in s
                f.close()
        except:
            ok = 0
            # make sure we close the file
    	    #if f.closed == 0: # zero means the file is open
            #	f.close()
        return ok


    def ReadXYSacFile(self,fname):
        """\nRead a SAC XY file (not tested much) 
        \tok = ReadSXYSacFile(thePath)
        The header is split into three arrays - floats, ints, and strings.
        The data are in two floating point arrays x and y.
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        #
        self.x = array.array('f')
        self.y = array.array('f')
        self.hf = array.array('f') # allocate the array for header floats
        self.hi = array.array('l') # allocate the array for header ints
        self.hs = array.array('c') # allocate the array for header characters
        #--------------------------------------------------------------
        # open the file
        #
        try:
            f = open(fname,'r')
            #--------------------------------------------------------------
            # parse the header
            #
            # The sac header has 70 floats, then 40 integers, then 192 bytes
            #    in strings. Store them in array (an convert the char to a
            #    list). That's a total of 632 bytes.
            #--------------------------------------------------------------
            self.hf.fromfile(f,70)     # read in the float values
            self.hi.fromfile(f,40)     # read in the int values
            self.hs.fromfile(f,192)
            #
            # only continue if it is a SAC file
            #
            #
            ok = self.IsXYSACfile(fname)
            if ok:
                #--------------------------------------------------------------
                # read in the seismogram points
                #--------------------------------------------------------------
                npts = self.hi[9]  # you just have to know it's in the 10th place
                #             # actually, it's in the SAC manual
                #
                mBytes = npts * 4
                #
                self.y.fromfile(f,npts) # the data are now in s
                self.x.fromfile(f,npts) # the data are now in s
                #
                f.close()
                #
                #--------------------------------------------------------------
        except:
            # make sure we close the file
            if f.closed == 0: # zero means the file is open
                f.close()
                ok = 0
        return(ok)


    def WriteSacBinary(self,ofname):
        """\nWrite a SAC file using the head arrays and array seis 
        \tok = WriteSacBinary(thePath)
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        try:
            f = open(ofname,'wb+')
            self.hf.tofile(f)
            self.hi.tofile(f)
            self.hs.tofile(f)
            self.seis.tofile(f)
            f.close()
            ok = 1
        except:
            print 'Error writing file ', ofname
            ok = 0
        return(ok)

        
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
        [month, date, ok] = self.monthdate(nzyear, nzjday)
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
        returns -12345 if a problem occurred.\n"""
        #
        #  Read in the Header
        #
        ok = self.ReadSacHeader(thePath)
        #
        if ok:
            return(self.GetHvalue(theItem))
        else:
            print "Problem in GetHvalueFromFile."
            return(-12345)


    def SetHvalueInFile(self, thePath,theItem,theValue):
        """\nQuick access to change a specific header item in a specified file.
        SetHvalueFromFile(thePath,theItem, theValue)
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        #
        ok = 0
        #
        #  Read in the Header
        #
        r_ok = self.ReadSacHeader(thePath)
        #
        if r_ok:
            before = self.GetHvalue(theItem)
            if before != 'NULL':
                c_ok = self.SetHvalue(theItem,theValue)
                if c_ok:
                    after = self.GetHvalue(theItem)
                    ok = self.WriteSacHeader(thePath)	
                    #
                    #print 'Changed ',before,' to ',after,' in ',thePath
                else:
                    print "Problem in SetHvalueInFile."
        return(ok)


    def IsValidSacFile(self, thePath):
        """\nQuick test for a valid SAC binary file file.
        IsValidSACFile(thePath)
        The "ok" value is one if no problems occurred, zero otherwise.\n"""
        #
        #  Read in the Header
        #
        ok = self.ReadSacHeader(thePath)
        #
        if ok:
            ok = self.IsSACfile(thePath)
        else:
            ok = 0
        return(ok)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import os.path
    if os.path.isfile('test2.sac'):
        os.remove('test2.sac')
