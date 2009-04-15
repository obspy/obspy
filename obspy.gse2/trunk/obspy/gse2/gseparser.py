#!/usr/bin/env python

from obspy import parser
from obspy.gse2 import libgse2
import numpy, os, time, sys

class GseParser(parser.Parser):
    """
    %s
    """ % (parser.__doc__)
    
    def __init__(self,file=False):
        if file:
            self.read(file)
    
    def read(self,gsefile):
        """Read seismogram from GSE2.0 file
        
        Header attributes are assigned as attributes, for example see:
        g = GseParser('test.gse')
        print g
        """
        try:
            os.path.exists(gsefile)
            # Old C wraped version:
            #(h,data) = ext_gse.read(gsefile)
            (h,data) = libgse2.read(gsefile)
        except IOError:
            assert 1, "No such file to write: " + gsefile
        #
        # define the h entries as attributes
        self.auxid = h['auxid']
        self.calib = h['calib']
        self.calper = h['calper']
        self.channel = h['channel']
        self.datatype = h['datatype']
        self.hang = h['hang']
        self.instype = h['instype']
        self.npts = h['n_samps']
        self.df = h['samp_rate']
        self.station = h['station']
        self.vang = h['vang']
        intsec = int(h['t_sec'])
        date = "%04d%02d%02d%02d%02d%02d" % (h['d_year'],h['d_mon'],h['d_day'],h['t_hour'],h['t_min'],intsec)
        self.julsec = self.date_to_julsec('%Y%m%d%H%M%S',date) + h['t_sec']-intsec
        #
        if not type(data) == list:
            self.trace = list(data)
        else:
            self.trace = data
    
    def write(self,gsefile):
        """Write seismogram to GSE2.0 file
        
        Necessary attribute for this operation is self.trace. Probably it's
        usefull to also fill self.df and self.julsec but anyway the
        remaining attributes are filled by defaults.
        
        >>> g = GseParser()
        >>> numpy.random.seed(815)
        >>> g.trace = numpy.random.random_integers(0,2**26,1000).tolist()
        >>> g.write("test.gse")
        0
        >>> h = GseParser("test.gse")
        >>> h.trace[0:5]
        [29627570, 66111839, 44843986, 32909075, 36408434]
        >>> sum(abs(numpy.array(h.trace)-numpy.array(g.trace)))
        0

        Here the raising of an exception is tested if the data exceed the
        maximim value 2^26
        >>> i = GseParser()
        >>> i.trace = [2**26+1]
        >>> i.write("testexcept.gse")
        Traceback (most recent call last):
        ...
        Exception: Compression Error, data must be less equal 2^26
        """
        # General attributes
        self.is_attr('trace',list,None,assertation=True)
        self.is_attr('df',float,200.)
        self.is_attr('station',str,'FUR',length=5)
        self.is_attr('channel',str,'SHZ',length=3)
        self.is_attr('julsec',float,0.0)
        self.is_attr('npts',int,len(self.trace))
        # GSE2 specific attributes
        self.is_attr('auxid',str,'VEL',length=4)
        self.is_attr('datatype',str,'CM6',length=3)
        self.is_attr('calib',float,1./(2*numpy.pi)) #calper not correct in gse_driver!
        self.is_attr('calper',float,1.)
        self.is_attr('instype',str,'LE-3D',length=6)
        self.is_attr('hang',float,-1.0)
        self.is_attr('vang',float,0.)
        
        intjulsec = int(self.julsec)
        (d_year,d_mon,d_day,t_hour,t_min,t_sec) = time.gmtime(intjulsec)[0:6]
        t_sec += self.julsec-intjulsec
        data = numpy.array(self.trace,dtype='l')

        # Maximum values above 2^26 will result in corrupted/wrong data!
        if data.max() > 2**26:
            raise Exception ,"Compression Error, data must be less equal 2^26"
        
        err = libgse2.write(
            {
            'd_year':d_year,
            'd_mon':d_mon,
            'd_day':d_day,
            't_hour':t_hour,
            't_min':t_min,
            't_sec':t_sec,
            'station':self.station,
            'channel':self.channel,
            'auxid':self.auxid,
            'datatype':self.datatype,
            'n_samps':self.npts,
            'samp_rate':self.df,
            'calib':self.calib,
            'calper':self.calper,
            'instype':self.instype,
            'hang':self.hang,
            'vang':self.vang
            }
            ,data,gsefile)
        # Old C wraped version
        #err = ext_gse.write((d_year, d_mon, d_day, t_hour, t_min, t_sec,
        #    self.station, self.channel, self.auxid, self.datatype, self.npts,
        #    self.df, self.calib, self.calper, self.instype, self.hang,
        #    self.vang), data, gsefile)
        return err


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
