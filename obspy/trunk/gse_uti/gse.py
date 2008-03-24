# Python gse module

from numpy import *
import os,sys
import gse_ext


def has_entry(header,key_,typ_,value,length=None):
	"""function for verifying that header has key_ of cetain type and, if
	given, certain length. If not, the header[key_] is set to value
	"""
	if not header.has_key(key_) or not isinstance (header[key_],typ_):
		print "WARNING: %s entry of header missing or not of %s" % (key_,typ_)
		print "forcing",key_,"=",value
		header[key_]=value
	if (length):
		if (len(header[key_]) > length):
			print "%s entry of header is > %i" % (key_,length)
			print "forcing",key_,"=",value
			header[key_]=value


def read(gsefile):
	"""NOTE: documentation is assigned after defenition by:
	read.__doc__ = gse_ext.read.__doc__"""
	try: 
		os.path.exists(gsefile)
		return gse_ext.read(gsefile)
	except IOError:
		print "No such file to write: " + gsefile
		sys.exit(2)
read.__doc__ = gse_ext.read.__doc__


def write(h,data,gsefile):
	"""write header h and data to gsefile

	write(header,data,gsefile)
	h    : tuple containing the header variables
	data      : LONG array containing the data to write
	gsefile   : target file to write
	"""
	# let's check if header has a the necessary tuples and if those are of
	# correct type
	has_entry(h,'d_year',int,2007)
	has_entry(h,'d_mon',int,11)
	has_entry(h,'d_day',int,22)
	has_entry(h,'t_hour',int,13)
	has_entry(h,'t_min',int,33)
	has_entry(h,'t_sec',float,24.123)
	has_entry(h,'station',str,'RTSH ',length=6)
	has_entry(h,'channel',str,'SHZ',length=4)
	has_entry(h,'auxid',str,'VEL ',length=5)
	has_entry(h,'datatype',str,'CM6 ',length=4)
	has_entry(h,'n_samps',int,len(data))
	has_entry(h,'samp_rate',float,62.5)
	has_entry(h,'calib',float,1./(2*pi)) #calper not correct in gse_driver!
	has_entry(h,'calper',float,1.)
	has_entry(h,'instype',str,'LE-3D ',length=7)
	has_entry(h,'hang',float,-1.0)
	has_entry(h,'vang',float,0.)

	# I have errors with the data pointer, only solution seems to explicitly copy it
	data2 = data.copy()
	err = gse_ext.write((h['d_year'], h['d_mon'], h['d_day'], h['t_hour'],
		h['t_min'], h['t_sec'], h['station'], h['channel'], h['auxid'],
		h['datatype'], h['n_samps'], h['samp_rate'], h['calib'], h['calper'],
		h['instype'], h['hang'], h['vang']), data2, gsefile)
	return err
