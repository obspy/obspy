# obspy module

import os,sys
from numpy import zeros,mean,concatenate,std,pi
from scipy.signal import iirfilter,lfilter
import ext_gse
import ext_recstalta

from ext_recstalta import rec_stalta

from ext_pk_mbaer import baerPick

from ext_arpicker import arPick

def bandpass(data,freqmin,freqmax,df=200,corners=4):
	"""Butterworth-Bandpass: filter data from freqmin to freqmax using
	corners corners
	"""
	fe=.5*df
	[b,a]=iirfilter(corners, [freqmin/fe, freqmax/fe], btype='band',ftype='butter',output='ba')
	return lfilter(b, a, data)

def readgse(gsefile):
	"""NOTE: documentation is assigned AFTER definition by:
	read.__doc__ = gse_ext.read.__doc__
	"""
	try: 
		os.path.exists(gsefile)
		return ext_gse.read(gsefile)
	except IOError:
		print "No such file to write: " + gsefile
		sys.exit(2)
readgse.__doc__ = ext_gse.read.__doc__


def writegse(h,data,gsefile):
	"""write header h and data to gsefile, the definition of the header is
	given in documentation of the extern C function appended after this
	documentation. Defaults are set automatically

	write(header,data,gsefile)
	h            : tuple containing the header variables
	data         : LONG array containing the data to write
	gsefile      : target file to write
	----------------------------------------------------------------------
	"""
	# 
	# function for testing correctness of header entries
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
	#
	# let's check if header has a the necessary tuples and if those are of
	# correct type
	has_entry(h,'d_year',int,2007)
	has_entry(h,'d_mon',int,05)
	has_entry(h,'d_day',int,27)
	has_entry(h,'t_hour',int,23)
	has_entry(h,'t_min',int,59)
	has_entry(h,'t_sec',float,24.123)
	has_entry(h,'station',str,'STAU ',length=6)
	has_entry(h,'channel',str,'SHZ',length=4)
	has_entry(h,'auxid',str,'VEL ',length=5)
	has_entry(h,'datatype',str,'CM6 ',length=4)
	has_entry(h,'n_samps',int,len(data))
	has_entry(h,'samp_rate',float,200.)
	has_entry(h,'calib',float,1./(2*pi)) #calper not correct in gse_driver!
	has_entry(h,'calper',float,1.)
	has_entry(h,'instype',str,'LE-3D ',length=7)
	has_entry(h,'hang',float,-1.0)
	has_entry(h,'vang',float,0.)

	# I have errors with the data pointer, only solution seems to explicitly copy it
	data2 = data.copy()
	err = ext_gse.write((h['d_year'], h['d_mon'], h['d_day'], h['t_hour'],
		h['t_min'], h['t_sec'], h['station'], h['channel'], h['auxid'],
		h['datatype'], h['n_samps'], h['samp_rate'], h['calib'], h['calper'],
		h['instype'], h['hang'], h['vang']), data2, gsefile)
	del data2
	return err
writegse.__doc__ = writegse.__doc__ + ext_gse.write.__doc__

def classicStaLta(a,Nsta,Nlta):
	"""Computes the standard STA/LTA from a given imput array a. The length of
	the STA is given by Nsta in samples, respectively is the length of the
	LTA given by Nlta in samples.
	"""
	m=len(a)
	stalta=zeros(m,dtype=float)
	start = 0
	stop = 0
	#
	# compute the short time average (STA)
	sta=zeros(len(a),dtype=float)
	pad_sta=zeros(Nsta)
	for i in range(Nsta): # window size to smooth over
		sta=sta+concatenate((pad_sta,a[i:m-Nsta+i]**2))
	sta=sta/Nsta
	#
	# compute the long time average (LTA)
	lta=zeros(len(a),dtype=float)
	pad_lta=ones(Nlta) # avoid for 0 division 0/1=0
	for i in range(Nlta): # window size to smooth over
		lta=lta+concatenate((pad_lta,a[i:m-Nlta+i]**2))
	lta=lta/Nlta
	#
	# pad zeros of length Nlta to avoid overfit and
	# return STA/LTA ratio
	sta[0:Nlta]=0
	return sta/lta

def delayedStaLta(a,Nsta,Nlta):
	"""Delayed STA/LTA, (see Withers et al. 1998 p. 97)
	This functions returns the characteristic function of the delayes STA/LTA
	trigger. Nsta/Nlta is the length of the STA/LTA window in points
	respectively"""
	m=len(a)
	stalta=zeros(m,dtype=float)
	on = 0;
	start = 0;
	stop = 0;
	#
	# compute the short time average (STA) and long time average (LTA)
	# don't start for STA at Nsta because it's muted later anyway
	sta=zeros(len(a),dtype=float)
	lta=zeros(len(a),dtype=float)
	for i in range(Nlta+Nsta+1,m):
		sta[i]=(a[i]**2 + a[i-Nsta]**2)/Nsta + sta[i-1]
		lta[i]=(a[i-Nsta-1]**2 + a[i-Nsta-Nlta-1]**2)/Nlta + lta[i-1]
		sta[0:Nlta+Nsta+50]=0
	return sta/lta

def recursiveStaLta(a,Nsta,Nlta):
	"""Recursive STA/LTA (see Withers et al. 1998 p. 98)
	NOTE: There exists a version of this trigger wrapped in C called
	rec_stalta in this module!"""
	m=len(a)
	#
	# compute the short time average (STA) and long time average (LTA)
	# given by Evans and Allen
	sta=zeros(len(a),dtype=float)
	lta=zeros(len(a),dtype=float)
	#Csta = 1-exp(-S/Nsta); Clta = 1-exp(-S/Nlta)
	Csta = 1./Nsta; Clta = 1./Nlta
	for i in range(1,m):
		# THERE IS A SQUARED MISSING IN THE FORMULA, I ADDED IT
		sta[i]=Csta*a[i]**2 + (1-Csta)*sta[i-1]
		lta[i]=Clta*a[i]**2 + (1-Clta)*lta[i-1]
	sta[0:Nlta]=0
	return sta/lta

def zdetect(a,Nsta,number_dummy):
	"""Z-detector, (see Withers et al. 1998 p. 99)
	This functions returns the characteristic function of the Z-detector.
	Nsta gives the number of points for the sta window"""
	m=len(a)
	#
	# Z-detector given by Swindell and Snell (1977)
	Z=zeros(len(a),dtype=float)
	sta=zeros(len(a),dtype=float)
	# Standard Sta
	pad_sta=zeros(Nsta)
	for i in range(Nsta): # window size to smooth over
		sta=sta+concatenate((pad_sta,a[i:m-Nsta+i]**2))
	a_mean=mean(sta)
	a_std=std(sta)
	return (sta-a_mean)/a_std

