#!/usr/bin/env python

from pylab import *
from numpy import *
import sys,os

def classicStaLta(a,Nsta,Nlta):
	"""Computes the standard STA/LTA from a given imput array a. The length of
	the STA is given by Nsta in samples, respectively is the length of the
	LTA given by Nlta in samples."""
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
	"""Delayed STA/LTA, (see Withers et al. 1998 p. 97)"""
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
	THERE IS A SQUARED MISSING IN HIS FORMULA, I ADDED IT"""
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
	"""Z-detector, (see Withers et al. 1998 p. 99)"""
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
	Z=(sta-a_mean)/a_std
	return Z

def minsec(x):
	return '%6.3fs' % (x)

def pick(event):
	if event.key=='p' and event.inaxes is not None:
		print event.xdata, event.ydata
	return None

def onSet(a,thres1,thres2,pre_sample,post_sample):
	try: 
		start = min(where(a>thres1)[0])
	except ValueError:
		start = len(a)
	try: 
		stop = min(where(a[start:] <thres2)[0]) + start
	except ValueError:
		stop = len(a)
	return [start,stop,start - pre_sample,stop + post_sample]
	
def plotTimes(start,stop,cut1,cut2,ylimit):
	plot([start,start],ylimit*1/2,color='r',linewidth=1)
	text(start,ylimit[1]*1/2,'start',color='r',fontsize=11)
	plot([stop,stop],ylimit*1/2,color='r',linewidth=1)
	text(stop,ylimit[1]*1/2,'stop',color='r',fontsize=11)
	plot([cut1,cut1],ylimit*1/4,color='b',linewidth=1)
	text(cut1,ylimit[1]*1/4,'cut1',color='b',fontsize=11)
	plot([cut2,cut2],ylimit*1/4,color='b',linewidth=1)
	text(cut2,ylimit[1]*1/4,'cut2',color='b',fontsize=11)
	plot([cut1,cut2],[ylimit[0]*1/4,ylimit[0]*1/4],color='b',linewidth=1)
	return None

def main(a,kind,sample_rate,sta,lta,thres1,thres2,pre,post):
	"""Program that plots trigger values and the corresponding characteristic
	function
	USAGE:  main(a,kind,sample_rate,sta,lta,thres1,thres2,pre,post)

	a            : signal of float32 point data
	sta          : sta window length in seconds
	lta          : lta window length in seconds
	kind         : one of 'classic', 'delayed', 'recursive', 'zdetect'
	sample_rate  : no of samples per second
	thrshl1      : threshold1 to trigger
	thrshl2      : threshold2 to trigger

	EXAMPLE:
	a = fromfile("loc_RJOB20020325181100.ascii",sep='\\n',dtype=float32)
	main(a,'classic',200,.25,2.5,4.0,1.0,0.5,2.5)

	"""

	if   kind == 'classic'  : trigger = classicStaLta
	elif kind == 'delayed'  : trigger = delayedStaLta
	elif kind == 'recursive': trigger = recursiveStaLta
	elif kind == 'zdetect'  : trigger = zdetect
	else:
		print """No valid kind given, give one of classic delayed recursive
		zdetect"""
		sys.exit(1)


	try: filename
	except NameError: filename = "Interactive Mode"
	# conversion to samples
	nsta = int(0.5 + sta* sample_rate)
	nlta = int(0.5 + lta* sample_rate)
	npre = int(0.5 + pre* sample_rate)
	npost= int(0.5 + post*sample_rate)
	
	b = zeros(len(a))
	b = trigger(a,nsta,nlta)
	[nstart,nstop,ncut1,ncut2] = onSet(b,thres1,thres2,npre,npost)

	# reconversion to seconds
	start = float(nstart)/sample_rate
	stop = float(nstop)/sample_rate
	cut1 = float(ncut1)/sample_rate
	cut2 = float(ncut2)/sample_rate
	t=arange(len(a),dtype=double)/sample_rate
	
	#
	# The plotting part
	#
	font = {'fontname'   : 'Times',
					'color'      : 'k',
					'fontweight' : 'bold',
					'fontsize'   : 14}
	
	name = trigger.__name__
	clf()
	# the upper plot
	ax = subplot(211)
	title(name + ' Trigger\nFile:'+filename,font)
	plot(t,a,color='k')
	#ylabel("Signal",font) #else floating point exception
	ax.fmt_xdata = minsec
	setp(ax.get_xticklabels(), visible=False)
	plotTimes(start,stop,cut1,cut2,array(ylim()))
	# the downer plot
	ax2 = subplot(212, sharex=ax)
	plot(t,b,color='k')
	#ylabel(name,font) #else floating point exception
	xlabel("Time in [s]",font)
	ax2.fmt_xdata = minsec
	setp(ax2.get_xticklabels(), fontsize=12)
	plotTimes(start,stop,cut1,cut2,array(ylim()))
	# display picks when 'p' is pressed
	connect('key_press_event', pick)
	show()


if __name__ == '__main__':
	linkname = os.path.basename(sys.argv[0])
	if   linkname == 'classic_stalta.py'  : kind = 'classic'
	elif linkname == 'delayed_stalta.py'    : kind = 'delayed'
	elif linkname == 'recursive_stalta.py'  : kind = 'recursive'
	elif linkname == 'zdetect.py'           : kind = 'zdetect'
	else: 
		print """Program behavior is dependent on the program filename. ===>
		Create symbolic link from trigger.py to one of
		classic_stalta.py
		delayed_stalta.py
		recursive_stalta.py
		zdetect.py
		and then rerun the program under the linked name
		e.g ln -s trigger.py classical_stalta.py && ./classical_stalta.py"""
	#
	# securly asking input arguments
	#
	filename = raw_input("""Give name of 1 coloumn ascii signal file: \n""")
	try:
		a=fromfile(filename,sep="\n")
		if len(a) == 0: raise ValueError
	except IOError:
		print """ERROR: Can't open file %s""" % (filename)
		sys.exit(1)
	except ValueError:
		print "ERROR: Can't read signal from file %s.\n\
		Does the file illegaly contain non numbers or comments?" % (filename)
		sys.exit(1)
	
	try:
		sample_rate = float(raw_input("Give number of samples per seconds: \n"))
		sta       = float(raw_input("Give STA window length in seconds: \n"))
		if not linkname == 'zdetect.py':
			lta       = float(raw_input("Give LTA window length in seconds: \n"))
		else: lta = 0
		thres1    = float(raw_input("Give threshold 1: \n"))
		thres2    = float(raw_input("Give threshold 2: \n"))
		pre       = float(raw_input("Give pre event time in seconds: \n"))
		post      = float(raw_input("Give post event time in seconds: \n"))
	except ValueError:
		print "ERROR: Given input is no number!"
		sys.exit(1)

	main(a,kind,sample_rate,sta,lta,thres1,thres2,pre,post)
