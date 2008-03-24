#!/usr/bin/env python2.5
import os, sys, re, time
import array as myarray
from glob import glob
from stalta import ext_recstalta
#from matplotlib import *
from numpy import *

def readbin(binfile):
	regex = re.compile('3cssan.brhbw.([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})([0-9]{2})\.1\.([A-z]*)\.bin')
	samp_rate = 200
	maxndat = 2000000 #arbitray, must be longer then file length!
	
	match = regex.search(binfile) # if not pass
	header = {}
	end_of_file_flag = False
	
	# fill the binary array and transform to numpy array
	arr = myarray.array('f')
	ifile = open(binfile, 'rb')
	try: 
		arr.fromfile(ifile,maxndat)
	except EOFError:
		end_of_file_flag = True
		pass
	ifile.close()
	data = array(arr[1::2],dtype=float64)
	time = array(arr[0::2],dtype=float64)
	# data = array(arr,dtype=float32).reshape(len(arr)/2,2)
	
	# check if ndat is big enough to reach the end of file
	if not end_of_file_flag:
		print "Error: Didn't reach end of %s, tried %i numbers" % (binfile,maxndat)
		sys.exit(1)
	
	# fill the header
	header['d_year'] = int(match.group(1))
	header['d_mon'] =  int(match.group(2))
	header['d_day'] =  int(match.group(3))
	header['t_hour'] = int(match.group(4))
	header['t_min'] =  int(match.group(5))
	header['t_sec'] =  int(match.group(6))
	header['station'] = match.group(7)
	header['n_samps'] = len(data)
	header['samp_rate'] = samp_rate
	
	return header, data, time

def recursiveStaLta(a,Nsta,Nlta):
	"""Recursive STA/LTA (see Withers et al. 1998 p. 98)
	THERE IS A SQUARED MISSING IN HIS FORMULA, I ADDED IT"""
	m=len(a)
	#
	# compute the short time average (STA) and long time average (LTA)
	# given by Evans and Allen
	sta=zeros(len(a),dtype=double)
	lta=zeros(len(a),dtype=double)
	#Csta = 1-exp(-S/Nsta); Clta = 1-exp(-S/Nlta)
	Csta = 1./Nsta; Clta = 1./Nlta
	for i in range(1,m):
		# THERE IS A SQUARED MISSING IN THE FORMULA, I ADDED IT
		sta[i]=Csta*a[i]**2 + (1-Csta)*sta[i-1]
		lta[i]=Clta*a[i]**2 + (1-Clta)*lta[i-1]
	sta[0:Nlta]=0
	return sta/lta

def print_pick(header,starttot,stoptot):
		start_min = int(starttot/60)
		start_sec = starttot - start_min*60
		stop_min  = int(stoptot/60)
		stop_sec  = stoptot  - stop_min*60
		print "%s %06d %04d%02d%02d %02d:%02d:%02d %02d:%05.2f - %02d:%05.2f [=%05.2f-%05.2f]" % (
			header['station'],
			header['n_samps'],
			header['d_year'],
			header['d_mon'],
			header['d_day'],
			header['t_hour'],
			header['t_min'],
			header['t_sec'],
			start_min,
			start_sec,
			stop_min,
			stop_sec,
			starttot,
			stoptot)

def trigger_onset(stalta,time_trace,header,thres1,thres2):
	try: 
		on = where(stalta>thres1)[0]
		ind = min(on)
		of = ind + where(stalta[ind:]<thres2)[0]
	except ValueError:
		return True
	#
	start = stop = indstart = indstop = 0
	while True:
		try: 
			buf = start
			indstart = min(where(on[indstart:]>stop)[0]) + indstart
			start = on[indstart]
			#
			indstop = min(where(of[indstop:]>start)[0]) + indstop
			stop = stop_print = of[indstop]
		except ValueError: 
			if not (start == buf) and not (start == 0):
				print_pick(header,time_trace[start],time_trace[len(stalta)-1])
			break
		#
		print_pick(header,time_trace[start],time_trace[stop])

#
# main program
#
clock = time.clock()
dir = "/import/hochfelln-data/beyreuth/classification/par_data"
locevents = (
	"3cssan.brhbw.20040614220000.1.RMOA.bin",
	#"3cssan.brhbw.20040614170000.1.RMOA.bin",
	"3cssan.brhbw.20040614000000.1.RMOA.bin",
	#"3cssan.brhbw.20040613180000.1.RMOA.bin",
	"3cssan.brhbw.20040612040000.1.RMOA.bin",
	"3cssan.brhbw.20040609200000.1.RMOA.bin",
	#"3cssan.brhbw.20040607170000.1.RMOA.bin",
	"3cssan.brhbw.20040604210000.1.RMOA.bin")

#for binfile in locevents:
#	binfile = os.path.join(dir,binfile)
for binfile in glob("%s/3cssan.brhbw.*.bin" % (dir)):
	print binfile
	samp_rate = 200
	# 3 60
	sta_winlen=int(1. * samp_rate)
	lta_winlen=int(10. * samp_rate)
	thres1 = 4.
	thres2 = 1.5
	pre_event = 0.5 * samp_rate
	post_event = 2.5 * samp_rate
	mindur = 2

	#clock = time.clock()
	header, a, time_trace = readbin(binfile)
	#print "Reading time:", time.clock() - clock, "s"; clock = time.clock()

	# staltapy = recursiveStaLta(a,sta_winlen,lta_winlen)
	stalta = ext_recstalta.rec_stalta(a,sta_winlen,lta_winlen)
	#print "Extern StaLta time:", time.clock() - clock, "s"; clock = time.clock()

	trigger_onset(stalta,time_trace,header,thres1,thres2)
	#print "MinMax time:", time.clock() - clock, "s"

	if False:
		figure(1)
		hold(False)
		ax = subplot(2,1,1)
		plot(time_trace,a)
		subplot(2,1,2,sharex=ax)
		plot(time_trace,stalta)
		hold(True)
		title(binfile)
		hold(False)
		show()
		raw_input("Hit enter to continue")


	del stalta, a, time_trace, header
print "Time:", time.clock() - clock, "s"
