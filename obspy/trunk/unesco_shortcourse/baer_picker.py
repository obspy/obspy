#!/usr/bin/env python

from pylab import *
from numpy import *
from modules.pick_baer.ext_pk_mbaer import baerPick
from modules.trigger import minsec, pick

def main(a,sample_rate,tdownmax,tupevent,thrshl1,thrshl2,preset_len,p_dur):
	""" Program to plot seismogram and corresponding picks of the baer picker
	USAGE: main(a,sample_rate,tdownmax,tupevent,thrshl1,thrshl2,preset_len,p_dur)
		a           : signal of float32 point data
		sample_rate : no of samples per second
	 	tdownmax    : if time exceeds tdownmax, the trigger is examined for validity, give tdownmax in seconds
	 	tupevent    : min nr of seconds for itrm to be accepted as a pick
	 	thrshl1     : threshold to trigger for pick (c.f. paper)
	 	thrshl2     : threshold for updating sigma  (c.f. paper)
	 	preset_len  : window length in seconds for the estimation of variance of SF(t) on preset()
	 	p_dur       : p_dur defines the time interval for which the maximum amplitude is evaluated in seconds

	EXAMPLE:
	a=fromfile("loc_RJOB20020325181100.ascii",sep="\n",dtype=float32)
	main(a,200,0.1,0.3,7,12,0.5,0.5)"""

	try: filename
	except NameError: filename = "Interactive Mode"
	
	# conversion to samples
	ntdownmax    = int(0.5 +  tdownmax  *sample_rate)
	ntupevent    = int(0.5 +  tupevent  *sample_rate)
	npreset_len  = int(0.5 +  preset_len*sample_rate)
	np_dur       = int(0.5 +  p_dur     *sample_rate)
	
	[nptime,pfm] = baerPick(a,sample_rate,ntdownmax,ntupevent,thrshl1,thrshl2,npreset_len,np_dur)

	# reconversion to seconds
	ptime = float(nptime)/sample_rate
	t = arange(len(a),dtype=float)/200
	
	font = {'fontname'   : 'Times',
					'color'      : 'k',
					'fontweight' : 'bold',
					'fontsize'   : 14}
	
	clf()
	hold(True)
	ax = subplot(111)
	title('Baer Picker\nFile:'+filename,font)
	#ylabel("Signal",font)
	plot(t,a,'k')
	ax.fmt_xdata = minsec
	# Plot phase pick pptime
	plot([ptime,ptime],array(ylim())*1/4,color='b',linewidth=1)
	text(ptime,ylim()[1]*1/4,'$P_{%s}$'%pfm,color='b',fontsize=14)
	connect('key_press_event', pick)
	show()


if __name__ == '__main__':
	#
	# securly asking input arguments
	#
	try:
		filename = raw_input("""Give name of 1 coloumn ascii file containing signal : \n""")
		a=fromfile(filename,sep="\n",dtype=float32)
		if len(a) == 0: raise ValueError
	except IOError:
		print """ERROR: Can't open File"""
		sys.exit(1)
	except ValueError:
		print "ERROR: Can't read signal from file %s.\n\
		Does the file illegaly contain non numbers or comments?" % (filename)
		sys.exit(1)
	
	try:
		sample_rate = float(raw_input("Give no of samples per second \n"))
	 	tdownmax    = float(raw_input("If time exceeds tdownmax, the trigger is examined for validity, give tdownmax in seconds: \n"))
	 	tupevent    = float(raw_input("Give min nr of seconds for itrm to be accepted as a pick: \n"))
	 	thrshl1     = float(raw_input("Give threshold to trigger for pick (c.f. paper): \n"))
	 	thrshl2     = float(raw_input("Give threshold for updating sigma  (c.f. paper): \n"))
	 	preset_len  = float(raw_input("Give window length in seconds for the estimation of variance of SF(t) on preset(): \n"))
	 	p_dur       = float(raw_input("Give p_dur defines the time interval for which the maximum amplitude is evaluated in seconds: \n"))
	except ValueError:
		print "ERROR: Given input is no number!"
		sys.exit(1)

	main(a,sample_rate,tdownmax,tupevent,thrshl1,thrshl2,preset_len,p_dur)

