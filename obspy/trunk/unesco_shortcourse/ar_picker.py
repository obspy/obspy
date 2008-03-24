#!/usr/bin/env python

from pylab import *
from numpy import *
from modules.ar_picker.ext_arpicker import arPick
from modules.trigger import minsec, pick

def main(a,b,c,sample_rate,f1,f2,lta_p,sta_p,lta_s,sta_s,m_p,m_s,l_p,l_s):
	""" Program to plot seismogram and corresponding picks of the AR picker
	USAGE: main(a,b,c,sample_rate,f1,f2,lta_p,sta_p,lta_s,sta_s,m_p,m_s,l_p,l_s)
		a           : Z signal of float32 point data
		b           : N signal of float32 point data
		c           : E signal of float32 point data
		sample_rate : no of samples per second
		f1          : frequency of lower Bandpass window
		f2          : frequency of upper Bandpass window
		lta_p       : length of LTA for parrival in seconds
		sta_p       : length of STA for parrival in seconds
		lta_s       : length of LTA for sarrival in seconds
		sta_s       : length of STA for sarrival in seconds
		m_p         : number of AR coefficients for parrival
		m_s         : number of AR coefficients for sarrival
		l_p         : length of variance window for parrival in seconds
		l_s         : length of variance window for sarrival in seconds
	
	EXAMPLE:
	a=fromfile("loc_RJOB20050801145719850.z",sep="\\n",dtype=float32)
	b=fromfile("loc_RJOB20050801145719850.n",sep="\\n",dtype=float32)
	c=fromfile("loc_RJOB20050801145719850.e",sep="\\n",dtype=float32)
	main(a,b,c,200,1.0,20.0,1.0,0.1,4,1,2,8,0.1,0.2)
	"""

	try:
		filename_Z
		filename_N
		filename_E
	except NameError: filename_Z = "Interactive Mode"

	[ptime,stime] = arPick(a,b,c,sample_rate,f1,f2,lta_p,sta_p,lta_s,sta_s,m_p,m_s,l_p,l_s)
	t = arange(len(a),dtype=float)/sample_rate
	
	font = {'fontname'   : 'Times',
					'color'      : 'k',
					'fontweight' : 'bold',
					'fontsize'   : 14}
	
	clf()
	#hold(True)
	ax = subplot(311)
	title('AR Picker\nFile:'+filename_Z,font)
	#ylabel("Z Signal",font)
	ax.fmt_xdata = minsec
	plot(t,a,'k')
	# Plot phase pick pptime
	plot([ptime,ptime],array(ylim())*1/4,color='b',linewidth=1)
	text(ptime,ylim()[1]*1/4,'P',color='b',fontsize=12)
	
	ax2=subplot(312, sharex=ax)
	#ylabel("N Signal",font)
	ax2.fmt_xdata = minsec
	plot(t,b,'k')
	plot([stime,stime],array(ylim())*1/4,color='r',linewidth=1)
	text(stime,ylim()[1]*1/4,'S',color='r',fontsize=12)
	
	ax3=subplot(313, sharex=ax)
	#ylabel("E Signal",font)
	ax3.fmt_xdata = minsec
	plot(t,c,'k')
	plot([stime,stime],array(ylim())*1/4,color='r',linewidth=1)
	text(stime,ylim()[1]*1/4,'S',color='r',fontsize=12)
	
	connect('key_press_event', pick)
	show()


if __name__ == '__main__':
	#
	# securly asking input arguments
	#
	try:
		filename_Z = raw_input("""Give name of 1 coloumn ascii file containing Z signal : \n""")
		a=fromfile(filename_Z,sep="\n",dtype=float32)
		if len(a) == 0: raise ValueError
		filename_N = raw_input("""Give name of 1 coloumn ascii file containing N signal : \n""")
		b=fromfile(filename_N,sep="\n",dtype=float32)
		if len(b) == 0: raise ValueError
		filename_E = raw_input("""Give name of 1 coloumn ascii file containing E signal : \n""")
		c=fromfile(filename_E,sep="\n",dtype=float32)
		if len(c) == 0: raise ValueError
	except IOError:
		print """ERROR: Can't open File"""
		sys.exit(1)
	except ValueError:
		print "ERROR: Can't read signal from file %s.\n\
		Does the file illegaly contain non numbers or comments?" % (filename)
		sys.exit(1)
	
	try:
		sample_rate = float(raw_input("Give no of samples per second \n"))
		f1          = float(raw_input("Give frequency of lower Bandpass window \n"))
		f2          = float(raw_input("Give frequency of upper Bandpass window \n"))
		lta_p       = float(raw_input("Give length of LTA for parrival in seconds \n"))
		sta_p       = float(raw_input("Give length of STA for parrival in seconds \n"))
		lta_s       = float(raw_input("Give length of LTA for sarrival in seconds \n"))
		sta_s       = float(raw_input("Give length of STA for sarrival in seconds \n"))
		m_p         =   int(raw_input("Give number of AR coefficients for parrival \n"))
		m_s         =   int(raw_input("Give number of AR coefficients for sarrival \n"))
		l_p         = float(raw_input("Give length of variance window for parrival in seconds\n"))
		l_s         = float(raw_input("Give length of variance window for sarrival in seconds\n"));
	except ValueError:
		print "ERROR: Given input is no number!"
		sys.exit(1)

	main(a,b,c,sample_rate,f1,f2,lta_p,sta_p,lta_s,sta_s,m_p,m_s,l_p,l_s)

