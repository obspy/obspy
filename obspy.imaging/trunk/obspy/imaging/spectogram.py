#!/scratch/seisoft/Python-2.5.2/bin/python

#computes and plots logarithmic spectogram of the input trace
#input in gse2 format
#needed specifications: sampling rate (Hz), number of steps, window length (both in samples)

import obspy,time
import glob, os
from pylab import *
from numpy import *

def spec(inp,sample_rate,samp_length,log):

# Draws a seismogram and a spectogram of the input data. Spectogram can be logarithmic in frequency if desired.
# Input parameters: inp - data in numpy array format
#										sample_rate - sampling frequency in Hz
#										samp_length - total length of the data trace (in samples)
#										log - set to True for logarithmic spectogram, False for normal spectogram
	
	
	samp_rate = 1/float(sample_rate)
	step_size = samp_length/80.
	samp_length_win = samp_length/10.

	nstep1 = pow(2,ceil(log2(step_size)))
	nstep2 = pow(2,floor(log2(step_size)))
	if abs(nstep1-step_size) < abs(nstep2-step_size):
		nstep = nstep1
	else:
		nstep = nstep2

	window_length_sample1 = pow(2,ceil(log2(samp_length_win)))
	window_length_sample2 = pow(2,floor(log2(samp_length_win)))
	if abs(window_length_sample1-samp_length_win) < abs(window_length_sample2-samp_length_win):
		window_length_sample = window_length_sample1
	else:
		window_length_sample = window_length_sample2

	hann = 0.5*(1-cos( 2*pi*arange(window_length_sample,dtype='f')/(window_length_sample-1)))

	data = inp - mean(inp)
	psd = zeros((int(window_length_sample/2),1))
	mx = max(data)
	mn = min(data)
	j = 0
	while (j + window_length_sample < samp_length):
		window = data[j:j+window_length_sample] * hann
		window = window - mean(window)
		window_fft = fft.fft(window)
		ampl_fft = (abs(window_fft[1:(len(window_fft)/2)+1]))**2
		ampl_fft = ampl_fft.reshape(-1,1)
		psd = concatenate((psd,ampl_fft),axis=1)
		j = j+nstep
		spectogram = 10*log10(psd[:,1:])

	shap = shape(spectogram)
	x = linspace(1,shap[1],shap[1])
	x = arange(0,shap[1]*(nstep*samp_rate),(nstep*samp_rate))
	upper = (1/samp_rate)/2
	lower = 1/(window_length_sample*samp_rate)
	far = shap[0]
	y = linspace(lower,upper,far)
	X,Y = meshgrid(x,y)
	subplot(212)
	pcolor(X,Y,spectogram)
	if log:
		semilogy()
	gap = (window_length_sample/2.)*samp_rate
	h = [-gap,(samp_length*samp_rate)-gap,lower,upper]
	axis(h)
	
	tick_dist = floor((samp_length*samp_rate)/6.)
	tick_big = 5*tick_dist
	ticks = arange(0,tick_big,tick_dist)
	xticks([0-gap,tick_dist-gap,(2*tick_dist)-gap,(3*tick_dist)-gap,(4*tick_dist)-gap,tick_big-gap],['0',tick_dist,2*tick_dist,3*tick_dist,4*tick_dist,tick_big] )
	xlabel('time (s)')
	ylabel('frequency (Hz)')
	subplot(211)
	v = [0,samp_length*samp_rate,min(data),max(data)]
	xx_list = []
	xx = range(0,len(data))
	for nn in xx:
		xx1 = nn*samp_rate
		xx_list.append(xx1)
	plot(xx_list,data)
	xlabel('time (s)')
	axis(v)
	xticks([0,tick_dist,2*tick_dist,3*tick_dist,4*tick_dist,tick_big])
	ylabel('amplitude (arbitrary)')
	savefig('spec.pdf')
	
