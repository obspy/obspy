from filter_bc import filter_bc
import matplotlib
matplotlib.rc('figure.subplot', hspace=.25, wspace=.25) #adjust subplot layout
matplotlib.rc('font', size=8) # adjust font size of plot
from matplotlib.pyplot import *
from numpy import *

npts=1026
dt=0.05

# Play around with boxcar filtering
fmax=1.0/(2*dt) # nyquist
f0 = float(raw_input('Give cut-off below Nyquist: fmax = %f Hz ' % fmax))
print 'f0 =', f0

# Uncomment from random points
y = random.rand(npts) - .5 # uniform random numbers, zero mean

# Spike at npts/2
#y = zeros(npts,dtype='float')
#y[npts/2] = 1

freq, H, y_filt = filter_bc(y,dt,f0)


#######################################################################################
# Plot the whole filtering process
#######################################################################################

close('all')

def adjust():
    ymin, ymax = ylim()
    ylim(ymin-.2, ymax+.2)

subplot(231)
plot(arange(0,npts)*dt,y)
adjust()
title('Original data')
xlabel('Time [s]')
ylabel('Amplitude')

subplot(232)
tmp = fft.rfft(y)
plot(freq[1:], abs(tmp[1:]))
adjust()
title('Amplitude spectrum')
xlabel('Frequency [Hz]')
ylabel('Arbitrary Amplitude')

subplot(233), 
plot(freq[1:], angle(tmp[1:]))
adjust()
title('Phase spectrum')
xlabel('Frequency [Hz]')
ylabel('Phase shift [rad]')

subplot(234)
plot(freq[1:], H[1:])
adjust()
title('The filter')
xlabel('Frequency [Hz]')
ylabel('Filter amplitude')

subplot(235)
plot(freq[1:], abs((tmp*H)[1:]) )
adjust()
title('The filtered spectrum')
xlabel('Time [s]')
ylabel('Frequency [Hz]')

subplot(236)
plot(arange(0,npts)*dt, y_filt)
adjust()
title('The filtered signal')
xlabel('Time [s]')
ylabel('Amplitude')

show()
