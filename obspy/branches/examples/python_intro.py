#####################
# PYTHON INTRO      #
#                   #
# 2009-11-27 Moritz #
#####################

# http://docs.python.org/tutorial/index.html
# http://docs.scipy.org/doc/
# http://www.obspy.org
# Starting ipython in a windows cmd consolue with -pylab option:
# ipython -pylab


#####################
# Basic Arithmetics #
#####################

2+5*7

5/3    #1, integer devision

5.0/3  #1.66666 correct

5./3+2

5./(3+2)

5./3**2

(5./3)**2

5.**18

5./0   #ZeroDivisionError

0./0   #ZeroDivisionError


############################
# Basic Data Types/Objects #
############################

# Intergers, floats, strings
a = 3
b = 3.56
c = 'Hello World'

# List
# Lists can contain all other data types, even functions list itselfs,...
# Index in Python starts with 0
d = []
e = [3, 4, 5]
f = ['hello','world','foo','bar']
e[0] #first index of list, here 3

# Tuples
# tuples are not oven used, similar to lists
x = (1,2,3,4)
x[0] #1

# Dictionary
# Dictionaries allow other datatypes than int as index.
g = {'Hello': 100, 'World': -50}
g['Hello'] #results in 100
g['World'] #results in -50

# Find out type
type(a)
type(g)

##########################
# Functions and Packages #
##########################

# Import module
x = 5
import numpy
math.exp(5)

# Import function from module
from numpy import exp
exp(5)

# Import as other name
import numpy as np
np.exp(5)

# Import all functions from module, not allowed on module level
from numpy import *

################
# Getting help #
################

# Help 
a = ['Hello', 'World']
help(a)

# Help on module
import numpy
help(math)
help(math.exp)

# List available methods in object or module
dir(math)
dir(a)

################
# Flow control #
################

# Python has no brackets, all is controlled by the indentation. Space or
# tabulator matters ==> use space
for i in range(5): #==[0,1,2,3,4]
    print i

if 'a' == 'a':
    print 'a=a'

try:
    z[0] = 'First index'
except IndexError:
    print "A list and index must be initialized before assigning it"

######################
# IPython essentials #
######################

#Starty ipython with -pylab option which
#preloads most numpy and plotting modules
# ipython -pylab

# Run Python program
run program.py

# Run Python program, in local namespace
run -i program.py

# Tab completions
import numpy
numpy.   #hit tab twice
numpy.ex #hit tab once

# Help
help math
math?  #same as help
math?? #show also source code

# History
#arrow up ---> show previous command in history
#arrow down -> show next command in history
num #arrow up -> show previous command starting with num
history # -> print the complete recorded history

# Basic Unix commands are mapped
ls  #list directory entries
mv file1 file2 #move file1 to file2
cp file1 file2 #copy file1 to file2

################
# 1 Dim Arrays #
################

# Arrays in python are efficiently handled with the 
# numpy module.
import numpy as np

# Allocating 1 Dim numpy array
N = 5 #size 5
x0 = np.arange(0, 10, 0.1) # sequence from 0 to 10 with step 0.1
x1 = np.zeros(N)
x2 = np.ones(N)
x3 = np.random.randn(N)
x4 = np.random.rand(N)
x5 = np.array([1,2,3,4]) # by converting from a list, inefficient

# Caution, type of arrays is important
y1 = np.ones(N, dtype='int32')
y3 = y1.copy()
y2 = np.ones(N, dtype='float64')
# For inplace multiplication, the type stays the same!
y1 *= 0.5
y2 *= 0.5
# For normal multiplication, the type changes
y3 = y3 * 0.5

# dot product
dot = np.dot(x3,x3)     #inner product
outer = np.outer(x3,x3) #outer product

# example of methods bound to the arrays
x0.max()
x0.min()
x0.std()
x0.mean()
x1 = x0.copy()
x0 -= x0.mean() # subtrace mean from e.g. time series

####################
# Multi Dim Arrays #
####################

# Allocating multi Dim numpy array
N = 5
M = 3
x5 = np.zeros((N,M))   # NxM array
x6 = np.zeros((N,M,N)) # NxMxN array
x7 = np.arange(15).reshape(N,M)

# dot product
dot = np.dot(x5.T, x5)  #.T == transpose
outer = np.dot(x5, x5.T)

# finding the shape
shape = x5.shape # shape[0] = N, shape[1] = M

# slicing (index start with 0!!)
x5[1,:] # second row
tind = np.array([0,1,2])
xind = np.array([1,2])
x5[tind,:] # first three rows
x5[:,xind] # second and third coloumn
tind = tind.reshape((-1,1))
xind = xind.reshape((1,-1))
x5[tind,xind] # second and third entry in the first three rows

###################
# Plotting Arrays #
###################

# * All 2D plotting routines are efficiently handled with the
#   matplotlib module
# * Gallery with different plots, click on them for seeing the source code
#   http://matplotlib.sourceforge.net/gallery.html
import matplotlib.pyplot as plt
import numpy as np

# Plot random series of y values
x3 = np.random.randn(N)
plt.figure()
plt.plot(x3)
plt.show() # you do not need this in an ipython shell

# Plot x agains y values
x4 = np.random.randn(N)
t = np.arange(len(x4)) #start value and interval are optional
plt.figure()
plt.plot(t, x4, 'r--') #red dotted lines
plt.plot(t, x3, 'gx-') #green lines with x markers in overlay
plt.show() #you do not need this in an ipython shell

# Clear current figure plot and plot histogram
plt.clf()
plt.hist(x4) #plot historgram

##############
# Seismology #
##############

# Basic seismology routines are already implemented in the 
# obspy module
from obspy.core import read

# Read in SAC, MSEED or GSE2
st = read("loc_RJOB20050831023349.z") #read in stream
tr = st[0] #first trace in stream, trace consits data block that is no gap
tr.stats #contains all the meta/header information
tr.stats.gse2 #contains gse2 specific meta/header
tr.data #contains data as numpy array, C contiguous memory layout

# Plotting
st.plot() # fast view, no control
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
npts, df = tr.stats.npts, tr.stats.sampling_rate
plt.plot( np.arange(npts)/df, tr.data)
plt.title(tr.stats.starttime)

# Write out SAC, MSEED or GSE2. Note only the header entries network,
# station, location, starttime, endtime, sampling_rate, npts, channel are converted!
st.write("myfile.sac", format='SAC') 
