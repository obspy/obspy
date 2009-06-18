#!/usr/bin/python
"""Small script to test weather memory leakage occurs during reading of
gse2 data. Run the script and monitor the memory usage for a while."""

import obspy.gse2.gseparser as gse2
from numpy import *

rand = array(1000*random.random(1e7),dtype='l')
file = "leak.gse"

print __doc__
print rand.tolist()[0:10]
g = gse2.GseParser()
g.trace = rand.tolist()
g.write(file)
del g

while True:
  g = gse2.GseParser(file)
  data = g.trace
  print len(data), len(rand)
  print rand.tolist()[0:10]
  print data[0:10]
  del data,g

os.remove(file)
