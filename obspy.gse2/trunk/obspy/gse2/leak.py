#!/usr/bin/python
"""Small script to test wheather memory leakage occurs during reading of
gse data or not"""

import __init__ as gse2
from numpy import *

rand = array(1000*random.random(1e7),dtype='l')

file = "leak.gse"

print rand[0:10]
gse2.write({}, rand, file)

while True:
  (header,data) = gse2.read(file)
  print len(data), len(rand)
  print rand[0:10]
  print data[0:10]
  del header, data

os.remove(file)
