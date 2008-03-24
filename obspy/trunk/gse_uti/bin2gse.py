#!/usr/bin/python
"""Script to transfer int32 bit binary data to GSE2 using a dummy header"""

from numpy import *
import gse
import sys, struct, os
import array as myarray

# Robert arbeitet try statement alle Punkte nacheinander ab?
try:
	binfile = sys.argv[1]
	ifile = open(binfile, 'rb')
except (IOError, IndexError):
#except IOError:
	print "ERROR: file %s does not exist"
	print __doc__
	sys.exit(1)
#except IndexError:
#	ndat = 190000

#gsefile = os.path.splitext(binfile)[0] + '.gse'
gsefile = binfile + '.gse'
end_of_file_flag = False
ndat = 190000

arr = myarray.array('h')
try: arr.fromfile(ifile,ndat)
except EOFError:
	end_of_file_flag = True
	pass
ifile.close()

# check if ndat was enough to reach the end of file
if not end_of_file_flag:
	print "Error: Didn't reach end of file, tried %i numbers" % ndat
	sys.exit(1)

# help(numpy) shows possible types, i.e. int32
data = array(arr,dtype=int32)
del arr

header = {}
header['samp_rate'] = 50.
header['station'] = 'BOD'

gse.write(header,data,gsefile)
