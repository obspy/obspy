#!/usr/bin/env python
"""Module wrapping again the read and write of the ext_gse.so, i.e. the
extern C code. Testing for correctness of header entries and existence of
files is implemented in this file in python. Note the reader and writer
require numpy arrays, as they are explicitly wrapped in C"""

import numpy
import os
import ext_gse

def has_entry(header,key_,typ_,default,length=None):
  """Function for testing correctness of header entries.
  
  key_ of cetain type and, if
  given, certain length. If not, the header[key_] is set to default
  """
  if not header.has_key(key_) or not isinstance (header[key_],typ_):
    print "WARNING: %s entry of header missing or not of %s" % (key_,typ_)
    print "forcing",key_,"=",default
    header[key_]=default
  if (length):
    if (len(header[key_]) > length):
      print "%s entry of header is > %i" % (key_,length)
      print "forcing",key_,"=",default
      header[key_]=default

def read(gsefile):
  """NOTE: documentation is assigned AFTER definition by:
  read.__doc__ = gse_ext.read.__doc__
  """
  if os.path.exists(gsefile):
    return ext_gse.read(gsefile)
  else:
    print gsefile + "does not exist"
    sys.exit(2)
read.__doc__ = ext_gse.read.__doc__


def write(h,trace,gsefile):
  """Write header h and trace to gsefile.
  
  The definition of the header is given in documentation of the extern C
  function appended after this documentation. Defaults are set
  automatically.

  write(h,trace,gsefile)
  h            : tuple containing the header variables
  trace        : LONG array containing the trace to write
  gsefile      : target file to write
  ----------------------------------------------------------------------
  """
  #
  # check if header has the necessary tuples and if those are of
  # correct type
  has_entry(h,'d_year',int,2007)
  has_entry(h,'d_mon',int,05)
  has_entry(h,'d_day',int,27)
  has_entry(h,'t_hour',int,23)
  has_entry(h,'t_min',int,59)
  has_entry(h,'t_sec',float,24.123)
  has_entry(h,'station',str,'STAU ',length=6)
  has_entry(h,'channel',str,'SHZ',length=4)
  has_entry(h,'auxid',str,'VEL ',length=5)
  has_entry(h,'tracetype',str,'CM6 ',length=4)
  has_entry(h,'n_samps',int,len(trace))
  has_entry(h,'samp_rate',float,200.)
  has_entry(h,'calib',float,1./(2*numpy.pi)) #calper not correct in gse_driver!
  has_entry(h,'calper',float,1.)
  has_entry(h,'instype',str,'LE-3D ',length=7)
  has_entry(h,'hang',float,-1.0)
  has_entry(h,'vang',float,0.)

  # I get errors with the trace pointer, only solution seems to explicitly copy it
  trace2 = trace.copy()
  err = ext_gse.write((h['d_year'], h['d_mon'], h['d_day'], h['t_hour'],
    h['t_min'], h['t_sec'], h['station'], h['channel'], h['auxid'],
    h['tracetype'], h['n_samps'], h['samp_rate'], h['calib'], h['calper'],
    h['instype'], h['hang'], h['vang']), trace2, gsefile)
  del trace2
  return err
write.__doc__ = write.__doc__ + ext_gse.write.__doc__
