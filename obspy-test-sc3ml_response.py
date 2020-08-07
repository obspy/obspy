from obspy.io.seiscomp.inventory import _read_sc3ml
inv=_read_sc3ml('IM.I31KZ_SC3_formatted.xml')
print(inv)
