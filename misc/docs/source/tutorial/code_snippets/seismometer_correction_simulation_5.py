from obspy import read, read_inventory

st = read("/path/to/IU_ULN_00_LH1_2015-07-18T02.mseed")
tr = st[0]
inv = read_inventory("/path/to/IU_ULN_00_LH1.xml")
tr.attach_response(inv)

pre_filt = [0.001, 0.005, 10, 20]
tr.remove_response(pre_filt=pre_filt, output="DISP",
                   water_level=60, plot=True)
