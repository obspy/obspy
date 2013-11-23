from obspy.station import read_inventory
import numpy as np

inv = read_inventory("./IRIS_single_channel_with_response.xml")
stage = inv[0][0][0].response.get_evalresp_response()
print np.abs(stage).ptp()
