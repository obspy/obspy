from obspy.station import read_inventory
inv = read_inventory("./IRIS_single_channel_with_response.xml")
stage = inv[0][0][0].response.get_evalresp_response()

################
# DEBUGGING START
import sys
__o_std__ = sys.stdout
sys.stdout = sys.__stdout__
from IPython.core.debugger import Tracer
Tracer(colors="Linux")()
sys.stdout = __o_std__
# DEBUGGING END
################
