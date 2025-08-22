
from os import times_result
from aqms.client import Client
import time

with Client("aqmsdev1.isti.net", 9321, timeout=60) as client:
# with Client("beryl3.gps.caltech.edu", 9101, timeout=60) as client:

    start = time.time()-600
    end = start + 10

    sr = client.get_samplerate(
                        # network="HV", station="PAUD", channel="HHZ"
                        # network="CI", station="SDG", channel="BHZ",
                        network="CI", station="SRT", channel="BHZ",
                    #     network="CI", station="SMR", channel="BHZ",
                        )   
    print(f'got sample rate: {sr}\n')

    times = client.get_times(
                        # network="HV", station="PAUD", channel="HHZ"
                        # network="CI", station="SDG", channel="BHZ",
                        network="CI", station="SRT", channel="BHZ",
                    #     network="CI", station="SMR", channel="BHZ",
                        )
    if times is not None:                       
        print('got times (UTC):', time.asctime(time.gmtime(times[0][0])), 
                time.asctime(time.gmtime(times[0][1])))
    else:
        print('got no time')
    print()

    # start = times[0][1] - 3600
    # end = start + 120


    # ch = client.get_channels()
    # print(ch)

    print(time.asctime(time.gmtime(start)), time.asctime(time.gmtime(end)))
    time.sleep(5)
    st = client.get_data(
                        network="HV", station="PAUD", channel="HHZ", # ISTI
                        # network="CI", station="SDG", channel="BHZ",
                        # network="CI", station="SRT", channel="BHZ",
                    #     network="CI", station="SMR", channel="BHZ",
                        start =start, end=end)
    print(st) 

