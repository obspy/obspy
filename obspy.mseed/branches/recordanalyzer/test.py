"""
Microseconds where the last two digits are not 00 should be written.

In this test case we have 279999 microseconds.
"""

from recordanalyzer import RecordAnalyser
from obspy.core import read
import os

# Read and write the record again.
st = read('BW.UH3.__.EHZ.D.2010.171.first_record')
tempfile = 'out.mseed'
st.write(tempfile, format='MSEED', reclen=512)
st2 = read(tempfile)
# The starttimes should match
assert st[0].stats.starttime == st2[0].stats.starttime
# Should also be true for the stream objects.
assert st[0].stats == st2[0].stats
# Use the record analyzer.
fp = open('BW.UH3.__.EHZ.D.2010.171.first_record', 'rb')
rec1 = RecordAnalyser(fp)
fp.close()
fp = open(tempfile, "rb")
rec2 = RecordAnalyser(fp)
fp.close()
# Both should have the same start times.
assert rec1.corrected_starttime == rec2.corrected_starttime
os.remove(tempfile)
