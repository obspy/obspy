from obspy.core import read
import os


files = os.listdir('')

for file in files:
    if file.startswith('.') or file.endswith('.py'):
        continue
    print
    print file
    st = read(file)
