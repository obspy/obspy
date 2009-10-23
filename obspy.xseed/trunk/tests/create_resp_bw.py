# Just for testing purposes.

# DELETE ALL OLD ORFEUS RESP FILES! OTHERWISE IT WILL JUST APPEND TO THE OLD
# FILES.

from glob import iglob
import os

# ADJUST FOR YOUR SYSTEM!
RDSEED_PATH = '/Users/lion/Downloads/rdseedv4.8/rdseed.intelMAC'

output_base = 'data' + os.sep + 'RESP' + os.sep + 'bw'
# Generate output directory if necessary.
# generate output directory
if not os.path.isdir(output_base):
    os.makedirs(output_base)

# Get folders in orfeus.
for file in iglob('data' + os.sep + 'bw' + os.sep + '*'):
    cur_file = file.split(os.sep)[-1]
    # Create output directory if necessary.
    working_folder = output_base + os.sep + cur_file
    working_folder = os.path.abspath(working_folder)
    if not os.path.isdir(working_folder):
        os.makedirs(working_folder)
    # Set the current working directory to the working folder.
    file = os.path.abspath(file)
    cur_dir = os.getcwd()
    os.chdir(working_folder)
    # Create the RESP files.
    os.system(RDSEED_PATH + ' -f ' + file + ' -R')
    os.chdir(cur_dir)
