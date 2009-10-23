from glob import iglob
from obspy.xseed import Parser
import os

seed_base = 'data' + os.sep + 'orfeus'
#input_base = 'data' + os.sep + 'RESP' + os.sep + 'orfeus'
output_base = 'output' + os.sep + 'RESP' + os.sep +  'orfeus'

# Create dir if necessary.
if not os.path.isdir(output_base):
    os.makedirs(output_base)
    
def compareRESPFiles(original, new):
    """
    Function to compare to RESP files.
    """
    org_file = open(original, 'r')
    new_file = open(new, 'r')
    org_list = []
    new_list = []
    for org_line in org_file:
        org_list.append(org_line)
    for new_line in new_file:
        new_list.append(new_line)
    # Skip the first line.
    for _i in xrange(1,len(org_list)):
        try:
            assert org_list[_i] == new_list[_i]
        except:
            # Skip if it is the header.
            if org_list[_i] == '#\t\t<< IRIS SEED Reader, Release 4.8 >>\n' and\
               new_list[_i] == '#\t\t<< obspy.xseed, Version 0.1.3 >>\n':
                continue
            # Skip if its a short time string.
            if org_list[_i].startswith('B052F22') and \
                new_list[_i].startswith('B052F22') and \
                org_list[_i].replace('\n',
                    ':00:00.0000\n'[-(len(new_list[_i]) - \
                                      len(org_list[_i])) - 1:]) == \
                new_list[_i]:
                continue
            if org_list[_i].startswith('B052F23') and \
                new_list[_i].startswith('B052F23') and \
                org_list[_i].replace('\n',
                    ':00:00.0000\n'[-(len(new_list[_i]) - \
                                      len(org_list[_i])) - 1:]) == \
                new_list[_i]:
                continue
            msg = '\nCompare failed for:\n' + \
                  'File :\t' +  original.split(os.sep)[-1] + \
                  '\nLine :\t' + str(_i+1) + '\n' + \
                  'EXPECTED:\n' + \
                  org_list[_i] + \
                  'GOT:\n' + \
                  new_list[_i]
            raise AssertionError(msg)

# Loop over all orfeus seed files.
for folder in iglob(seed_base + os.sep + '*'):
    # Loop over orfeus files.
    for file in iglob(folder + os.sep + '*'):
        print file,
        print '\t|| Channels:',
        # Create the folder for the new RESP files.
        RESP_folder = file.replace('data' + os.sep,
                                        'output' + os.sep + 'RESP' + os.sep)
        if not os.path.isdir(RESP_folder):
            os.makedirs(RESP_folder)
        # Original RESP folder.
        Org_RESP_folder = RESP_folder.replace('output' + os.sep,
                                              'data' + os.sep)
        # Create the RESP file.
        sp = Parser()
        sp.parseSEEDFile(file)
        sp.getChannelResponse(folder = RESP_folder)
        # Compare all the created RESP files.
        for RESP in iglob(RESP_folder + os.sep + '*'):
            print '[' + RESP.split(os.sep)[-1].split('.')[-1] + ' ...',
            org_RESP = RESP.replace('output' + os.sep, 'data' + os.sep)
            compareRESPFiles(org_RESP, RESP)
            print 'OK]',
        print ''