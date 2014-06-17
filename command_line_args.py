#!/usr/bin/python3

import argparse
class command_line:
    '''Test the parsing of command line arguments. Run e.g. in bash with
    ./command_line_args.py -v -i ./data -o ~/hiwi/Taupy/ '''

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-v', '--verbose', '-d','--debug', action='store_true',
                            help='increase output verbosity')
        parser.add_argument('-i', '--input_dir',
                            help = 'set directory of input velocity model')
        parser.add_argument('-o', '--output_dir',
                            help = 'set where to write the .taup model')

        args = parser.parse_args()
        
        print(args.verbose)
        # note the following don't have a value!
        #print(args.v) 
        #print(args.debug)
        #print(args.d)
        DEBUG=args.verbose
        print(DEBUG)
        print('input dir is: ' + args.input_dir)
        print('output dir is: ' + args.output_dir)

    @classmethod
    def main(cls):
        test = command_line()

if __name__ == '__main__':
    command_line.main()
