import os
import sys
import shutil

from args import parse_args

import train
import test

def main():
    global args
    if len(sys.argv) > 1:
        args = parse_args()
        print('----- Experiments parameters -----')
        for k, v in args.__dict__.items():
            print(k, ':', v)
    else:
        print('Please provide some parameters for the current experiment. Check-out args.py for more info!')
        sys.exit()


    

if __name__ == '__main__':
    main()