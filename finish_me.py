import os
import time
import argparse 

parser= argparse.ArgumentParser()
parser.add_argument('-t', '--time', type=int, default=0, help='waktu sleep')
args = parser.parse_args()

if args.time > 0: 
    print(f'sleeping for {args.time} seconds')
    time.sleep(args.time)

training_folder = "./dataset"

os.system(f"rm -rf {training_folder}")
# upload models to google drive
