import os
import time
import argparse
from _uploader import uploader

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", type=int, default=0, help="waktu sleep")
args = parser.parse_args()

if args.time > 0:
    print(f"sleeping for {args.time*3600} seconds")
    time.sleep(args.time * 3600)

training_folder = "./dataset"
uploader()
os.system(f"rm -rf {training_folder}")

# upload models to google drive
