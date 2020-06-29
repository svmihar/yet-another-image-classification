import os
import time
import logging
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t",
    "--time",
    type=int,
    default=11,
    help="mau berapa kali diulang dalam setengah jam?",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="./dataset/models/",
    help="model directory to upload",
)
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def uploader(model_path = args.model): 
    os.chdir(model_path)
    root = "../../" if "./dataset/models" in model_path else "../"
    os.system("drive add_remote")
    c += 1
    os.chdir(root)
