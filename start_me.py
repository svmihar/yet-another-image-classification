import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--link",
    type=str,
    default="https://drive.google.com/uc?id=1HMkzNpiUrndGyvVqsvKBbD3yTkjOnydH",
    help="link google drive datasetnya ",
)
args = parser.parse_args()

commands = [
    "mkdir dataset",
    "pip install drive-cli",
    f"gdown {args.link}",
    'unzip "Copy of shopee-product-detection-dataset.zip" -d dataset/',
    'rm -rf "Copy of shopee-product-detection-dataset.zip" ',
    "mv dataset/test/test/* dataset/test/",
    "mv dataset/train/train/* dataset/train/",
    "rm -rf dataset/test/test",
    "rm -rf dataset/train/train",
]

for cmd in commands:
    os.system(cmd)
