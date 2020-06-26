import os 

os.system('mkdir dataset')
os.system('gdown https://drive.google.com/uc?id=1HMkzNpiUrndGyvVqsvKBbD3yTkjOnydH')
os.system('unzip "Copy of shopee-product-detection-dataset.zip" -d dataset/')
os.system('rm -rf "Copy of shopee-product-detection-dataset.zip" ')
os.system("mv dataset/test/test/* dataset/test/")
os.system("mv dataset/train/train/* dataset/train/")
os.system("rm -rf dataset/test/test")
os.system("rm -rf dataset/train/train")
