import os
import time
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)



os.chdir('./dataset/models')
c = 0
while True: 
    logger.info('sleeping for 30 minutes')
    time.sleep(1800)
    c+=1
    os.system('drive add_remote')
    if c>11: 
        os.chdir('../')
        os.system('finish_me.py')

