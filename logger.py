import os.path as osp
import time
import sys
import logging

import glob
import os


def setup_logger(logpth):
    logfile = 'ModelLog-{}.log'.format(time.strftime('%d-%m-%Y'))

    new_file = True
    try:
        logfile = max(glob.glob(f'{logpth}/*.log'), key=os.path.getctime)
        new_file = False
    except ValueError:
        logfile = osp.join(logpth, logfile)
    
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
   
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
    
    if new_file:
        try:
            with open('model/model.py', 'r') as txt_file:
                logging.getLogger().info(txt_file.read())
        except FileNotFoundError: ...

    return new_file


if __name__ == '__main__':
    print(setup_logger('savedmodels/b1c3f549827ccdf294d1a1b219eea92b'))
