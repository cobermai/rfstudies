import psutil
import os
import logging

def memo_usage(log: logging.Logger = None):
    pid = os.getpid()
    memory_usage = psutil.Process(pid).memory_info()[0] / 2. ** 30
    if log == None:
        print("you are currently using " + memory_usage + "GByte of Memory")
    else:
        log.debug("you are currently using " + memory_usage + "GByte of Memory")