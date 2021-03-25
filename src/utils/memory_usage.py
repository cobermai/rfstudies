import psutil
import os

def memo_usage():
    pid = os.getpid()
    memoryUse = psutil.Process(pid).memory_info()[0] / 2. ** 30
    print("you are currently using " + memoryUse + "GByte of Memory")

