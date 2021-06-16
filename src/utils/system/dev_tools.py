"""collection of utilities to track workload of the system conveniently"""
import os
import logging
import psutil


def memo_usage(log: logging.Logger):
    """log memory usage of the current process
    :param log: logger for logging the memory usage
    """
    pid = os.getpid()
    memory_usage = psutil.Process(pid).memory_info()[0] / 2. ** 30
    log.debug("you are currently using " + memory_usage + "GByte of Memory")


def memo_usage_print():
    """print memory usage of the current process"""
    pid = os.getpid()
    memory_usage = psutil.Process(pid).memory_info()[0] / 2. ** 30
    print("you are currently using " + memory_usage + "GByte of Memory")
