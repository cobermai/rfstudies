"""
This module helps to easily define a logger that can write log messages to the console, a file and even Telegram
The log messages are custom
"""
import os
import logging
from sys import stderr
from datetime import datetime

def logger(level_console_handler: str = "INFO", level_file_handler: str ="DEBUG", name: str = "MLOG") -> logging.Logger:
    """
    This function returns a custom logger. It writes log outputs in the console and
    a log file. The format is custom. The warning level is defined by the input strings. use example:
    log = make_loggger("INFO", "DEBUG")
    log.warning("This is a warning")
    #  a warning will be written in the console and a more detailed one in the log file
    :param level_console_handler: Defines the level from which log messages will be displayed in the console.
    Has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param level_file_handler: Defines the level from which log messages will be displayed in the console.
    Has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param name: the name of the logger. default for the ml project: ML + LOG = MLOG
    :return: a Logger object"""

    log = logging.getLogger(name)  # create logger
    if log.hasHandlers():
        log.log(level=log.getEffectiveLevel(), msg= "Info: logger initializer was called and MLOG already existed")
    else:
        log.setLevel("DEBUG")  # defines when a log file will be created and passed to the handlers (a lower boundary)
        # FILE log handler: writes log files to a txt file
        if level_file_handler is not None:
            format_string = "%(name)-10s|%(asctime)s|%(levelname)-8s|"+\
                            "%(filename)-20s|line%(lineno)3s|%(funcName)-15s|%(message)s"
            formatter_file_handler = logging.Formatter(format_string)  # create formatter
            date = str(datetime.date(datetime.today()))
            dir_path = os.path.dirname(__file__) + "/log_files/"
            file_name = "log_file_" + date + ".log"
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            if not os.path.isfile(dir_path + file_name):
                with open(dir_path + file_name, "w") as file:
                    file.write("name      |date                   " +
                            "|level   |filename            |line_no|function_name  |message\n")
            file_handler = logging.FileHandler(dir_path + file_name) # create file handler (written to log file)
            file_handler.setLevel(level_file_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            file_handler.setFormatter(formatter_file_handler)  # setting the custom format to the console handler
            log.addHandler(file_handler)  # add this hanlder to the logger

        # CONSOLE log handler: writes log files to the python console
        if level_console_handler is not None:
            format_string = "%(name)s: %(levelname)-5s[line%(lineno)3s|%(funcName)-15s]:%(message)s"
            formatter_console_handler = logging.Formatter(format_string)  # create formatter
            console_handler = logging.StreamHandler(stderr)  # create console handler (output in the console)
            console_handler.setLevel(level_console_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_handler.setFormatter(formatter_console_handler)  # setting the custom format to the console handler
            log.addHandler(console_handler)  # add this handler to the logger
    return log
