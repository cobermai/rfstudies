"""
This module helps to easily define a logger that can write log messages to the console, a file and even Telegram
The log messages are custom
"""
import os
import logging
from sys import stderr
from datetime import datetime
import requests
import telegram_handler

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
            file_name = "log_file_" + date + ".txt"
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

def logger_add_tg(log: logging.Logger, level_telegram_handler: str) -> None:
    """
    This will add a log handler to log that will write logs to telegram.
    You need to have a telegram bot setup with access informatino located in the right folder.
    :param log: the logger to which a tg_handler should be added
    :param level_telegram_handler: the level the tg_handler should work at
    """
    format_string = "%(levelname)-5s[line%(lineno)3s|%(funcName)-15s]:\n<b>%(message)s</b>"
    path = os.path.expanduser("~/.config/telegram_bot/tg_tqdm.txt")
    with open(path) as file:
        formatter_tg = telegram_handler.HtmlFormatter(format_string)
        tgh = telegram_handler.TelegramHandler(token=file.readline()[:-1],
                                               chat_id=file.readline()[:-1],
                                               level=level_telegram_handler,
                                               disable_notification=True)
        tgh.setFormatter(formatter_tg)
        log.addHandler(tgh)

def try_logger_add_tg(log: logging.Logger, level_telegram_handler: str) -> None:
    """
    checks for internet connection and connection file without throwing errors
    :param log: the logger to which a tg_handler should be added
    :param level_telegram_handler: the level the tg_handler should work at
    """
    try:
        requests.get("https://home.cern/", timeout=1)  # check if internet connection is possible
        logger_add_tg(log, level_telegram_handler)  # try adding tg logger
    except FileNotFoundError:
        log.debug("Adding tg logger failed due to missing connection information. Logging will continue without.")
    except requests.exceptions.ConnectionError:
        log.debug("Adding tg logger failed due to missing internet connection. Logging will continue without.")
