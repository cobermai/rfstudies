import logging
from datetime import datetime
from sys import stderr
import os

def logger(level_console_handler: str = "INFO", level_file_handler: str ="DEBUG", name: str = "MLOG"):
    """
    This function returns a custom logger. It writes log outputs in the console and
    a log file. The format is custom. The warning level is defined by the input strings. use ex:
    log = make_loggger("INFO", "DEBUG")
    log.warning("This is a warning")  #  a warning will be written in the console and a more detailed on in the log file
    :param level_console_handler: Defines the level from which log messages will be displayed in the console. Has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param level_file_handler: Defines the level from which log messages will be displayed in the console. Has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param name: the name of the logger. ML + LOG = MLOG """

    log = logging.getLogger(name)  # create logger
    if log.hasHandlers():
        log.makeRecord(name= "init_logger_msg", level=log.getEffectiveLevel(),
                       msg= "Info: logger initializer was called and MLOG already existed")
    else:
        log.setLevel("DEBUG")  # when a log file will be created (and next handled by the log handlers) (like a lower boundary)
        # FILE log handler: writes log files to a txt file
        if level_file_handler != None:
            format_string = "%(name)-10s|%(asctime)s|%(levelname)-8s|%(filename)-20s|line%(lineno)3s|%(funcName)-15s|%(message)s"
            formatter_fh = logging.Formatter(format_string)  # create formatter
            date = str(datetime.date(datetime.today()))
            dir_path = os.path.dirname(__file__) + "/log_files/"
            file_name = "log_file_" + date + ".txt"
            if not os.path.isdir(dir_path): os.mkdir(dir_path)
            if not os.path.isfile(dir_path + file_name):
                with open(dir_path + file_name, "w") as f:
                    f.write("name      |date                   |level   |filename            |line_no|function_name  |message\n")
            fh = logging.FileHandler(dir_path + file_name) # create file handler (written to log file)
            fh.setLevel(level_file_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            fh.setFormatter(formatter_fh)  # setting the custom format to the console handler
            log.addHandler(fh)  # add this hanlder to the logger

        # CONSOLE log handler: writes log files to the python console
        if level_console_handler!= None:
            format_string = "%(name)s: %(levelname)-5s[line%(lineno)3s|%(funcName)-15s]:%(message)s"  # add logger name if differnet loggers exist
            formatter_ch = logging.Formatter(format_string)  # create formatter
            ch = logging.StreamHandler(stderr)  # create console handler (output in the console)
            ch.setLevel(level_console_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            ch.setFormatter(formatter_ch)  # setting the custom format to the console handler
            log.addHandler(ch)  # add this handler to the logger

        return log

def logger_add_tg(log: logging.Logger, level_telegram_handler: str = None) -> None:
    """
    This will add a log handler to log that will write logs to telegram.
    You need to have a telegram bot setup with access informatino located in the right folder.
    """
    import telegram_handler  # telegram log handler
    if level_telegram_handler != None:
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