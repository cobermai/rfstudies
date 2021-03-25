import logging
from datetime import datetime
from sys import stderr

def logger(logger_name: str = "log", level_console_handler="INFO", level_file_handler=""):
    """
    This function returns a custom logger. It writes log outputs in the console and
    a log file and can even write them to Telegram. The format is custom. The warning level is defined by the
    input strings. use:
    log = make_loggger("INFO", "DEBUG", "DEBUG")
    log.warning("This is a warning")  #  a warning will be written in the console and a more detailed on in the log file
    :param level_console_handler: string, has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param level_file_handler: string, has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param level_telegram_handler: string, has to be contained in (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if level_console_handler not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] or \
            level_file_handler not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", ""]:
        raise Exception("The inpus string has to be one of (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    log = logging.getLogger(logger_name)  # create logger
    log.setLevel("DEBUG")
    if log.hasHandlers():
        print("logger named ""API_logger"" already has handler. Ddeleting old hanlders and adding new ones")
        for handler in log.handlers:
            log.removeHandler(handler)

    # FILE log handler: writes log files to a txt file
    if level_file_handler != "":
        format_string = "%(asctime)s|%(levelname)-8s|%(filename)-20s|line%(lineno)3s|%(funcName)-15s|%(message)s"
        formatter_fh = logging.Formatter(format_string)  # create formatter
        date = str(datetime.date(datetime.today()))
        filepath_logfile = "/utils/log_file_"
        fh = logging.FileHandler(filepath_logfile + date + ".txt") # create file handler (written to log file)
        fh.setLevel(level_file_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        fh.setFormatter(formatter_fh)  # setting the custom format to the console handler
        log.addHandler(fh)  # add this hanlder to the logger

    # CONSOLE log handler: writes log files to the python console
    format_string = "%(levelname)-5s[line%(lineno)3s|%(funcName)-15s]:%(message)s"  # add logger name if differnet loggers exist
    formatter_ch = logging.Formatter(format_string)  # create formatter
    ch = logging.StreamHandler(stderr)  # create console handler (output in the console)
    ch.setLevel(level_console_handler)  # set output lvl (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    ch.setFormatter(formatter_ch)  # setting the custom format to the console handler
    log.addHandler(ch)  # add this handler to the logger

    return log

def log_w_tg(log: logging.Logger, level_telegram_handler: str = "") -> None:
    """
    This will add a log handler to log that will write logs to telegram.
    You need to have a telegram bot setup with access informatino located in the right folder.
    """
    import telegram_handler  # telegram log handler
    if level_telegram_handler != "":
        format_string = "%(levelname)-5s[line%(lineno)3s|%(funcName)-15s]:\n<b>%(message)s</b>"
        telegram_access_data_filepath = "/home/lfischl/.config/telegram_bot/logC_bot.txt"
        with open(telegram_access_data_filepath,
                  "r") as file:
            formatter_tg = telegram_handler.HtmlFormatter(format_string)
            tgh = telegram_handler.TelegramHandler(token=file.readline()[:-1],
                                                   chat_id=file.readline()[:-1],
                                                   level=level_telegram_handler,
                                                   disable_notification=True)
            tgh.setFormatter(formatter_tg)
            log.addHandler(tgh)