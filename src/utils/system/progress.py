"""a collection of utilities to monitor progess in loops"""
import os
import sys
from tqdm import tqdm

def tg_tqdm(iter_obj, desc: str= None):
    """handy telegram tqdm"""
    try:
        path = os.path.expanduser("~/.config/telegram_bot/tg_tqdm.txt")
        with open(path) as file:
            return tqdm(iter_obj, desc=desc, token = file.readline()[:-1], chat_id = file.readline()[:-1])
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError("found no Telegram access information --> " +
                "Showing Telegram progess bar is not possible") from fnf_error

def working_on(msg: str):
    """printing progress inside of loops with overwriting prev. line"""
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()
