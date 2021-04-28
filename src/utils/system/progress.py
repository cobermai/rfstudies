from tqdm import tqdm
import os
import sys


def tg_tqdm(iter_obj, desc: str= None):
    try:
        path = os.path.expanduser("~/.config/telegram_bot/tg_tqdm.txt")
        with open(path) as file:
            return tqdm(iter_obj, desc=desc, token = file.readline()[:-1], chat_id = file.readline()[:-1])
    except FileNotFoundError as fnf_err:
        raise FileNotFoundError("found no Telegram access information --> Showing Telegram progess bar is not possible")

def working_on(msg: str):
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()
