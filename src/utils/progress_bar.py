from tqdm import tqdm
from tg_tqdm import tg_tqdm

if "SHOW_PROGRESS_BAR" not in locals(): SHOW_PROGRESS_BAR = ""

def get_bar(iter_obj):
    if SHOW_PROGRESS_BAR == "tqdm":
        return tqdm(iter_obj)
    elif SHOW_PROGRESS_BAR == "tg_tqdm":
        try:
            with open("/home/lfischl/.config/telegram_bot/tg_tqdm.txt") as file:
                return tg_tqdm(iter_obj, file.readline()[:-1], file.readline()[:-1])
        except FileNotFoundError as fnf_err:
            print("found no access information --> Showing Telegram progess bar is not possible")
            print(fnf_err)
            return iter_obj
    else:
        return iter_obj

"""  here is an older Idea of how to show progress
import sys
sys.stdout.write("\r progress: %d%%" % (progress_num))
sys.stdout.flush()
"""