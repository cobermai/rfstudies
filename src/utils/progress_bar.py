from tqdm import tqdm
import os

# This is a trick to ensure global compability. If the global variable SHOW_PROGRESS_BAR has
# not been set yet, it gets set to ""
if "SHOW_PROGRESS_BAR" not in locals(): SHOW_PROGRESS_BAR = None

def get_bar(iter_obj):
    """
    Ads a visible progress bar (tqdm) in either the command line or on the TG messenger. The global variable
    SHOW_PROGRESS_BAR defines the settings
    :param iter_obj: any iteration object in a for loop
    """
    if SHOW_PROGRESS_BAR == "tqdm":
        return tqdm(iter_obj)
    elif SHOW_PROGRESS_BAR == "tg_tqdm":
        try:
            path = os.path.expanduser("~/.config/telegram_bot/tg_tqdm.txt")
            with open(path) as file:
                return tqdm(iter_obj, token = file.readline()[:-1], chat_id = file.readline()[:-1])
        except FileNotFoundError as fnf_err:
            print("found no access information --> Showing Telegram progess bar is not possible")
            print(fnf_err)
            return iter_obj
    else:
        return iter_obj

"""  Older version to show progress. This has to be put inside the loop
import sys
sys.stdout.write("\r progress: %d%%" % (progress_num))
sys.stdout.flush()
"""