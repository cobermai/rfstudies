import compress_pickle
import pandas as pd
import os
import glob
# This python programm checks if the context data output is exactly the same as in the old version of the API.
# Just a brude force value by value comparison.

new_pickles_dir = "/home/lfischl/cernbox/SWAN_projects/xbox3/data/"
old_pickles_dir = "/home/lfischl/cernbox/SWAN_projects/xbox3/data/cpickles/"
new_filenames = [os.path.split(filepath)[1] for filepath in glob.glob(new_pickles_dir + "*") if os.path.isfile(filepath)]
old_filenames = [os.path.split(filepath)[1] for filepath in glob.glob(old_pickles_dir + "*") if os.path.isfile(filepath)]

print("The symmetric difference of filenames: \n")
print(set(new_filenames).symmetric_difference(set(old_filenames)))
print("\n")

for filename in set(new_filenames).intersection(set(old_filenames)):
    print("looking at the files with name: " + filename)
    cdold = compress_pickle.load(old_pickles_dir + filename, "gzip")
    cdnew = compress_pickle.load(new_pickles_dir + filename, "gzip")

    #testing if all filenames are equal
    if "context_data" in filename:
        for key in ["name_list", "data_label", "data_type", "timestamp"]:
            if cdold[key] != cdnew[key]:
                print("attention: difference in " + key)
                print(cdold[key])
                print(cdnew[key])

        #testing if all entries of data-list are similar
        for num in range(3):  #range(len(cdold["data_list"])):
            for key in list(cdold["data_list"][0].keys()):
                if key != "Timestamp" :
                    if cdold["data_list"][num][key] - cdnew["data_list"][num][key] > 1e-13: print(key + ": old " + str(cdold["data_list"][num][key]) + " new: " + str(cdnew["data_list"][num][key]))
                else:
                    if cdold["data_list"][num][key] - cdnew["data_list"][num][key] != pd.Timedelta(0) : print(key + ": old timestamp: " + str(cdold["data_list"][num][key]) + " new timestamp: " + str(cdnew["data_list"][num][key]))
    else:
        pass