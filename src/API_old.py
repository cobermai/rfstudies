import os
import glob
import time
import datetime
import numpy as np
import compress_pickle
import pandas as pd
from functools import partial
from multiprocessing import Pool
from bisect import bisect_left
from scipy.interpolate import interp1d
from scipy import signal

import json
"""
Simple API to read / clean / pickle / unpickle CLIC files stored in TDMS format
"""
def read_JSON(file, return_dict=False):
    print("CHANGE TO LOWER CASE read_json")

def read_json(file, return_dict=False):
    """
    function reading a json file and returning a list of parameters or a dictionary
    :param file: setting filename
    :param return_dict: boolean to specify if the data return is a dictionary or a list of variables
    :return: a dictionary or a list of variables
    """

    with open(file, "r") as read_file:
        data = json.load(read_file)

    if os.path.isdir(data["ml_for_alarm_EOS_absolute_path"]):
        access2eos = True
        print("Access to EOS possible")
        root_directory = data["ml_for_alarm_EOS_absolute_path"]
    else:
        access2eos = False
        print("Access to EOS Impossible")
        # search for a list of root_directory depending on the machine used to execute the notebook
        for directory in data['list_of_alternative_absolute_paths']:
            if os.path.isdir(directory):
                root_directory = directory

    data["root_directory"] = root_directory
    path = root_directory + data["pickled_dataset_relative_path"]
    data["dataset_directory"] = path
    source_path = root_directory + data["dataset_relative_path"]

    list_of_timestamps = data["list_of_timestamps"]
    list_of_data_types = data["list_of_data_types"]
    field_studied = data["field_studied"]
    list_of_runs_studied = data["list_of_runs_studied"]

    dict_of_label_ratio = {}
    for key in data["dict_of_label_ratio"].keys():
        dict_of_label_ratio[int(key)] = data["dict_of_label_ratio"][key]
    data["dict_of_label_ratio"] = dict_of_label_ratio

    label_string = {}
    for key in data["label_string"].keys():
        label_string[int(key)] = data["label_string"][key]

    if return_dict:
        return data
    else:
        return access2eos, root_directory, path, source_path, \
               list_of_timestamps, list_of_data_types, dict_of_label_ratio, label_string, \
               field_studied, list_of_runs_studied



def group_files_from_a_directory(path):
    """
    function which parse a directory and return CLIC TMDS files grouped in a list of dictionaries.
    for a given day, up to 4 files can exist: 2 actual data (event and trend), 2 fast access / temporary data
    :param path: folder to parse
    :return listOfDictOfFiles: list of files grouped in dictionaries
    """

    listOfEventDataFiles = glob.glob(path + "Event*.tdms")
    print("\n", "Number of EventData files in ", path, " = ", len(listOfEventDataFiles))

    #  find timestamps
    listOfTimestamp = []
    for eventDataFile in listOfEventDataFiles:
        timestamp = eventDataFile.split(path)[1].split("_")[1].split(".")[0]
        listOfTimestamp.append(timestamp)
    listOfTimestamp.sort()
    print("\n", "Number of timestamps based on EventData files in ", path, " = ", len(listOfTimestamp))

    # group files
    listOfDictOfFiles = []
    for timestamp in listOfTimestamp:
        listOfFileWRTTimestamp = glob.glob(path + "*" + timestamp + "*")

        if len(listOfFileWRTTimestamp) == 2:
            # assuming TrendData and EventData exist

            dictOfFiles = {}
            dictOfFiles["timestamp"] = timestamp
            for file in listOfFileWRTTimestamp:
                if "Trend" in file:
                    if "index" in file:
                        dictOfFiles["TrendData_index"] = file
                    else:
                        dictOfFiles["TrendData"] = file
                if "Event" in file:
                    if "index" in file:
                        dictOfFiles["EventData_index"] = file
                    else:
                        dictOfFiles["EventData"] = file
            listOfDictOfFiles.append(dictOfFiles)

        else:
            # assuming TrendData is missing
            print("Timestamp ", timestamp, " has no TrendData: ", listOfFileWRTTimestamp)

    print("\n", "Number of set of files in ", path, " = ", len(listOfDictOfFiles))
    print("")

    return listOfDictOfFiles


def checking_the_existence_of_pickle(listOfDictOfFiles,
                                     data_type_list,
                                     list_of_configs,
                                     destination_path):
    listOfNotAlreadyExistingPickles = []

    for dictOfFiles in listOfDictOfFiles:

        timestamp = dictOfFiles["timestamp"]

        parsing = False
        for data_type in data_type_list:
            for config in list_of_configs:
                fileName = pickle_name(config, data_type, timestamp)
                if not os.path.isfile(destination_path + fileName):
                    print(fileName, " does NOT already exists")
                    parsing = True
        if not parsing:
            print("In ", timestamp, " everything already exists")
        else:
            listOfNotAlreadyExistingPickles.append(dictOfFiles)

    return listOfNotAlreadyExistingPickles


def get_lists_of_gn_for_different_labels(EventData, number_of_random_class_0=-1):
    """
    function used to get the list of pulses marked as breakdown, ok and corrupt
    bd = 0
    ok = 1
    corrupt = -1
    :param EventData:
    :param number_of_random_class_0: number of "ok pulse" in the "list of ok pulses", if -1 then every ok pulses are returned
    :return list_of_gn__ok: the list of bd group name
    :return list_of_gn__bd: the list of before bd group name
    :return list_of_gn__corrupt: pulse in a corrupt pattern...
    """
    necessary_channels = ["DC_Down", "DC_Up",
                          "PEI_Amplitude", "PEI_Phase", "PER_log",
                          "PKI_Amplitude", "PKI_Phase", "PKR_log",
                          "PSI_Amplitude", "PSI_Phase",
                          "PSR_Amplitude", "PSR_log", "PSR_Phase"]
    necessary_channels.sort()
    necessary_channels = set(necessary_channels)

    list_of_gn = []  # the complete list of group_name
    list_of_gn__bd = []  # the list of breakdown according to CLIC team
    list_of_gn__corrupt = []  # the list of corrupt pulses

    ithBD = 0  # counter of breakdowns
    for group in EventData.groups():

        gn = group.name

        # generating a list with every group_name as strings
        list_of_gn.append(gn)

        available_channels = []
        empty_channel = False
        for channel in group.channels():
            channel_renamed = channel.name.replace(" ", "_")
            available_channels.append(channel_renamed)
            if len(channel[:]) == 0:
                empty_channel = True

        available_channels = set(available_channels)

        if "wf_start_time" in channel.properties.keys():
            missing_start_time = False
        else:
            missing_start_time = True

        if necessary_channels.issubset(available_channels) and \
                not missing_start_time and \
                not empty_channel:

            if ("down" in gn):  # then the current pulse is labeled as breakdown
                ithBD += 1  # counter of breakdowns
                list_of_gn__bd.append(gn)

        else:  # corrupt pulse
            print("\n ! ! ! Corrupt pulse :", gn, " ! ! !",
                  "\n ! ! ! Missing star time ?", missing_start_time,
                  "\n ! ! ! Empty channel ?", empty_channel, )
            list_of_gn__corrupt.append(gn)

    # generating a list of ok pulses
    gn__no_ok = set(list_of_gn__bd +
                    list_of_gn__corrupt)
    gn__ok = list(set(list_of_gn).difference(gn__no_ok))
    if number_of_random_class_0 == -1:
        list_of_gn__ok = gn__ok
    else:
        list_of_random_id = np.random.randint(0, number_of_random_class_0, len(gn__ok))
        list_of_gn__ok = []
        for id in list_of_random_id:
            list_of_gn__ok.append(gn__ok[id])

    # some simple sanity checks
    print("\n-------------------------------------------------------------\n",
          "Number of elements is the file:", len(list_of_gn), "\n",
          "Number of bd elements is the file:", len(list_of_gn__bd), "\n",
          "Number of ok elements is the file:", len(list_of_gn__ok), "\n",
          "Number of corrupt elements is the file:", len(list_of_gn__corrupt), "\n",
          "Balance = ", len(list_of_gn) - len(list_of_gn__bd) - len(list_of_gn__ok) - len(list_of_gn__corrupt))

    return list_of_gn__ok, list_of_gn__bd, list_of_gn__corrupt


def extract_trend_data_wrt_timestamp(TrendData, target_timestamp, previous_day_TrendData=None):
    """
    function used to get the trend data point the closest to a given timestamp
    :param TrendData:
    :param target_timestamp:
    :param previous_day_TrendData: trend data of the previous day
    :return results: dictionary with the results
    """

    results = {}
    # theoretical list of channels
    list_of_channels = [
        "Loadside_win",
        "Tubeside_win",
        "Collector",
        "Gun",
        "IP_before_PC",
        "PC_IP",
        "WG_IP",
        "IP_Load",
        "IP_before_structure",
        "US_Beam_Axis_IP",
        "Klystron_Flange_Temp",
        "Load_Temp",
        "PC_Left_Cavity_Temp",
        "PC_Right_Cavity_Temp",
        "Bunker_WG_Temp",
        "Structure_Input_Temp",
        "Chiller_1",
        "Chiller_2",
        "Chiller_3",
        "PKI_FT_avg",
        "PSI_FT_avg",
        "PSR_FT_avg",
        "PEI_FT_avg",
        "PKI_max",
        "PSI_max",
        "PSR_max",
        "PEI_max",
        "BLM_TIA_min",
        "BLM_min",
        "DC_Down_min",
        "DC_Up_min",
        "BLM_TIA_Q",
        "PSI_Pulse_Width",
        "Pulse_Count",
        "Timestamp"]

    if target_timestamp is None:

        for channel in list_of_channels:
            results[channel] = None

    else:

        # find candidates over different groups (usually only one but could be several groups)
        list_of_candidate = []
        for group in TrendData.groups():
            current_candidate = {"delta": np.inf, "id": np.inf, "group": group, "TrendData": "current"}
            for channel in group.channels():
                if "Timestamp" in channel.name:
                    for id in range(0, len(channel)):
                        delta = (target_timestamp - channel[id]) / np.timedelta64(1, 's')
                        if (delta < current_candidate["delta"]) and (delta >= 0.):
                            current_candidate["delta"] = delta
                            current_candidate["id"] = id
                        # else:
                        # break
            list_of_candidate.append(current_candidate)

        if previous_day_TrendData is None:
            print("No TrendData from the previous day provided")
        else:
            # find candidates over different groups (usually only one but could be several groups)
            for group in previous_day_TrendData.groups():
                current_candidate = {"delta": np.inf, "id": np.inf, "group": group, "TrendData": "previous"}
                for channel in group.channels():
                    if "Timestamp" in channel.name:
                        for id in range(0, len(channel)):
                            delta = (target_timestamp - channel[id]) / np.timedelta64(1, 's')
                            if (delta < current_candidate["delta"]) and (delta >= 0.):
                                current_candidate["delta"] = delta
                                current_candidate["id"] = id
                            # else:
                            # break
                list_of_candidate.append(current_candidate)

        # finding the best candidate over the different candidate
        best_candidate = {"delta": np.inf, "id": np.inf, "group": None, "TrendData": "?"}
        for candidate in list_of_candidate:
            if candidate["delta"] < best_candidate["delta"]:
                best_candidate = candidate

        if best_candidate['TrendData'] == "current":

            # getting the list of channels in the current TrendData file
            list_of_actual_channels = []
            for channel in TrendData[best_candidate["group"].name].channels():
                list_of_actual_channels.append(channel.name.replace(" ", "_"))

            # creating a results dictionary with the context data
            for channel in TrendData[best_candidate["group"].name].channels():
                results[channel.name.replace(" ", "_")] = TrendData[best_candidate["group"].name][channel.name][
                    best_candidate["id"]]

        elif best_candidate['TrendData'] == "previous":

            # getting the list of channels in the current TrendData file
            list_of_actual_channels = []
            for channel in previous_day_TrendData[best_candidate["group"].name].channels():
                list_of_actual_channels.append(channel.name.replace(" ", "_"))

            # creating a results dictionary with the context data
            for channel in previous_day_TrendData[best_candidate["group"].name].channels():
                results[channel.name.replace(" ", "_")] = \
                previous_day_TrendData[best_candidate["group"].name][channel.name][
                    best_candidate["id"]]

        else:

            print("Problem retriving TrendData")

        # comparison with the normal list of channel
        diff = list(set(list_of_channels) - set(list_of_actual_channels))
        if not (diff == []):
            print("! ! ! WARNING: Missing channel in TrendData with target_timestamp ",
                  target_timestamp,
                  "! ! !\n",
                  diff)

    return results


def extract_channel_and_start_time(channel):
    """
    function used to extract information from a channel
    if needed, a rescaling can be performed
    :param channel:
    :return data:
    :return start_time:
    """

    # getting the data
    data = channel[:]

    # getting the stat time of the pulse
    if "wf_start_time" in channel.properties.keys():
        start_time = channel.properties["wf_start_time"]
    else:
        start_time = None

    compute = True
    if data is None:
        compute = False
    elif len(data) == 0:
        compute = False

    if not compute:

        # print("\n","!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!","\n",
        #     " Problem in pulse: start_time = ", start_time,"; len(data) = ",len(data),"\n",
        #     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data = None
        start_time = None

    else:

        # normalizing the signal
        if "Scale_Type" in channel.properties.keys():
            if channel.properties["Scale_Type"] == "Polynomial":
                Scale_Coeff_c0 = channel.properties["Scale_Coeff_c0"]
                Scale_Coeff_c1 = channel.properties["Scale_Coeff_c1"]
                Scale_Coeff_c2 = channel.properties["Scale_Coeff_c2"]
                data = Scale_Coeff_c0 + Scale_Coeff_c1 * data + Scale_Coeff_c2 * data ** 2

        # if the signal is a phase, reconstruction of the phase
        if "Phase" in channel.name:
            data = phase_reconstruction(data)

        # convert to simple precision
        data = data.astype('float32')

    return data, start_time


def extract_start_time(channel):
    """
    function used to extract information from a channel
    if needed, a rescaling can be performed
    :param channel:
    :return start_time:
    """

    # extracting the stat time of the pulse
    if "wf_start_time" in channel.properties.keys():
        start_time = channel.properties["wf_start_time"]
    else:
        start_time = None

    return start_time


def phase_reconstruction(data):
    """
    function used to transform the phase signals into some continous signals
    a simple algorithm detects if the variation between 2 consecutive points is bigger than
    the value criterion (1.5 = .75 * 2 works well) and if it is, a jump in the correct direction
    is added to compensate a likely modulo operation in the data. The jumps is +2 or -2 as the
    phase is encoded between -1 and 1.
    TODO: the phase should start at 0 ?!
    :param data:
    :return data:
    """
    initial_phase = data[0]
    criterion = 1.5
    jump = 2.
    for i in range(1, len(data)):
        if (data[i] - data[i - 1]) > criterion:
            data[i] = data[i] - jump
        elif (data[i] - data[i - 1]) < -criterion:
            data[i] = data[i] + jump
        else:
            data[i] = data[i]

    # forcing the phase to start at 0
    data = data - initial_phase

    return data  # , initial_phase


def get_pulse_amplitude_and_duration(data):
    """
    function used to recompute the amplitude and duration of a pulse.
    see study_amplitude_and_duration_of_pulse.ipnb for to visualize the algorithm
    :param data: input waveform
    :return amplitude: estimated amplitude of the top plateau, not the peak
    :return duration: estimated duration of the top plateau
    """

    compute = True
    if data is None:
        compute = False
    elif len(data) == 0:
        compute = False

    if compute:
        ratio = 0.5  # ratio used with the maximum amplitude of the data in order to define which points are in the pulse
        threshold = np.max(data) * ratio
        selected_data = data[(data > threshold)]
        duration = len(selected_data) / 12.  # in nano sec, sampling frequency = 12Ghz
        # amplitude = np.mean(selected_data)
        amplitude = np.median(selected_data)
    else:
        duration = 0.
        amplitude = 0.

    return amplitude, duration


def get_pulse_idea_of_pdf(data):
    """
    function used to extract a few statistical information
    """

    compute = True
    if data is None:
        compute = False
    elif len(data) == 0:
        compute = False

    if compute:
        D1 = np.quantile(data, .10)
        median = np.quantile(data, .50)
        D9 = np.quantile(data, .90)
        minimum = np.min(data)
        maximum = np.max(data)
        average = np.mean(data)
    else:
        D1 = np.nan
        median = np.nan
        D9 = np.nan
        minimum = np.nan
        maximum = np.nan
        average = np.nan

    return D1, median, D9, minimum, maximum, average


def pickle_name(data_label, data_type, timestamp):
    """
    function defining the name of a pickle file using different attributes
    :param data_label: class of signal
    :param data_type: type of signal
    :param timestamp: timestamp of an EventDataFile
    :return filename:
    """

    if data_label == 0:
        label_name = "breakdown"
    elif data_label == 1:
        label_name = "healthy"
    elif data_label == -1:
        label_name = "corrupt"
    else:
        label_name = data_label

    filename = str(timestamp) + "__" + label_name + "__" + data_type + ".p.gz"

    return filename


def write_pickle(data_list, name_list, data_label, data_type, timestamp, path):
    """
    function writing a pickle
    :param data_list: list of arrays containing the data
    :param name_list: list of group.name (= pulse's name) corresponding to each piece of data in data_list
    :param data_label: unique value defining class of signals (breakdown, etc.)
    :param data_type: unique value defining type of signals (PSI, PEI, etc., =channel.name)
    :param timestamp: timestamp of an EventDataFile
    :param path: path where the file is stored
    """
    data = {"data_list": data_list,
            "name_list": name_list,
            "data_label": data_label,
            "data_type": data_type,
            "timestamp": timestamp}
    filename = path + pickle_name(data_label, data_type, timestamp)
    print("\n", "Write file ", filename)
    # pickle.dump( data, open(filename,"wb") )
    compress_pickle.dump(data, filename, compression="gzip")


def read_pickle_from_filename(filename):
    """
    function reading a pickle
    :param filename:
    :return data_list: list of arrays
    :return name_list: list of pulse name
    """
    print("\n", "Read file ", filename)
    data = compress_pickle.load(filename)
    return data["data_list"], data["name_list"], data["data_label"], data["data_type"], data["timestamp"]


def find_nearest_future_timestamp(ts, index):
    """
    function looks which future timestamp in the index is closest to ts
    :param ts: timestamp
    :param index: list of timestamps
    :return: nearest future timestamp
    """
    # in case there is no future timestamp-> ts is older than any timestamp in index
    t0 = [np.datetime64('1970-01-01')]
    s = sorted(index)  # sorts timestamps
    i = bisect_left(s, ts)  # finds index of next timestamp
    nearest_ts = min(t0 + s[max(0, i): i + 2], key=lambda t: abs(ts - t))  # takes element left to index
    return nearest_ts


def study_a_dictOfFiles(list_of_configs, destination_path, data_type_list, dictOfFiles):
    """
    function generating pickles according to one timestamp
    :param list_of_configs: the list of labels parsed
    :param destination_path: where the results are stored
    :param dictOfFiles: a dictionary with a list of files associated to a given timestamp
    :param data_type_list: name of the fields studied
    """

    from nptdms import TdmsFile  # import here to avoid importing when not reading tdmsfile

    t = time.time()

    timestamp = dictOfFiles["timestamp"]  # timestamp of the current dictionary
    check_if_files_exist = True  # avoid to redo file which are already existing

    if check_if_files_exist:
        parsing = False
        for data_type in data_type_list:
            for config in list_of_configs:
                fileName = pickle_name(config, data_type, timestamp)
                if not os.path.isfile(destination_path + fileName):
                    print(fileName, " does NOT already exists")
                    parsing = True
        if not parsing:
            print("In ", timestamp, " everything already exists")
    else:
        parsing = True

    if parsing:

        print("Start reading files :\n",
              dictOfFiles["EventData"], "\n",
              dictOfFiles["TrendData"], "\n", )

        # reading files
        TrendData = TdmsFile.read(dictOfFiles["TrendData"])
        EventData = TdmsFile.read(dictOfFiles["EventData"])

        # quick and dirty fix to use TrendData of the previous day when searching for the closest context data
        string_day = dictOfFiles["TrendData"].split('.tdms')[0].split('_')[-1]
        datetime_day = datetime.datetime.strptime(string_day, "%Y%m%d")
        datetime_previous_day = datetime_day - datetime.timedelta(1)
        string_previous_day = datetime_previous_day.strftime("%Y%m%d")
        name_of_previous_TrendData_file = dictOfFiles["TrendData"].replace(string_day, string_previous_day)
        if os.path.isfile(name_of_previous_TrendData_file):
            # reading an extra file
            previous_day_TrendData = TdmsFile.read(name_of_previous_TrendData_file)
        else:
            previous_day_TrendData = None

        # list of files with different labels
        list_of_gn__ok, list_of_gn__bd, list_of_gn__corrupt = \
            get_lists_of_gn_for_different_labels(EventData, -1)

        # parsing the different labels to explore
        for config in list_of_configs:
            firstEvent = True

            print("")
            print("\n Parsing file with timestamp ", timestamp, " and focus on label ", config, " out of ",
                  list_of_configs)

            if config == 0:
                list_of_gn = list_of_gn__bd
            elif config == 1:
                list_of_gn = list_of_gn__ok
            elif config == -1:
                list_of_gn = list_of_gn__corrupt

            if not list_of_gn:  # if empty list
                # nothing to study ==> nothing to pickle ==> creation of empty pickles

                print("\n Empty config ", config, " for timestamp", timestamp)

                for data_type in data_type_list:
                    fileName = pickle_name(config, data_type, timestamp)
                    tmp = pd.DataFrame(columns=['label_CLIC', 'timestamp', data_type])
                    tmp.to_pickle(destination_path + fileName)
                    print("\n", "Write file ", destination_path + fileName)

                write_pickle(data_list=[],
                             name_list=[],
                             data_label=config,
                             data_type="context_data",
                             timestamp=timestamp,
                             path=destination_path)

            else:  # elif list_of_gn is not empty

                print("\n NOT empty config ", config, " for timestamp", timestamp)

                data_type_list__available = []
                context_data_list = []
                name_list = []

                id_group_name = 0
                for group in EventData.groups():  # go through each event from file

                    # if the pulse is a pulse we need to study
                    if group.name in list_of_gn:
                        id_group_name += 1

                        dataset = {'label_CLIC': [config]}

                        first_channel = True
                        for channel in group.channels():  # go through all channels

                            channel_renamed = channel.name.replace(" ", "_")

                            if channel_renamed not in data_type_list__available:
                                data_type_list__available.append(channel_renamed)

                            # extracting DataEvent
                            data, start_time = extract_channel_and_start_time(channel)

                            # if the pulse doesn't have a start_time, we skip it
                            if start_time is None:
                                if first_channel:
                                    print("\n",
                                          "Parsing file with timestamp ", timestamp, " and focus on label ", config,
                                          "\n",
                                          "--> Pulse with no start time, group.name = ", group.name)
                                dataset['timestamp'] = None
                                dataset[channel_renamed] = [np.array([np.NaN])]
                            else:
                                dataset['timestamp'] = start_time
                                dataset[channel_renamed] = [np.array(data)]

                            # for the first signal in the list of group.channels(), we search for the TrendData (context_data)
                            if first_channel:  # in loop on groups.channels()
                                first_channel = False
                                # using the timestamp of the pulse, extracting the closest context_data from TrendData
                                context_data = extract_trend_data_wrt_timestamp(TrendData, start_time,
                                                                                previous_day_TrendData)
                                # memorizing the context_data
                                context_data_list.append(context_data)
                                # memorizing the name of the pulse
                                name_list.append(group.name)
                                # memorizing the name of the channels

                            if channel_renamed in ["PKI_Amplitude", "PEI_Amplitude", "PSI_Amplitude"]:
                                amplitude, duration = get_pulse_amplitude_and_duration(data)
                                context_data[channel_renamed + "__recomputed_amplitude"] = amplitude
                                context_data[channel_renamed + "__recomputed_duration"] = duration

                            if channel_renamed in ["DC_Down", "DC_Up"]:
                                D1, median, D9, minimum, maximum, average = get_pulse_idea_of_pdf(data)
                                context_data[channel_renamed + "__recomputed_D1"] = D1
                                context_data[channel_renamed + "__recomputed_median"] = median
                                context_data[channel_renamed + "__recomputed_D9"] = D9
                                context_data[channel_renamed + "__recomputed_minimum"] = minimum
                                context_data[channel_renamed + "__recomputed_maximum"] = maximum
                                context_data[channel_renamed + "__recomputed_average"] = average

                        df_new_event = pd.DataFrame(dataset)

                        if firstEvent:
                            firstEvent = False
                            df = df_new_event
                        else:
                            df = df.append(df_new_event, ignore_index=True)

                # next_bd = []
                # for ts in df["timestamp"].values:
                # df_bd = df[df["label_CLIC"]==0]
                # next_bd.append(find_nearest_future_timestamp(ts,df_bd.timestamp.values))
                # df["next_bd"] = next_bd

                for data_type in data_type_list:
                    fileName = pickle_name(config, data_type, timestamp)
                    if data_type in data_type_list__available:
                        df[['label_CLIC', 'timestamp', data_type]].to_pickle(destination_path + fileName)
                        print("\n", "Write file ", destination_path + fileName)
                    elif data_type == "context_data":
                        # write the pickle for the context_data
                        write_pickle(data_list=context_data_list,
                                     name_list=name_list,
                                     data_label=config,
                                     data_type="context_data",
                                     timestamp=timestamp,
                                     path=destination_path)
                    else:
                        print("! ! ! WARNING : data_type ", data_type, " not available in ", data_type_list__available,
                              " ! ! !")
                        tmp = pd.DataFrame(columns=['label_CLIC', 'timestamp', data_type])
                        tmp.to_pickle(destination_path + fileName)
                        print("\n", "Write file ", destination_path + fileName)

        print("\n", "End of parsing file with timestamp ", timestamp, ":", time.time() - t)


def pickle_a_directory(source_path, destination_path, numberOfProcesses=1, fracture=0):
    """
    function parsing a directory and generating pickles for each combiation of:
    timestamp (day)
    channel (type of signal)
    label (as provided in the original data)
    :param source_path: directory to parse
    :param destination_path:
    :param numberOfProcesses:
    """

    # parameters used for parsing
    list_of_configs = [1, 0, -1]

    # list of data_type to expect
    data_type_list = ["DC_Down", "DC_Up",
                      "PEI_Amplitude", "PEI_Phase", "PER_log",
                      "PKI_Amplitude", "PKI_Phase", "PKR_log",
                      "PSI_Amplitude", "PSI_Phase",
                      "PSR_Amplitude", "PSR_log", "PSR_Phase",
                      "context_data"]
    # "BLM", "BLM_TIA", "Col.",  # data_type not used

    # grouping files by timestamp in a directory
    listOfDictOfFiles = group_files_from_a_directory(source_path)

    if fracture == 0:
        pass
    elif fracture != 0:
        listOfDictOfFiles = listOfDictOfFiles[0:int(len(listOfDictOfFiles) / 2**fracture)]

    print("Size of the list to study:", len(listOfDictOfFiles))
    listOfDictOfFiles = checking_the_existence_of_pickle(listOfDictOfFiles,
                                                         data_type_list,
                                                         list_of_configs,
                                                         destination_path)
    print("After removing already existing pickles, the size of the list to study is:", len(listOfDictOfFiles))

    # creating the destination path if needed
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if numberOfProcesses == 1:
        print("SERIAL COMPUTATION\n")
        idFile = 0
        for dictOfFile in listOfDictOfFiles:  # debug [listOfDictOfFiles[0]]:
            idFile += 1
            study_a_dictOfFiles(list_of_configs, destination_path, data_type_list, dictOfFile)
            print(idFile, " file(s) pickled out of ", len(listOfDictOfFiles), "!")

    elif numberOfProcesses > 1:
        print("PARALLEL COMPUTATION\n")
        partial_function = partial(study_a_dictOfFiles, list_of_configs, destination_path, data_type_list)
        with Pool(processes=numberOfProcesses) as pool:
            pool.map(partial_function, listOfDictOfFiles)
            # debug # pool.map(partial_function, [listOfDictOfFiles[0]])


def study_a_dictOfFiles__TrendData_only(destination_path, dictOfFiles):
    """
    function generating pickles according to one timestamp
    :param list_of_configs: the list of labels parsed
    :param destination_path: where the results are stored
    :param dictOfFiles: a dictionary with a list of files associated to a given timestamp
    :param data_type_list: name of the fields studied
    """

    from nptdms import TdmsFile  # import here to avoid importing when not reading tdmsfile

    # check if the file exists already
    if os.path.isfile(destination_path + "TrendData_" + dictOfFiles["timestamp"] + ".p.gz"):

        print(dictOfFiles["TrendData"], " already computed.")

    else:

        t = time.time()

        print("Start studying file :\n",
              dictOfFiles["TrendData"], "\n", )

        TrendData = TdmsFile.read(dictOfFiles["TrendData"])

        results = {}
        for group in TrendData.groups():
            for channel in group.channels():

                if channel.name.replace(" ", "_") not in results.keys():
                    results[channel.name.replace(" ", "_")] = []
                results[channel.name.replace(" ", "_")].extend(TrendData[group.name][channel.name][:])

        max_length = 0
        str_length = dictOfFiles["timestamp"] + "\n"
        for key in results.keys():
            str_length += key + ": length = " + str(len(results[key])) + '\n'
            max_length = np.max([max_length, len(results[key])])
        print_length = False
        for key in results.keys():
            if len(results[key]) != max_length:
                print_length = True
                results[key].extend([np.nan] * (max_length - len(results[key])))
        if print_length:
            print(str_length)

        tmp = pd.DataFrame(results)
        tmp.to_pickle(destination_path + "TrendData_" + dictOfFiles["timestamp"] + ".p.gz")

        print(time.time() - t, " s to study file :\n",
              dictOfFiles["TrendData"], "\n", )


def pickle_a_directory__TrendData_only(source_path, destination_path, numberOfProcesses=1):
    """
    function parsing a directory and generating pickles for each combiation of:
    timestamp (day)
    channel (type of signal)
    label (as provided in the original data)
    :param source_path: directory to parse
    :param destination_path:
    :param numberOfProcesses:
    """

    # grouping files by timestamp in a directory
    listOfDictOfFiles_all = group_files_from_a_directory(source_path)

    listOfDictOfFiles = []
    for dictOfFiles in listOfDictOfFiles_all:
        if not os.path.isfile(destination_path + "TrendData_" + dictOfFiles["timestamp"] + ".p.gz"):
            listOfDictOfFiles.append(dictOfFiles)

    print("Size of the list to study:", len(listOfDictOfFiles))

    # creating the destination path if needed
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if numberOfProcesses == 1:
        print("SERIAL COMPUTATION\n")
        idFile = 0
        for dictOfFile in listOfDictOfFiles:  # debug [listOfDictOfFiles[0]]:
            idFile += 1
            study_a_dictOfFiles__TrendData_only(destination_path, dictOfFile)
            print(idFile, " file(s) pickled out of ", len(listOfDictOfFiles), "!")

    elif numberOfProcesses > 1:
        print("PARALLEL COMPUTATION\n")
        partial_function = partial(study_a_dictOfFiles__TrendData_only, destination_path)
        with Pool(processes=numberOfProcesses) as pool:
            pool.map(partial_function, listOfDictOfFiles)


def read_a_subset_of_pickles(path, list_of_timestamps_regex,
                             list_of_data_types,
                             dict_of_label_ratio={0: 1., 1: 1.},
                             take_nth_sample=1,
                             interp_kind='previous',
                             df_of_pulses_selected=None):
    """
    function which reads many pickles and return them concatenated in dictionaries
    :param path: path of the folder to explore
    :param list_of_timestamps_regex: list of patterns in a regexp form such as
    ["2018"] or ["201801"] or ["20180121","20180123"]
    :param list_of_data_types: list of patterns in a regexp form such as
    ["*Amplitude*","*Phase*"] or ["PSI_Amplitude","PEI_Amplitude","meta_data"]
    :param list_of_labels: list of patterns in a regexp form such as
    [0,1,-1]
    :param dict_of_label_ratio: dictionary defining the percentage of points of each label, default = 1 (100%)
    ### TODO ### :param interpolate: percentage defining the number of points describing a pulse, default = 1 (100%)
    :return:  matrix with the shape (number of pulses x number of type of signals (channels) x length of signal)
    number of pulses depend on :
    - the size of the list in each pickle
    - the number of pickle we want to append (different label and different timestamp)
    """

    print("Path of pickled data:", path)
    print("Predicted path of raw data:", path[:-10] + "/")
    print("")

    # # one should no use the raw data folder because one does not want to store it just for this
    # listOfDictOfFiles = group_files_from_a_directory(path[:-10] +"/") #root path
    # list_of_timestamps_available = [listOfDictOfFile["timestamp"] for listOfDictOfFile in listOfDictOfFiles]

    list_of_files = glob.glob(path + "*__healthy__context_data.p.gz")
    list_of_timestamps_available = [file.split(path)[1][0:8] for file in list_of_files]

    list_of_timestamps = [ts for ts in list_of_timestamps_available if any(xs in ts for xs in list_of_timestamps_regex)]

    firstTimestamp = True
    id_timestamp = 0
    for timestamp in list_of_timestamps:
        id_timestamp += 1
        print("Reading timestamp ", timestamp, " (", id_timestamp, '/', len(list_of_timestamps), ')')
        if not firstTimestamp:
            print("   Shape of the datafame = ", df.shape)

        firstLabel = True
        for data_label in dict_of_label_ratio.keys():

            firstDataType = True
            for data_type in list_of_data_types:
                fileName = pickle_name(data_label, data_type, timestamp)
                df_new_channel = pd.read_pickle(path + fileName)

                if data_type == "context_data":
                    df_new_channel = pd.DataFrame(df_new_channel["data_list"])

                if firstDataType:
                    firstDataType = False
                    df_new_label = df_new_channel
                else:
                    cols_to_use = df_new_channel.columns.difference(df_new_label.columns)
                    df_new_label = df_new_label.merge(df_new_channel[cols_to_use], left_index=True, right_index=True,
                                                      sort=False)

            if df_of_pulses_selected is None:

                if firstLabel:
                    firstLabel = False
                    df_new_timestamp = df_new_label.sample(frac=dict_of_label_ratio[data_label],
                                                           random_state=0)
                else:
                    df_new_timestamp = df_new_timestamp.append(df_new_label.sample(frac=dict_of_label_ratio[data_label],
                                                                                   random_state=0),
                                                               ignore_index=True, sort=False)

            else:

                if firstLabel:
                    firstLabel = False
                    df_new_timestamp = pd.merge(df_new_label, df_of_pulses_selected, on='timestamp')
                else:
                    df_new_timestamp = df_new_timestamp.append(
                        pd.merge(df_new_label, df_of_pulses_selected, on='timestamp'),
                        ignore_index=True, sort=False)

        # parsing the feature / colmuns
        for feature in df_new_timestamp.columns:
            # if the feature is not empty
            if len(df_new_timestamp[feature]) > 0:
                # if we deal with a time series
                if isinstance(df_new_timestamp[feature].iloc[0], np.ndarray):
                    # if the time series is smaller than 3200 points
                    if len(df_new_timestamp[feature].iloc[0]) < 3200:

                        # we upscale

                        # general way

                        # TODO: some ".apply" function could be used ?! still it's not too slow for now
                        x_low = np.linspace(0, 1, num=len(df_new_timestamp[feature].iloc[0]), endpoint=True)
                        x_high = np.linspace(0, 1, num=3200, endpoint=True)
                        for pulse_id in range(len(df_new_timestamp[feature])):
                            # https: // docs.scipy.org / doc / scipy / reference / tutorial / interpolate.html
                            f = interp1d(x_low, df_new_timestamp[feature].iloc[pulse_id], kind=interp_kind)
                            df_new_timestamp[feature].iloc[pulse_id] = f(x_high)

                    # if we need to downscale
                    if take_nth_sample != 1:
                        for pulse_id in range(len(df_new_timestamp[feature])):
                            # TODO: interpolation should be used
                            df_new_timestamp[feature].iloc[pulse_id] = df_new_timestamp[feature].iloc[pulse_id][
                                                                       ::take_nth_sample]

        if firstTimestamp:
            firstTimestamp = False
            df = df_new_timestamp
        else:
            df = df.append(df_new_timestamp, ignore_index=True, sort=False)

    return df


def select_timestamp_wrt_feature_and_class(df_input, dict_of_features):
    # function which returns the timestamp of pulses according to a given dict_of_features
    # the output is a concactenation which is equivalent to an or condition, some refiltering might be necessary

    df_output = pd.DataFrame()
    for feature, dict_of_ratio in dict_of_features.items():
        for classe, ratio in dict_of_ratio.items():
            df_output = pd.concat(
                [df_output,
                 df_input[df_input[feature] == classe].sample(frac=ratio, random_state=42)
                 ]
            )

    df_output = df_output.drop_duplicates(subset="timestamp")[['timestamp']]
    return df_output


def time_series_to_spectrogram(df, settings, use_log=True):
    fs = 12e12  # 12 GHz
    window = settings["spectrogram"]["window"]  # default is tukey
    nperseg = settings["spectrogram"]["nperseg"]  # size of the window, should not be over 64
    offset = settings["spectrogram"]["offset"]  # jump between windows
    noverlap = nperseg - offset  # overlap between windows
    # maybe using the default value? but if there is some overlap, some information could be present twice in the data
    detrend = settings["spectrogram"][
        "detrend"]  # default is constant, linear could get more information out of the spectrogram (less smooth)

    for feature in settings['list_of_data_types']:

        # create the new feature
        for mode in range(int(nperseg / 2 + 1)):
            df[feature + "_mode_" + str(mode)] = 0
            df[feature + "_mode_" + str(mode)] = df[feature + "_mode_" + str(mode)].astype(object)  # quick and dirty

        # compute pulse by pulse the new features
        for ipulse in range(len(df)):
            f, t, Sxx = signal.spectrogram(df[feature].iloc[ipulse],
                                           fs,
                                           nperseg=nperseg,
                                           noverlap=noverlap,
                                           detrend=detrend,
                                           window=window
                                           )
            # allocate the new features
            for mode in range(int(nperseg / 2 + 1)):

                array = np.log10(Sxx[mode, :])

                if use_log:
                    array[array == np.inf] = np.finfo(np.float32).max
                    array[array == np.nan] = np.finfo(np.float32).min  # nan is most likely due to a 0 in log
                    array[array == -np.inf] = np.finfo(np.float32).min
                    df[feature + "_mode_" + str(mode)].iloc[ipulse] = array
                else:
                    df[feature + "_mode_" + str(mode)].iloc[ipulse] = array

    return df
