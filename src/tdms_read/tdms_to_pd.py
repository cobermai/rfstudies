import nptdms
import pandas as pd
import os.path

def get_pd(path: str, format: str = "matrix_of_vectors", ch_of_interrest: list = None) -> pd.DataFrame:
    """
    This function reads a tdms file and calls tdmsfile_to_pd to convert it to a DataFrame.
    The format options are:
    "matrix_of_vectors": no further requirements to the tdms file format.
        In each cell of the dataframe a vector with the channel data is placed.
    "matrix_of_scalars": requires all channels to have the same length.
        In each column the channel data of all groups with the same channel name are concatenated.
    """
    if os.path.isfile(path):
        with nptdms.TdmsFile.read(path) as tdmsfile:
            print("The file has been read")
            return tdmsfile_to_pd(tdmsfile, format, ch_of_interrest)
    else:
        raise Exception("The given path does not exist.")

def tdmsfile_to_pd(tdms:nptdms.TdmsFile, format: str = "matrix_of_vectors", ch_of_interrest: list = None) -> pd.DataFrame:
    """
    This function transforms a tdmsfile into a pandas data frame. In order to be able to write all tdms data into one
    dataframe, all groups have to have the same channels. The output channels can be restricted to the channels of
    interrest..
    ch_of_interrest: list of strings of the channels names
    format: string, one of {"matrix_of_vectors", "matrix_of_scalars"}
    "matrix_of_vectors": no further requirements to the tdms file format.
        In each cell of the dataframe a vector with the channel data is placed.
    "matrix_of_scalars": requires all channels to have the same length.
        In each column the channel data of all groups with the same channel name are concatenated
    """

    ### Testing if the channel names are all similar. If they differ from the first one, they will not be added to the df.
    if ch_of_interrest == None: ch_of_interrest = [ch.name for ch in tdms.groups()[0].channels()]

    ch_name_list = set(ch_of_interrest)
    for grp in tdms.groups(): ch_name_list.intersection_update()

    if ch_of_interrest != ch_name_list: print("The channels are not all the same, the corrupt ones will not be added to the dataframe.")


    df = pd.DataFrame(columns=ch_name_list)  # initializing the DataFrame
    if format == "matrix_of_scalars":  # in xbox, use this for TrendData
        for grp in tdms.groups():
            df = df.append(tdmsgroup_to_pd(grp), ch_name_list)
    elif format == "matrix_of_vectors":  # in xbox, use this for EventData
        for grp in tdms.groups():
            df.loc[grp.name] = [grp[ch_name].data for ch_name in ch_name_list]
            # if a channel is scaled you can rescale it by using ch.read_data(scaling=True), but the channel properties
            # have to be the generic NI ones in order to work. For further information read nptdms/scaling.py
    else:
        print("The format you have given is unknown")

    return df

def tdmsgroup_to_pd(grp:nptdms.TdmsGroup, index_ch: str = None) -> pd.DataFrame:
    # It is required that all channels have the same length

    ### Testing if the channels all have the same length and warning if this is not the case.
    # filling the dataframe will fail if this is not the case.
    length_ch = len(grp.channels()[0][:])
    for ch in grp.channels():
        if length_ch != len(ch[:]): print("WARNING: The channels have differnt lengths, they can not be concatenated, this will lead to an error.")

    index_list = list(range(length_ch)) if index_ch==None else grp[index_ch]
    df = pd.DataFrame(index= index_list, columns = [ch.name for ch in grp.channels()])
    for ch in grp.channels():
        df[ch.name] = ch.data
        # if a channel is scaled you can rescale it by using ch.read_data(scaling=True), but the channel properties
        # have to be the generic NI ones in order to work. For further information read nptdms/scaling.py
    return df