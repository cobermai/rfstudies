import nptdms
import pandas as pd
import numpy as np
import os.path


def path_to_pd(path: str,
               format_type: str = "matrix_of_vectors",
               ch_of_interest=None,
               grp_of_interest=None) -> pd.DataFrame:
    """
    This function reads a tdms file and calls tdmsfile_to_pd to convert it to a DataFrame.
    The format type options are:
    "matrix_of_vectors": no further requirements to the tdms file format.
        In each cell of the dataframe a vector with the channel data is placed.
    "matrix_of_scalars": requires all channels to have the same length.
        In each column the channel data of all groups with the same channel name are concatenated.
    :param path: The path of the .tdms file
    :param format_type: The format the pandas dataframe should have.
    :param ch_of_interest: The channels that should be included in the channels of interest.
    """
    if os.path.isfile(path):
        with nptdms.TdmsFile.read(path) as tdmsfile:
            return tdmsfile_to_pd(tdms=tdmsfile,
                format_type=format_type,
                ch_of_interest={ch.name for ch in tdmsfile.groups()[0].channels()}
                        if ch_of_interest is None else set(ch_of_interest),
                grp_of_interest={grp.name for grp in tdmsfile.groups()}\
                        if grp_of_interest is None else set(grp_of_interest))
    else:
        raise Exception("The given path does not exist.")


def tdmsfile_to_pd(tdms: nptdms.TdmsFile,
                   format_type: str = "matrix_of_vectors",
                   ch_of_interest=None,
                   grp_of_interest=None,
                   test_interests=False) -> pd.DataFrame:
    """
    This function transforms a tdmsfile into a pandas data frame. The output channels can be restricted to the
     channels of interest.
    "matrix_of_vectors": no further requirements to the tdms file format.
        In each cell of the dataframe a vector with the channel data is placed.
    "matrix_of_scalars": requires all channels to have the same length.
        In each column the channel data of all groups with the same channel name are concatenated
    :param tdms: tdms file to be converted, already opened.
    :param format_type: The format type the pandas dataframe should have. One of {"matrix_of_vectors", "matrix_of_scalars"}
    :param ch_of_interest: The channels that should be included in the channels of interest.
    :param grp_of_interest: The groups that should be included in the channels of interest.
    :param test_interests: If True, tests if the channels of interest and group of interests can be transformed into a df.
    """

    ch_of_interest = {ch.name for ch in tdms.groups()[0].channels()} if ch_of_interest is None else set(ch_of_interest)
    grp_of_interest = {grp.name for grp in tdms.groups()} if grp_of_interest is None else set(grp_of_interest)
    if test_interests:
        # Testing if the channel names are all similar. The differing ones will not be added to the df.
        tmp_ch_names = ch_of_interest
        for grp in tdms.groups():
            ch_of_interest.intersection_update([ch.name for ch in grp.channels()])
        if ch_of_interest != tmp_ch_names:
            print("The channels given do not exist in the ch_of_interest or differ from group to group, " +
                  "the corrupt channels will not be added to the dataframe.")

        # Testing if the group names are all similar. The differing ones will not be added to the df.
        if not grp_of_interest.issubset({grp.name for grp in tdms.groups()}):
            print("Not all groups given exist in the tmds file, the corrupt ones will not be added to the df.")
            grp_of_interest = grp_of_interest.intersection_update({grp.name for grp in tdms.groups()})

    if format_type == "matrix_of_scalars":  # in xbox, use this for TrendData
        df = pd.DataFrame(columns=ch_of_interest)  # initializing the DataFrame
        for grp_name in grp_of_interest:
            df = df.append(tdmsgroup_to_pd(tdms[grp_name]), ch_of_interest)
    elif format_type == "matrix_of_vectors":  # in xbox, use this for EventData
        # WARNING code is not beautiful, because pd.DataFrames are 2-dim but the data is not. Pandas wants to
        # automate writing in vectors as scalars, so a workaround had to be found.
        df = pd.DataFrame(index=grp_of_interest, columns=ch_of_interest.union({"tmp"}), dtype=object)
        for grp_name in grp_of_interest:
            if 0 == sum([len(ch.data) for ch in tdms[grp_name].channels()]):
                actual_channels = list(ch_of_interest.intersection({ch.name for ch in tdms[grp_name].channels()}))
                df.loc[[tdms[grp_name].name], actual_channels] = np.full([len(actual_channels)], np.NaN)
            else:
                actual_channels = list(ch_of_interest.intersection({ch.name for ch in tdms[grp_name].channels()}))
                df.loc[[tdms[grp_name].name], actual_channels + ["tmp"]] = np.array(
                    [tdms[grp_name][ch_name].data for ch_name in actual_channels] + [np.NaN], dtype=object)
        df.drop(columns=["tmp"], inplace=True)
    else:
        raise ValueError("The format_type you have given is unknown.")

    return df


def tdmsgroup_to_pd(grp: nptdms.TdmsGroup,
                    ch_of_interest=None,
                    test_interests=False,
                    index_ch: str = None) -> pd.DataFrame:
    """
    This function transforms a tdms group into a pandas data frame. The output channels can be restricted to the
    channels of interest.
    :param grp: tdms group to be converted, already opened.
    :param index_ch: which channel should be the intex in the pd.DataFrame. If None, indices are generic.
    :param ch_of_interest: The channels that should be included in the channels of interest.
    :param test_interests: If True, tests if the channels of interest can be transformed into a df.
    """
    # Testing if the channels of interest do all exist in the group and discarding unused ones
    if test_interests and ch_of_interest != ch_of_interest.intersection({ch.name for ch in grp.channels()}):
        print(ch_of_interest)
        print("The channels of interreset above do not all occure in " + grp.name +
              ". The none existing ones are discarted.")
        ch_of_interest.intersection_update({ch.name for ch in grp.channels()})

    # Giving an index List only makes sence if the there is no channel with more (longer) data.

    max_len = max([len(ch.data[:]) for ch in grp.channels()])
    if index_ch is None:
        index_list = list(range(max_len))
    else:
        if len(grp[index_ch][:]) != max_len:
            raise IndexError("There is a channel in the group " + grp.name + " longer than your given index channel.")
        else:
            index_list = list(range(max_len))

    df = pd.DataFrame(index=index_list, columns=[ch.name for ch in grp.channels()])
    for ch_name in ch_of_interest:
        df[grp[ch_name].name] = grp[ch_name].data
    return df
