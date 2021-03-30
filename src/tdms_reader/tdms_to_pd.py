import nptdms
import pandas as pd
import os.path

def path_to_pd(path: str, format: str = "matrix_of_vectors", ch_of_interest: list = None) -> pd.DataFrame:
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
            return tdmsfile_to_pd(tdmsfile, format, ch_of_interest)
    else:
        raise Exception("The given path does not exist.")

def tdmsfile_to_pd(tdms:nptdms.TdmsFile,
                   format: str = "matrix_of_vectors",
                   ch_of_interest= None,
                   grp_of_interest= None,
                   test_tdmsfiledata = True) -> pd.DataFrame:
    """
    This function transforms a tdmsfile into a pandas data frame. In order to be able to write all tdms data into one
    dataframe, all groups have to have the same channels. The output channels can be restricted to the channels of
    interest..
    ch_of_interest: list or set of strings or a single string of the channels names
    format: string, one of {"matrix_of_vectors", "matrix_of_scalars"}
    "matrix_of_vectors": no further requirements to the tdms file format.
        In each cell of the dataframe a vector with the channel data is placed.
    "matrix_of_scalars": requires all channels to have the same length.
        In each column the channel data of all groups with the same channel name are concatenated
    """

    ch_of_interest = {ch.name for ch in tdms.groups()[0].channels()} if ch_of_interest == None else set(ch_of_interest)
    grp_of_interest = {grp.name for grp in tdms.groups()} if grp_of_interest == None else set(grp_of_interest)
    if test_tdmsfiledata == True:
        ### Testing if the channel names are all similar. The differing ones will not be added to the df.
        tmp_ch_names = ch_of_interest
        for grp in tdms.groups(): ch_of_interest.intersection_update([ch.name for ch in grp.channels()])
        if ch_of_interest != tmp_ch_names: print("The channels given do not exist in the ch_of_interest or differ from group to group, the corrupt channels will not be added to the dataframe.")

        ### Testing if the group names are all similar. The differing ones will not be added to the df.
        if not grp_of_interest.issubset({grp.name for grp in tdms.groups()}):
            print("Not all groups given exist in the tmds file, the corrupt ones will not be added to the df.")
            grp_of_interest = grp_of_interest.intersection_update({grp.name for grp in tdms.groups()})

    df = pd.DataFrame(columns=ch_of_interest)  # initializing the DataFrame
    if format == "matrix_of_scalars":  # in xbox, use this for TrendData
        for grp in [tdms[grp_name] for grp_name in grp_of_interest]:
            df = df.append(tdmsgroup_to_pd(grp), ch_of_interest)
    elif format == "matrix_of_vectors":  # in xbox, use this for EventData
        for grp in [tdms[grp_name] for grp_name in grp_of_interest]:
            df.loc[grp.name] = [grp[ch_name].data for ch_name in ch_of_interest]
            # if a channel is scaled you can rescale it by using ch.read_data(scaling=True), but the channel properties
            # have to be the generic NI ones in order to work. For further information read nptdms/scaling.py
    else:
        print("The format you have given is unknown")

    return df

def tdmsgroup_to_pd(grp:nptdms.TdmsGroup, index_ch: str = None, ch_of_interest = None) -> pd.DataFrame:
    # It is required that all channels have the same length

    ### Testing if the channels all have the same length and warning if this is not the case.
    # filling the dataframe will fail if this is not the case.
    length_ch = len(grp.channels()[0][:])
    for ch in grp.channels():
        if length_ch != len(ch[:]): print("WARNING: The channels have differnt lengths, they can not be concatenated, this will lead to an error.")

    ch_of_interest = set(ch_of_interest)
    tmp_ch_names = ch_of_interest
    ch_of_interest.intersection_update(grp.channels())
    if ch_of_interest != tmp_ch_names: print(
        "The channels given do not exist in the ch_of_interest or differ from group to group, the corrupt channels will not be added to the dataframe.")

    index_list = list(range(length_ch)) if index_ch==None else grp[index_ch]
    df = pd.DataFrame(index= index_list, columns = [ch.name for ch in grp.channels()])
    for ch in [grp[ch_name] for ch_name in ch_of_interest]:
        df[ch.name] = ch.data
        # if a channel is scaled you can rescale it by using ch.read_data(scaling=True), but the channel properties
        # have to be the generic NI ones in order to work. For further information read nptdms/scaling.py
    return df


"""class TdmsReader():
    tdmsfile: nptdms.TdmsFile = None
    groupnames: set = None
    channelnames: set = None
    is_tested: bool = False
    format: str = "matrix_of_vectors"
    ch_of_interest: set = None
    grp_of_interest: set = None
    output_file = None  # for inplace operation

    def __init__(self,
                 tdms: nptdms.TdmsFile,
                 format: str = "matrix_of_vectors",
                 ch_of_interest=None,
                 grp_of_interest=None,
                 test_tdmsfiledata=True):
        self._set_tdmsfile(tdms)
        self.format = format
        if ch_of_interest == None:
            self.ch_of_interest = self.channelnames
        else:
            ch_of_interest.intersection(self.channelnames)
        self.grp_of_interest = self.groupnames if grp_of_interest==None else grp_of_interest.intersection(self.groupnames)

    def _set_tdmsfile(self, tdms):
        self.is_tested = False
        self.groupnames = {grp.name for grp in self.tdmsfile.groups()}
        self.channelnames = {ch.name for ch in tdms.groups()[0].channels()}
        for grp in tdms.groups(): self.channelnames.intersection_update({ch.name for ch in grp.channels()})
        self.tdmsfile = tdms

    def _test_channels(self):
        tmp_ch_names = self.ch_of_interest
        for grp in tdms.groups(): self.ch_of_interest.intersection_update([])
        if self.ch_of_interest != tmp_ch_names: print(
            "The channels given do not exist in the ch_of_interest or differ from group to group, the corrupt channels will not be added to the dataframe.")

    def _teset_groups(self):
        if not self.grp_of_interest.issubset():
            print("Not all groups given exist in the tmds file, the corrupt ones will not be added to the df.")
            self.grp_of_interest.intersection_update({grp.name for grp in self.tdmsfile.groups()})
            """