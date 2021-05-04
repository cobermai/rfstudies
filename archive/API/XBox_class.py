import os
import glob
import datetime
from compress_pickle import dump as cpickle_dump
import pandas as pd
import numpy as np
import nptdms
from gc import collect
from time import time
from multiprocessing import Pool
from src.utils.system.logger import logger
from archive.tdms_reader.tdms_to_pd import tdmsfile_to_pd
log = logger("DEBUG", None)

def get_ts(name: str) -> str:
    if os.path.isfile(name):  # if the input is a path
        return os.path.split(name)[1][:-5].split("_")[1]
    elif name[-5:] == ".tdms" and "/" not in name:  # if input is only a filename not a path
        return name[:-5].split("_")[1]
    else:
        log.error("The function get_ts only works for tdms filenames and pahts")

class _DataSet:
    name: str = None
    abbreviation: str = None
    file_paths: list = None
    ch_of_interest: list = None
    def __init__(self, name: str, abbreviation: str, file_paths: list, ch_of_interest: list):
        self.name = name
        self.abbreviation = abbreviation
        self.file_paths = file_paths
        self.ch_of_interest = ch_of_interest


class XBoxDataSet:
    """
    This is the superclass of for a whole XBox dataset. For every Test stand (currently XBox2 and XBox3) there is one
    subclass with the test-stand specific needs.
    One object of this class represents one folder with tdms files of event and trend data.
    """
    dir_path: str = None # path of the folder where the tdms files are stored
    dest_path = None  # where the outputfiles (eg. compressed_pickle files) should be stored

    def __init__(self, dir_path: str, dest_path: str):  # TODO pass settings as dict. (see json)
        if os.path.isdir(dir_path) and os.path.isdir(dest_path):
            self.dest_path = dest_path
            self.dir_path = dir_path
        else:
            log.error("The directory path does not exist.")

    def get_unique_timestamp(self):
        td_file_paths = [os.path.split(filepath)[1] for filepath in glob.glob(self.dir_path + "Trend*.tdms")]
        ed_file_paths = [os.path.split(filepath)[1] for filepath in glob.glob(self.dir_path + "Event*.tdms")]
        ed_ts = set([get_ts(path) for path in td_file_paths])
        td_ts = set([get_ts(path) for path in ed_file_paths])
        return ed_ts.symmetric_difference(td_ts)


class XBox2DataSet(XBoxDataSet):
    number_of_processes: int = 1
    eds: _DataSet = None
    tds: _DataSet = None
    daystamp_list: list = None
    def __init__(self, settings: dict):
        super().__init__(settings["dir_path"], settings["dest_path"])  # initializes the template class XBoxDS
        self.eds = _DataSet(name = "EventData",
                      abbreviation = "ed",
                      file_paths = glob.glob(self.dir_path + "Event*.tdms"),
                      ch_of_interest = settings["ed_ch_of_interest"])
        self.tds = _DataSet(name = "TrendData",
                      abbreviation = "td",
                      file_paths = glob.glob(self.dir_path + "Trend*.tdms"),
                      ch_of_interest = settings["td_ch_of_interest"])
        self.daystamp_list = list({get_ts(path) for path in self.tds.file_paths}.intersection({get_ts(path) for path in self.tds.file_paths}))
        self.number_of_processes = settings["number_of_workers_for_paralell_computation"]

    def partial_function(self, daystamp) -> None: XBox2DataTuple(self,daystamp).transform()

    def transform(self, proc_list= None):
        if proc_list != None: self.daystamp_list = [self.daystamp_list[i] for i in proc_list]
        if self.number_of_processes ==1:
            log.debug("serial computing")
            for daystamp in self.daystamp_list:
                XBox2DataTuple(self, daystamp).transform()
        elif self.number_of_processes > 1:
            log.debug("computing with " + str(self.number_of_processes) + " processes in paralell")
            with Pool(processes=self.number_of_processes) as pool:
                pool.map(self.partial_function, self.daystamp_list)

    def _scale_channel(self, ch: nptdms.TdmsChannel) -> None:
        """
        The scaling arguments have to be the generic NI ones in order to work with the nptdms package. The channel
        properties are changed here such that they fit they are generic.
        """
        if ch.properties.get("Scale_Type", None) == "Polynomial":
            add_scaling_prop = {
                "NI_Scaling_Status": "unscaled",
                "NI_Number_Of_Scales": 1,
                "NI_Scale[0]_Scale_Type": ch.properties["Scale_Type"],
                "NI_Scale[0]_Polynomial_Coefficients_Size": 3}
            for i in range(add_scaling_prop['NI_Scale[0]_Polynomial_Coefficients_Size']):
                add_scaling_prop["NI_Scale[0]_Polynomial_Coefficients[%d]" % i] = ch.properties[
                    "Scale_Coeff_c" + str(i)]
            ch.properties.update(add_scaling_prop)



class XBoxDataTuple:
    """
    This is the superclass of one XBox data tuple. For each day of recording there is trend data and event data.
    Both together make up a tuple of xbox data.  For every Test stand (currently XBox2 and XBox3) there is one
    subclass with the test-stand specific needs.
    One object of this class represents two tdms files, event and trend data.
    """
    pset = None  # parent Datas Set
    datestamp: str = None  # timestamp of day the data as a string
    date: datetime.datetime = None  # timestamp converted to a datetime format
    ed = None
    cd = None
    def __init__(self, parent_ds, daystamp: str) -> None:
        self.pset = parent_ds
        self.daystamp = daystamp
        self.date = datetime.datetime.strptime(daystamp, "%Y%m%d")

class XBox2DataTuple(XBoxDataTuple):
    def __init__(self, parent_ds: XBox2DataSet, daystamp: str) -> None:
        super().__init__(parent_ds, daystamp)

    def transform(self):
        log.info("Transforming data of the day " + self.daystamp)
        self.ed = EventData(self)
        self.ed.API()
        log.debug("EventData of " + self.daystamp + " has been transformed")
        self.cd = ContextData(self)
        self.cd.API()
        log.debug("Trend + ContextData of " + self.daystamp + " has been transformed")



class XBoxData():
    type: str = None
    ptuple: XBoxDataTuple = None
    pset = None
    def __init__(self, ptuple, type: str):
        if type not in ["Event", "Trend"]:
            log.error("type must be either \"Event\" or \"Trend\" , but " + type + " has been passed to XBoxData.__init__")
        else:
            self.type = type
        self.ptuple = ptuple
        self.pset = self.ptuple.pset

    def get_file_name(self) -> str: return self.type + "Data_" + self.ptuple.daystamp + ".tdms"
    def get_file_path(self) -> str: return self.pset.dir_path + self.get_file_name()

class EventData(XBoxData):
    index_list: list = None  # a list of group names, used as index for the context_data dataframe
    context_data: pd.DataFrame = None  # a dataframe containing contextdata (descriptive statistic of the event data)
    def __init__(self, ptuple: XBox2DataTuple):
        super().__init__(ptuple, "Event")

    def _set_context_data(self, tdmsfile: nptdms.TdmsFile):
        starttime_exists = lambda grp: all(["wf_start_time" in list(ch.properties.keys()) for ch in grp.channels()])
        t = time()
        self.index_list = [grp.name for grp in tdmsfile.groups() if starttime_exists(grp)]
        self.context_data = pd.DataFrame([tdmsfile[grp_name]["DC Down"].properties["wf_start_time"] for grp_name in self.index_list],
                     index = self.index_list,
                     columns=["starttime"])

        # Calculating the "pulse length" is computationally expensive, this is the reason this part takes so long.
        # Using .apply was used to try to speed this part up, but that didnt work.
        additional_df = pd.DataFrame([self.descr_stat_data(tdmsfile[grp_name]) for grp_name in self.index_list],
                     index=self.index_list,
                     columns=self.descr_stat_clmname(tdmsfile.groups()[0]))
        self.context_data = self.context_data.merge(additional_df, right_index=True, left_index=True, how="inner")

    def API(self, light_memory: bool = True) -> None:
        with nptdms.TdmsFile.read(self.get_file_path()) as tdmsfile:
            log.debug("Finished reading EventData of " + self.ptuple.daystamp)
            self._set_context_data(tdmsfile)  # and index list

            data = pd.DataFrame(index=self.index_list, columns=["starttime"])
            data["starttime"] = self.context_data["starttime"]
            if light_memory:  # faster for big tdms files
                data["init"] = np.nan
                for ch_name in self.pset.eds.ch_of_interest:
                    t0 = time()
                    data.columns = ["starttime", ch_name]
                    data_list = []
                    for grp_name in self.index_list:
                        data_list.append(self._get_data(tdmsfile[grp_name][ch_name]))
                    data[ch_name] = data_list
                    #data[ch_name] = tdmsfile_to_pd(tdms= tdmsfile,
                    #                               format= "matrix_of_vectors",
                    #                               ch_of_interest= [ch_name],
                    #                               grp_of_interest = self.index_list,
                    #                               test_tdmsfiledata= False)
                    self._write_cpickle_breakdown(ch_name, data)
                    self._write_cpickle_healthy(ch_name, data)
                    log.debug(str(time() - t0) + " sek work for " + ch_name)
            else:  # faster for small tmds files
                data = tdmsfile_to_pd(tdms= tdmsfile,
                                      format= "matrix_of_vectors",
                                      ch_of_interest= self.pset.eds.ch_of_interest,
                                      grp_of_interest = self.index_list)
                #data[self.pset.eds.ch_of_interest.copy()] = np.nan
                #for grp_name in self.index_list:
                #    grp = tdmsfile[grp_name]
                #    for ch_name in self.pset.eds.ch_of_interest:
                #        data.loc[grp.name, ch_name] = self._get_data(grp[ch_name])

                for ch_name in self.pset.eds.ch_of_interest:
                    self._write_cpickle_breakdown(ch_name, data)
                    self._write_cpickle_healthy(ch_name, data)
            del data
            collect()  # let the garbage collector come (from module gc)

    def descr_stat_clmname(self, grp: nptdms.TdmsGroup) -> list:
        clm_list = []
        for ch_name in ["PKI Amplitude", "PEI Amplitude", "PSI Amplitude"]:
            clm_list += [ch_name + "__recomputed_amplitude", ch_name + "__recomputed_duration"]
        for ch_name in ["DC Down", "DC Up"]:
            clm_list += [ch_name + "__recomputed_" + x for x in
                         ["D1", "median", "D9", "minimum", "maximum", "average"]]
        return clm_list

    def descr_stat_data(self, grp: nptdms.TdmsGroup) -> list:
        """
        This function adds some discriptive statistic of the data in speciffic channels
        """
        pulse_amplitude = lambda amp: np.median(amp[amp > amp.max() / 2])

        pulse_duration = lambda amp: sum(amp > amp.max() / 2) / 12.
        ret = []
        for ch_name in ["PKI Amplitude", "PEI Amplitude", "PSI Amplitude"]:
            amp = self._get_data(grp[ch_name])
            ret += [pulse_amplitude(amp), pulse_duration(amp)]

        for ch_name in ["DC Down", "DC Up"]:
            dc = self._get_data(grp[ch_name])
            ret += [np.quantile(dc, .10), np.quantile(dc, .50), np.quantile(dc, .90), np.min(dc), np.max(dc),
                    np.mean(dc)]

        return ret

    def _get_data(self, ch: nptdms.TdmsChannel) -> np.ndarray:
        self.pset._scale_channel(ch)  # some channels are unscaled
        data = ch.data if ch.properties.get("Scale_Type", None) != None else ch.read_data(scaled=True)
        return np.array(data)

    def _write_cpickle_breakdown(self, ch_name, df: pd.DataFrame) -> None:
        ch__name = ch_name.replace(" ", "_")  # we would like to use channel names with underscore
        cpickle_filepath = self.pset.dest_path + self.ptuple.daystamp + "__" + "breakdown" + "__" + ch__name + ".p.gz"
        bd_index_list = df.index.str.contains("Breakdown")
        df.rename(columns={"starttime": "timestamp", ch_name:ch__name})[bd_index_list][["timestamp", ch__name]] \
            .to_pickle(cpickle_filepath, "gzip")

    def _write_cpickle_healthy(self, ch_name: str, df: pd.DataFrame) -> None:
        ch__name = ch_name.replace(" ", "_")  # we would like to use channel names with underscore
        cpickle_filepath = self.pset.dest_path + self.ptuple.daystamp + "__" + "healthy" + "__" + ch__name + ".p.gz"
        h_index_list = df.index.str.contains("Log")
        df.rename(columns={"starttime": "timestamp", ch_name:ch__name})[h_index_list][["timestamp", ch__name]] \
            .to_pickle(cpickle_filepath, "gzip")


class ContextData(XBoxData):
    df: pd.DataFrame = None
    def __init__(self, ptuple: XBox2DataTuple):
        super().__init__(ptuple, "Trend")

    def API(self):
        with nptdms.TdmsFile.read(self.get_file_path()) as tdmsfile:
            log.debug("Finished reading TrendData of " + self.ptuple.daystamp)
            self.df = pd.DataFrame(columns = self.pset.tds.ch_of_interest)
            for grp in tdmsfile.groups(): # there is only one group in the trend data files
                tmpdf = pd.DataFrame(columns = self.pset.tds.ch_of_interest)
                if "Timestamp" not in self.pset.tds.ch_of_interest: log.error("\"Timestamp\" has to be in the td.ch_of_interest")
                for ch_name in self.pset.tds.ch_of_interest:
                    tmpdf[ch_name] = grp[ch_name].data
                self.df = self.df.append(tmpdf)
            self.df.to_pickle(self.pset.dest_path + "TrendData_" + self.ptuple.daystamp + ".p.gz", "gzip")

            ed_starttimes = self.ptuple.ed.context_data["starttime"].to_numpy(dtype=int)
            ed_starttimes = ed_starttimes.reshape((len(ed_starttimes), 1))  #clm vector
            td_starttimes = self.df["Timestamp"].to_numpy(dtype = int)
            td_starttimes = td_starttimes.reshape((1, len(td_starttimes)))  # row vector
            min_location = np.where(ed_starttimes > td_starttimes , td_starttimes, -np.inf).argmax(axis = 1)
            if ed_starttimes[0,0] < td_starttimes[0,0]:
                first_event_has_trend_data = self.get_prev_day_trend_data()
                if first_event_has_trend_data:
                    min_location = np.where(ed_starttimes > td_starttimes , td_starttimes, -np.inf).argmax(axis = 1)
                else:
                    min_location = min_location[1:]
                    # The join of the trend and event data happens with a inner join, so the first event data is removed
            self.df = self.df.loc[min_location]
            self.df.reset_index(drop=True, inplace=True)  # forget old index, the now ones fit with ed
            context_df = self.ptuple.ed.context_data
            self.df = self.df.merge(context_df.loc[:, context_df.columns != "starttime"].reset_index(), left_index=True, right_index=True, how="inner") # merge with eventdata on the new indices
            self.df.set_index("index", inplace=True)  # use the event data group names as indices
            self.df.rename(columns={name: name.replace(" ", "_") for name in self.df.columns}, inplace=True)  # make _ in clm names

            self._write_cpickle_breakdown()
            self._write_cpickle_healthy()

    def get_prev_day_trend_data(self):
        log.info("Some Event was happending before the first TrendData entry of the day " + self.ptuple.daystamp)
        year = int(self.ptuple.daystamp[0:3])
        month = int(self.ptuple.daystamp[3:5])
        day = int(self.ptuple.daystamp[5:7])
        prev_day_daystamp = (datetime.date(year, month, day) + datetime.timedelta(-1)).strftime("%Y%m%d")
        prev_day_ed_file_path = self.pset.dir_path  + "TrendData_" + prev_day_daystamp + ".tdms"
        if os.path.isfile(prev_day_ed_file_path):
            log.debug("Reading the prev day TrendData in order to get the latest data set.")
            tdmsfile = nptdms.TdmsFile.read(prev_day_ed_file_path)
            grp = tdmsfile.groups()[0]
            data_list = [ch.data[-1] for ch in [grp[ch_name] for ch_name in self.pset.tds.ch_of_interest]]
            self.df.loc[-1] = data_list
            self.df.sort_index()
            self.df.reset_index(drop=True, inplace=True)
            return True
        else:
            log.warning("No previouse day TrendData found, skipping the first EventData")
            return False


    def _write_cpickle_breakdown(self) -> None:
        cpickle_filepath = self.pset.dest_path + self.ptuple.daystamp + "__breakdown__context_data.p.gz"
        bd_index_list = self.df.index.str.contains("Breakdown")
        td_dict = {"data_list": self.df[bd_index_list].to_dict(orient="records"),
                "name_list": self.df[bd_index_list].index.to_list(),
                "data_label": 0,
                "data_type": "context_data",
                "timestamp": self.ptuple.daystamp}
        cpickle_dump(td_dict, cpickle_filepath, "gzip")

    def _write_cpickle_healthy(self) -> None:
        cpickle_filepath = self.pset.dest_path + self.ptuple.daystamp + "__healthy__context_data.p.gz"
        h_index_list = self.df.index.str.contains("Log")
        td_dict = {"data_list": self.df[h_index_list].to_dict(orient="records"),
                   "name_list": self.df[h_index_list].index.to_list(),
                   "data_label": 1,
                   "data_type": "context_data",
                   "timestamp": self.ptuple.daystamp}
        cpickle_dump(td_dict, cpickle_filepath, "gzip")