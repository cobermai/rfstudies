import glob
import os
import h5py
from tdms_read import ConverterToHdf
from filter import Filter
from utils.progress_bar import working_on
from utils.logger import logger
log = logger("DEBUG")
from virtual_hdf import VirtualHdf
import numpy as np

def transformation(tdms_dir: str, hdf_dir: str):
    ## read tdms
    ConverterToHdf(tdms_dir=tdms_dir,
                   hdf_dir=hdf_dir,
                   check_already_converted=True,
                   num_processes=1).run()

    ## create virtual Event and Trend hdf files
    td_file_path = hdf_dir + "../" + "TrendDataVirtual.vhdf"
    #VirtualHdf(td_file_path).add(glob.glob(hdf_dir + "Trend*.hdf"))

    ed_file_path = hdf_dir + "../" + "EventDataVirtual.vhdf"
    #VirtualHdf(ed_file_path).add(glob.glob(hdf_dir + "Event*.hdf"))

    ## Filter
    def td_filter_rule(file_path: str, hdf_path: str)->bool:
        # checking the data on the level of tdms groups
        ret = False
        try:
            f = h5py.File(file_path, "r+")
            ch_shapes = [f[hdf_path][key].shape[0] for key in f[hdf_path].keys()]
            len_equal = all([ch_shape == ch_shapes[0] for ch_shape in ch_shapes])
            ret = not len_equal or len(f[hdf_path].keys())!=35
            if not ret: ret = any([any(np.isnan(f[hdf_path][key][:])) for key in f[hdf_path].keys()])
        except OSError:  # error in opening the file, then file can not be closed
            log.warning("reading the hdf file failed in " + file_path + " " + hdf_path )
            ret = True
        except:  # error in the calculation file has to be closed
            log.warning("filter_rule error in " + file_path + " " + hdf_path)
            f.close()
            ret = True
        else:  # when no error occured file has to be closed
            f.close()
        finally:  # return the return value no matter Errors occurred
            return ret
    def on_filter_do(file_path: str, hdf_path: str) -> None:
        with h5py.File(file_path, "r+") as f:
            print("Deleting link " + hdf_path)
            del f[hdf_path]

    td_filter = Filter(filter_rule=td_filter_rule,
                       apply_on_layer = 1,
                       on_filter_do=on_filter_do)
    td_filter.apply(td_file_path)

    def ed_filter_rule(file_path: str, hdf_path: str)->bool:
        ret = False
        try:
            f = h5py.File(file_path, "r+")
            ret = f[hdf_path].attrs.get("Timestamp", None)==None
            if not ret:  # I dont want to do the expensive work when its not necessary
                ch_len = [f[hdf_path][key].shape[0] for key in f[hdf_path].keys()]
                ret = ch_len.count(500)!=8 or ch_len.count(3200)!=8
            if not ret:
                ret = any([any(np.isnan(f[hdf_path][key][:])) for key in f[hdf_path].keys()])

        except OSError:  # error in opening the file, then file can not be closed
            log.warning("reading the hdf file failed in " + file_path + " " + hdf_path )
            ret = True
        except:  # error in the calculation file has to be closed
            log.warning("filter_rule error in " + file_path + " " + hdf_path)
            f.close()
            ret = True
        else:  # when no error occured file has to be closed
            f.close()
        finally:
            return ret
    ed_filter = Filter(filter_rule=ed_filter_rule,
                       apply_on_layer = 1,
                       on_filter_do=on_filter_do)
    ed_filter.apply(ed_file_path)



if __name__=="__main__":
    transformation(tdms_dir = os.path.expanduser("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/"),
                   hdf_dir = os.path.expanduser("~/output_files/unfiltered/"))
    # transformation(tdms_dir = "/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2/",
    #               hdf_dir = "/eos/project/m/ml-for-alarm-system/private/CLIC_data_transfert/Xbox2_hdf/")