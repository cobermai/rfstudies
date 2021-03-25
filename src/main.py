from XBox_class import *
SHOW_PROGRESS_BAR = "tqdm"
from time import time
t0 = time()
guesses = [pre + "CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2/" for pre in ["/home/lfischl/cernbox/", "/home/thomas/cernboxML/private/", "?Christoph/","?Andrea/"]] + ["/eos/project-m/ml-for-alarm-system/private/CLIC_data_transfert/"]
for possible_dir in guesses:
    if os.path.isdir(possible_dir):
        dir_path = possible_dir

settings = {
    "dir_path": "/home/lfischl/project_data/CLIC_DATA_Xbox2_T24PSI_2/" ,  # dir_path,
    "dest_path": "/home/lfischl/output_files/",  # if not set this will be defined later as xbox3/data
    "ed_ch_of_interest": ['DC Down',
                          'DC Up',
                          'PEI Amplitude',
                          'PEI Phase',
                          'PER log',
                          'PKI Amplitude',
                          'PKI Phase',
                          'PKR log',
                          'PSI Amplitude',
                          'PSI Phase',
                          'PSR Amplitude',
                          'PSR log',
                          'PSR Phase'],
    "td_ch_of_interest": ['Timestamp',
                          'Loadside win',
                          'Tubeside win',
                          'Collector',
                          'Gun',
                          'IP before PC',
                          'PC IP',
                          'WG IP',
                          'IP Load',
                          'IP before structure',
                          'US Beam Axis IP',
                          'Klystron Flange Temp',
                          'Load Temp',
                          'PC Left Cavity Temp',
                          'PC Right Cavity Temp',
                          'Bunker WG Temp',
                          'Structure Input Temp',
                          'Chiller 1',
                          'Chiller 2',
                          'Chiller 3',
                          'PKI FT avg',
                          'PSI FT avg',
                          'PSR FT avg',
                          'PEI FT avg',
                          'PKI max',
                          'PSI max',
                          'PSR max',
                          'PEI max',
                          'BLM TIA min',
                          'BLM min',
                          'DC Down min',
                          'DC Up min',
                          'BLM TIA Q',
                          'PSI Pulse Width',
                          'Pulse Count'],
    "number_of_workers_for_paralell_computation": 4
}
#tdmsfile = nptdms.TdmsFile.read("/home/lfischl/cernbox/CLIC_data_transfert/CLIC_DATA_Xbox2_T24PSI_2/TrendData_20180328.tdms")
xb2ds = XBox2DataSet(settings)
xb2ds.transform()

    #log.debug("For Day " + data.timestamp + " API took " + str(time.time() - t00) + " sek")  # or use %timeit

print("in total took (sek) " + str(time() - t0))  # or use %timeit

