import nptdms
import numpy as np
import pandas as pd
import time
import glob
import matplotlib
import matplotlib.pyplot as plt

inpath = "/home/lfischl/project_data/CLIC_DATA_Xbox2_T24PSI_2/"
paths = glob.glob(inpath + "Event*.tdms")
tdms = nptdms.TdmsFile(paths[70])
grp = tdms.groups()[100]
print(grp.name)

ch_name_w_rate = lambda grp,x: [ch.name for ch in grp.channels() if ch.properties["wf_samples"]==x]
ch_data_w_rate = lambda grp,x: [ch.data for ch in grp.channels() if ch.properties["wf_samples"]==x]
df500 = pd.DataFrame(ch_data_w_rate(grp,500), index = ch_name_w_rate(grp,500)).T
df3200 =  pd.DataFrame(ch_data_w_rate(grp,3200), index = ch_name_w_rate(grp,3200)).T


pltsize = 20
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
cnt = 0
for clm in df500.columns:
    cnt+=1
    plt.subplot(2,4,cnt)
    plt.plot(df500[clm], label = clm)
    plt.legend()
plt.show()

pltsize = 20
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
cnt = 0
for clm in df3200.columns:
    cnt+=1
    plt.subplot(2,4,cnt)
    plt.plot(df3200[clm], label = clm)
    plt.legend()
plt.show()


pltsize = 10
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
for clm in df500.columns:
    if clm != "Col.":
        plt.plot( df500[clm], label = clm)
        plt.legend()
plt.show()

pltsize = 30
plt.figure(figsize=(pltsize * 2 ** (1 / 2), pltsize), dpi=100)
list_bd_grps = [grp for grp in tdms.groups() if "Breakdown" in grp.name]
list_log_grps = [grp for grp in tdms.groups() if "Log" in grp.name]
for cnt in range(8):
    grp = list_bd_grps[np.random.randint(len(list_bd_grps))]
    df3200 = pd.DataFrame(ch_data_w_rate(grp,3200), index=ch_name_w_rate(grp,3200)).T
    plt.subplot(4,2,cnt+1)
    for clm in df3200.columns:
        if clm != "":
            plt.plot( df3200[clm], label = clm)
            plt.legend()

    df500 = pd.DataFrame(ch_data_w_rate(grp, 500), index=ch_name_w_rate(grp, 500)).T


    grp = list_log_grps[np.random.randint(len(list_log_grps))]
    df3200 = pd.DataFrame(ch_data_w_rate(grp,3200), index=ch_name_w_rate(grp,3200)).T
    plt.subplot(4,2,cnt+1)
    for clm in df3200.columns:
        if clm != "":
            plt.plot( df3200[clm], label = clm)
            plt.legend()
plt.show()

