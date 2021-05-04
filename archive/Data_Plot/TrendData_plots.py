import nptdms
import numpy as np
import pandas as pd
import time
import glob
import matplotlib
import matplotlib.pyplot as plt

inpath = "/home/lfischl/project_data/CLIC_DATA_Xbox2_T24PSI_2/"
paths = glob.glob(inpath + "Trend*.tdms")
tdms = nptdms.TdmsFile(paths[70])
grp = tdms.groups()[0]

df = grp.as_dataframe()


df = grp.as_dataframe()
pltsize = 20
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
cnt = 0
for clm in df.columns:
    if clm not in ['Timestamp', "Gun", "Tubeside win", "Loadside win", "Collector" ]:
        cnt+=1
        plt.subplot(5,6,cnt)
        plt.plot( df[clm], label = clm)
        plt.legend()
plt.show()


pltsize = 10
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
for clm in df.columns:
    if "Temp" in clm:
        plt.plot( df[clm], label = clm)
        plt.legend()
plt.show()

pltsize = 10
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
for clm in df.columns:
    if "max" in clm or "avg" in clm:
        plt.plot( df[clm], label = clm)
        plt.legend()
plt.show()

pltsize = 10
plt.figure(figsize=(pltsize*2**(1/2), pltsize), dpi=100)
for clm in df.columns:
    if "DC" in clm or "Pulse Width" in clm:
        plt.plot( df[clm], label = clm)
        plt.legend()
plt.show()