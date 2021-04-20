import numpy as np
import pandas as pd
import nptdms
import os
import matplotlib.pyplot as plt
from src.tdms_reader.tdms_to_pd import tdmsfile_to_pd

class tdms_analysis:
    tdms:nptdms.TdmsFile = None
    def __init__(self, tdmsfile:nptdms.TdmsFile):
        self.tdms = tdmsfile
        print("### Analyzing .tdms file " + self.tdms.properties["name"])

    def main(self) -> pd.DataFrame:
        self.check_channels()
        self.check_channel_properties()

        print("\n#Channel structure")
        print(self.struct_grp_ch())
        print("\n#Channel data size structure")
        print(self.struct_grp_chdatasize())

        self.plt_arbitrary_group()
        return tdmsfile_to_pd(self.tdms)

    def struct_grp_ch(self):
        # produces a df of all channel names of all groups
        max_ch_number = max([len(grp.channels()) for grp in self.tdms.groups()])
        return pd.DataFrame(data = [[ch.name for ch in grp.channels()] for grp in self.tdms.groups()],
                                    index = [grp.name for grp in self.tdms.groups()],
                                    columns = ["ch" + str(k) for k in range(max_ch_number)])

    def struct_grp_chdatasize(self):
        # produces a df of all channel-data-shapes of all channels in all groups
        clms=set()
        for grp in self.tdms.groups(): clms.update({ch.name for ch in grp.channels()})
        return pd.DataFrame(data = [[ch.data.shape[0] for ch in grp.channels()] for grp in self.tdms.groups()],
                                    index = [grp.name for grp in self.tdms.groups()],
                                    columns = ["ch" + str(k) for k in clms])

    def check_channels(self) -> None:
        # tests if all channels in the tdms file are named equally
        chn_union = set()
        for grp in self.tdms.groups(): chn_union.update({ch.name for ch in grp.channels()})
        chn_intersect = chn_union
        for grp in self.tdms.groups(): chn_intersect.intersection_update({ch.name for ch in grp.channels()})

        if chn_intersect == chn_union:
            print("Channel check passed for " + self.tdms.properties["name"])
        else:
            print("The following channels do not ocure in all groups in " + self.tdms.properties["name"] + ": " +
                str(chn_union.symmetric_difference(chn_intersect)))

    def check_channel_properties(self):
        # tests if all channels with the same name have the same channel property names
        chpn_union = set()
        for grp in self.tdms.groups():
            for ch in grp.channels():
                chpn_union.update(set(ch.properties.keys()))
        chpn_intersect = chpn_union

        for grp in self.tdms.groups():
            for ch in grp.channels():
                chpn_intersect.intersection_update(set(ch.properties.keys()))

        if chpn_intersect == chpn_union:
            print("Channel property names check passed for " + self.tdms.properties["name"])
        else:
            print("The following channels do not ocure in all groups in" + self.tdms.properties["name"] + ": " +
                str(chpn_union.symmetric_difference(chpn_intersect)))
        return chpn_union

    def plt_arbitrary_group(self):
        num_grps = len(self.tdms.groups())
        grp = self.tdms.groups()[np.random.randint(num_grps)]
        fig, ax = plt.subplots()
        for ch in grp.channels():
            ax.plot(ch.data)
        ax.set(xlabel='time', ylabel='value', title=grp.name)
            # ax.grid()
        plt.show()

path = os.path.expanduser("~/project_data/CLIC_DATA_Xbox2_T24PSI_2/EventData_20180616.tdms")
tdmsfile = nptdms.TdmsFile(path)
tdms_analysis(tdmsfile).main()

