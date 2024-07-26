import warnings

import numpy as np
from tabulate import tabulate, SEPARATING_LINE

from auspex import __version__


class MergeStatistics(object):
    def __init__(self, merge_stats_binned, merge_stas_overall):
        resolution, num_data, i_mean, i_over_sigma, completeness, redundancy, r_pim, r_merge, r_meas, cc_half = merge_stats_binned.get_stats_as_list()
        self.i_mean = np.char.mod('%.2f', i_mean)
        self.i_over_sigma = np.char.mod('%.2f', i_over_sigma)
        self.redundancy = np.char.mod('%.2f', redundancy)
        self.completeness = np.char.mod('%.1f', completeness*100)
        self.r_pim = np.char.mod('%.4f', r_pim)
        self.r_merge = np.char.mod('%.4f', r_merge)
        self.r_meas = np.char.mod('%.4f', r_meas)
        self.cc_half = np.char.mod('%.4f', cc_half)
        self.res_range = ['{:=5.2f} -{:=5.2f}'.format(_.max(), _.min()) for _ in resolution]
        # self.res_range[-1][0:5] = '  Inf'
        self.res_mean = np.char.mod('%.2f', [_.mean() for _ in resolution])
        self.num_data = np.char.mod('%d', num_data)
        self.merge_stats_overall = merge_stas_overall
        self.header = ["Resolution", "#Data", "%Complete", "Redundancy", "<I>", "<I/s>", "cc1/2", "Rmerge", "Rpim", "Rmeas"]
        #self.stats_dict = self.format_dict()

    def format_dict_by_column(self):
        stats_dict = dict()
        stats_dict["Resolution"] = self.res_range
        stats_dict["#Data"] = self.num_data
        stats_dict["%Complete"] = self.completeness
        stats_dict["Redundancy"] = self.redundancy
        stats_dict["<I>"] = self.i_mean
        stats_dict["<I/s>"] = self.i_over_sigma
        stats_dict["cc1/2"] = self.cc_half
        stats_dict["Rmerge"] = self.r_merge
        stats_dict["Rpim"] =self.r_pim
        stats_dict["Rmeas"] = self.r_meas
        return stats_dict

    def format_dict_by_row(self):
        stats_list = []
        for i in range(len(self.res_range)):
            stats_list.append([self.res_range[i], self.num_data[i], self.completeness[i], self.redundancy[i],
                               self.i_mean[i], self.i_over_sigma[i],
                               self.cc_half[i], self.r_merge[i], self.r_pim[i], self.r_meas[i]])
        return stats_list

    def format_stats_overall(self):
        stats_list = ['{:=5.2f} -{:=5.2f}'.format(self.merge_stats_overall.ires_binned[1],
                                                  self.merge_stats_overall.ires_binned[0]),
                      '{:d}'.format(self.merge_stats_overall.num_data_binned),
                      '{:.1f}'.format(self.merge_stats_overall.completeness_binned*100),
                      '{:.2f}'.format(self.merge_stats_overall.redundancy_binned),
                      '{:.2f}'.format(self.merge_stats_overall.i_mean_binned),
                      '{:.2f}'.format(self.merge_stats_overall.i_over_sigma_binned),
                      '{:.4f}'.format(self.merge_stats_overall.cc_half_binned),
                      '{:.4f}'.format(self.merge_stats_overall.r_merge_binned),
                      '{:.4f}'.format(self.merge_stats_overall.r_pim_binned),
                      '{:.4f}'.format(self.merge_stats_overall.r_meas_binned)]
        return stats_list

    def print_stats_table(self):
        # table = tabulate(self.stats_dict, headers='keys', colalign=('right', 'right', 'center', 'center', 'right', 'right','right', 'right', 'right', 'right'), disable_numparse=True)
        stats_list_binned = self.format_dict_by_row()
        table = tabulate([self.header, *self.format_dict_by_row(), SEPARATING_LINE, self.format_stats_overall()], headers='firstrow',
                         colalign=('right', 'right', 'center', 'center', 'right', 'right','right', 'right', 'right', 'right'), disable_numparse=True)
        row_len = table.find("\n") - 1
        print("#"*row_len)
        print("{:#^{}s}".format("   Intensity Statistics   ", row_len))
        print("#" * row_len)
        print(table)

def auspex_init(version, command_line):
    print("")
    print("            ######################################################## ")
    print("           #                    _   _   _ ____  ____  _______  __   #")
    print("           #       A           / \ | | | / ___||  _ \| ____\ \/ /   #")
    print("           #   /MmmOmmM\      / _ \| | | \___ \| |_) |  _|  \  /    #")
    print("           #       #         / ___ \ |_| |___) |  __/| |___ /  \    #")
    print("           #      /#\       /_/   \_\___/|____/|_|   |_____/_/\_\   #")
    print("           #                                        {:>13s}   #".format("Version " + str(version)))
    print("            ######################################################## ")
    print("\nCOMMAND LINE: auspex {0}".format(command_line))


def suppress_warnings():
    warnings.filterwarnings("ignore", message="The integral is probably divergent, or slowly convergent.")
    warnings.filterwarnings("ignore", message="divide by zero encountered in")
    warnings.filterwarnings("ignore", message="invalid value encountered in cumprob_c_intensity")
    warnings.filterwarnings("ignore", message="The occurrence of roundoff error is detected")
    warnings.filterwarnings("ignore", message="shmem: mmap:")
    warnings.filterwarnings("ignore", message="BTL coordinating structure")

def generate_plot():
    print("_______________________________________________________________________________\n")
    print("                                GENERATING PLOTS:                              ")
    print("        (Depending on the size of the data set, this may take a moment)        \n")