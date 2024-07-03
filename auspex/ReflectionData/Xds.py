import numpy as np
import copy
import itertools
import math

from iotbx.xds import read_ascii
import scitbx_array_family_flex_ext as flex

import auspex.BinnedData
from .ReflectionBase import *


class XdsParser(ReflectionParser):
    """The Parser class to process xds files.

    """

    def __init__(self):
        super(XdsParser, self).__init__()
        self.hkl_by_multiplicity = None
        self.intensity_by_multiplicity = None
        self.ires_by_multiplicity = None
        self.sig_by_multiplicity = None

    def read_hkl(self, filename: str = None, merge_equivalents: bool = True):
        """Read the given XDS HKL file.

        :param filename: File or path to file.
        :type filename: str
        :param merge_equivalents: Whether to merge the observations. Default is True.
        :type merge_equivalents: bool
        :return: None
        """
        # use iotbx.mtz to read mtz file
        with open(filename) as ascii_hkl:
            self._obj = read_ascii.reader(ascii_hkl)
        self._space_group = self._obj.crystal_symmetry().space_group()
        # read IOBS
        self._I = np.array(self._obj.iobs, dtype=float)
        # read SIGMA(IOBS)
        self._sigI = np.array(self._obj.sigma_iobs, dtype=float)
        # read hkl
        self._hkl = np.array(self._obj.miller_indices)
        self._resolution = np.array(self._obj.unit_cell.d(self._obj.miller_indices))
        if merge_equivalents is True:
            self._merge()
        self._filename = filename

    def _merge(self):
        """Record the merged data.

        :return: None
        """
        merged_miller = self._obj.as_miller_array(merge_equivalents=True)
        self._I_merged = np.array(merged_miller.data())
        self._hkl_merged = np.array(merged_miller.indices())
        self._sigI_merged = np.array(merged_miller.sigmas())
        self._resolution_merged = np.array(self._obj.unit_cell.d(merged_miller.indices()))
        self._multiplicity_merged = merged_miller.multiplicities().data().as_numpy_array()
        self._complete_set = merged_miller.complete_set()

    def unique_redundancies(self) -> np.ndarray[Literal["N"], int]:
        """Get redundancy of each reflection in merged data.

        :return: array of redundancy
        :rtype: 1d ndarray
        """
        redund = miller.merge_equivalents(
            self._obj.as_miller_array(merge_equivalents=False).map_to_asu()).redundancies().data().as_numpy_array()
        return np.unique(redund)

    def group_by_redundancies(self):
        """Get the lists of indices/observations/resolutions grouped by the number of redundancy.

        :returns: tuple(indices_container, obs_container, resolution_container)
            WHERE
            list indices_container: lists of indices
            list obs_container: lists of observations
            list resolution_container: lists of resolutions
        """
        if not hasattr(self, '_hkl_merged'):
            self._merge()

        # get redundancy of each reflection in merged data
        redund = miller.merge_equivalents(
            self._obj.as_miller_array(merge_equivalents=False).map_to_asu()).redundancies().data().as_numpy_array()


        # get multiplicity of each reflection in merged data
        multi = self._multiplicity_merged

        # get unique redundancies
        uni_redund = np.unique(redund)

        # creat containers for indices, obs and resolution grouped by the number of redundancies
        indices_container = [list() for _ in range(uni_redund.size)]
        obs_container = [list() for _ in range(uni_redund.size)]
        resolution_container = [list() for _ in range(uni_redund.size)]
        sigma_container = [list() for _ in range(uni_redund.size)]

        # shrinkable shallow copy for unmerged indices, obs and resolution
        tmp = self._hkl
        tmp_obs = self._I
        tmp_resol = self._resolution
        # tmp_i_over_sig = self._I / self._sigI
        tmp_sig = self._sigI

        for idx, redund_num in enumerate(uni_redund):  # loop through unique redundancy
            args_redund = np.where(redund == redund_num)[0]
            multi_of_args_redund = multi[args_redund]
            # separate the args_redund by the multiplicities of corresponding reflections
            args_redund_separated = [args_redund[multi_of_args_redund == uni] for uni in
                                     np.unique(multi_of_args_redund)]
            for args in args_redund_separated:  # loop through args_redund separated by multiplicity
                # create iterator for merged data with specific multiplicity and redundancy
                it = self._hkl_merged[args]
                hkl_view = tmp.view([('a', int), ('b', int), ('c', int)])  # 1d view of Nx3 matrix

                set_by_multiplicity = list()
                for hkl_index in it:
                    sym_operator = miller.sym_equiv_indices(self._space_group, hkl_index.tolist())
                    set_by_multiplicity.append([_.h() for _ in sym_operator.indices()])
                # set_by_multiplicity: NxMx3 array,
                # N: number of obs with specific multiplicity and redundancy
                # M: multiplicity
                set_by_multiplicity = np.array(set_by_multiplicity, int)
                multiplicity = set_by_multiplicity.shape[1]

                logic_or = np.zeros(tmp.shape[0], dtype=bool)
                for i in range(multiplicity):
                    # compare_set: Nx3 array
                    compare_set = copy.deepcopy(set_by_multiplicity[:, i, :]).view([('a', int), ('b', int), ('c', int)])
                    # logic_or:  Nx1 bool array, true if obs exists,
                    # N: the number of # of unmerged reflections (shrinkable after each loop)
                    logic_or = logic_or | np.in1d(hkl_view, compare_set)
                # fill out the container
                indices_container[idx].append(tmp[logic_or])
                obs_container[idx].append(tmp_obs[logic_or])
                resolution_container[idx].append(tmp_resol[logic_or])
                sigma_container[idx].append(tmp_sig[logic_or])
                # shrink reflections
                tmp = copy.deepcopy(tmp[~logic_or])
                tmp_obs = tmp_obs[~logic_or]
                tmp_resol = tmp_resol[~logic_or]
                tmp_sig = tmp_sig[~logic_or]
            indices_container[idx] = np.concatenate(indices_container[idx]).reshape(args_redund.size, redund_num, 3)
            obs_container[idx] = np.concatenate(obs_container[idx]).reshape(args_redund.size, redund_num)
            resolution_container[idx] = np.concatenate(resolution_container[idx]).reshape(args_redund.size, redund_num)[:, 0]
            sigma_container[idx] = np.concatenate(sigma_container[idx]).reshape(args_redund.size, redund_num)
        # apply sigma filter
        for idx, redund_num in enumerate(uni_redund):
            if redund_num == 1:
                valid_args = sigma_container[idx] > 0.
                sigma_container[idx] = sigma_container[idx][valid_args]
                indices_container[idx] = indices_container[idx][valid_args]
                obs_container[idx] = obs_container[idx][valid_args]
                resolution_container[idx] = resolution_container[idx][valid_args[:, 0]]
            else:
                invalid_ind = np.argwhere(sigma_container[idx] < 0.)[:, 0]
                invalid_idx, invalid_idx_counts = np.unique(invalid_ind, return_counts=True)
                #
                ind_to_remove = invalid_idx[invalid_idx_counts == redund_num]
                sigma_container[idx] = np.delete(sigma_container[idx], ind_to_remove, 0)
                indices_container[idx] = np.delete(indices_container[idx], ind_to_remove, 0)
                obs_container[idx] = np.delete(obs_container[idx], ind_to_remove, 0)
                resolution_container[idx] = np.delete(resolution_container[idx], ind_to_remove, 0)

                for invalid_redund_num in range(1, redund_num):
                    invalid_ind = np.argwhere(sigma_container[idx] < 0.)
                    invalid_ind, invalid_ind_counts = np.unique(invalid_ind[:, 0], return_counts=True)
                    valid_redund_num = redund_num - invalid_redund_num

                    ind_to_downgrade = invalid_ind[invalid_ind_counts == invalid_redund_num]
                    if ind_to_downgrade is None or ind_to_downgrade.size == 0:
                        continue

                    sig_array_to_downgrade = sigma_container[idx][ind_to_downgrade]
                    sig_array_to_downgrade_args = sig_array_to_downgrade > 0.


                    sig_array_downgraded = \
                        sig_array_to_downgrade[sig_array_to_downgrade_args].reshape(sig_array_to_downgrade.shape[0], valid_redund_num)
                    obs_array_downgraded = \
                        obs_container[idx][ind_to_downgrade][sig_array_to_downgrade_args].reshape(sig_array_to_downgrade.shape[0], valid_redund_num)
                    indices_array_downgraded = \
                        indices_container[idx][ind_to_downgrade][sig_array_to_downgrade_args].reshape(sig_array_to_downgrade.shape[0], valid_redund_num, 3)
                    resolution_array_downgraded = \
                        resolution_container[idx][ind_to_downgrade]


                    idx_multi_valid = np.argwhere(uni_redund == valid_redund_num).flatten()[0]

                    sigma_container[idx_multi_valid] = np.append(sigma_container[idx_multi_valid], sig_array_downgraded, 0)
                    indices_container[idx_multi_valid] = np.append(indices_container[idx_multi_valid], indices_array_downgraded, 0)
                    obs_container[idx_multi_valid] = np.append(obs_container[idx_multi_valid], obs_array_downgraded, 0)
                    resolution_container[idx_multi_valid] = np.append(resolution_container[idx_multi_valid], resolution_array_downgraded, 0)

                    sigma_container[idx] = np.delete(sigma_container[idx], ind_to_downgrade, 0)
                    indices_container[idx] = np.delete(indices_container[idx], ind_to_downgrade, 0)
                    obs_container[idx] = np.delete(obs_container[idx], ind_to_downgrade, 0)
                    resolution_container[idx] = np.delete(resolution_container[idx], ind_to_downgrade, 0)

        self.hkl_by_multiplicity = indices_container
        self.intensity_by_multiplicity = obs_container
        self.ires_by_multiplicity = resolution_container
        self.sig_by_multiplicity = sigma_container
        #return indices_container, obs_container, resolution_container

    def merge_stats_cmpt(self) \
            -> tuple[np.ndarray[Literal["N"], np.float32], np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32], np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32], np.ndarray[Literal["N"], np.float32]]:
        _obs = self.intensity_by_multiplicity
        _resolution = self.ires_by_multiplicity
        uni_redund = self.unique_redundancies()
        redundant_counts = [len(_) for _ in _resolution[1:]]
        total_redundant_counts = np.sum(redundant_counts)

        # ires_redundant = np.concatenate(_resolution[1:])

        redundant_counts_idx_lower = np.insert(np.cumsum(redundant_counts)[:-1], 0, 0)
        redundant_counts_idx_upper = np.cumsum(redundant_counts)

        # preallocate arrays. better efficiency
        r_pim_components = np.zeros(total_redundant_counts, dtype=float)
        r_meas_components = np.zeros(total_redundant_counts, dtype=float)
        r_merge_components = np.zeros(total_redundant_counts, dtype=float)
        r_denominator = np.zeros(total_redundant_counts, dtype=float)
        cc_sig_epsilon_squared = np.zeros(total_redundant_counts, dtype=float)
        cc_x_i_bar = np.zeros(total_redundant_counts, dtype=float)

        for ind, redund in enumerate(uni_redund):
            if redund == 1:
                continue
            numerate = np.abs(_obs[ind] - np.mean(_obs[ind], axis=1)[:, None]).sum(1)

            r_denominator[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] \
                = _obs[ind].sum(1)

            r_pim_components[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] \
                = (np.sqrt(1. / (redund - 1.)) * numerate)

            r_meas_components[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] = \
                (np.sqrt(redund / (redund - 1.)) * numerate)

            r_merge_components[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] \
                = numerate

            cc_sig_epsilon_squared[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] \
                = 1. / (redund - 1.) * (np.square(_obs[ind]).sum(axis=1)
                                      - np.square(_obs[ind].sum(1)) / redund) * 2. / redund

            cc_x_i_bar[redundant_counts_idx_lower[ind-1]:redundant_counts_idx_upper[ind-1]] \
                = np.mean(_obs[ind], axis=1)

        return r_pim_components, r_meas_components, r_merge_components, r_denominator, \
            cc_sig_epsilon_squared, cc_x_i_bar

    def merge_stats_overall(self) -> auspex.BinnedData.BinnedStatistics:
        r_pim_cmpt, r_meas_cmpt, r_merge_cmpt, r_denominator, cc_sig_epsilon_cmpt, cc_x_i_bar_cmpt = self.merge_stats_cmpt()
        ires_unique = np.concatenate(self.ires_by_multiplicity)
        num_data = ires_unique.size
        i_mean_hkl = np.concatenate(
            (self.intensity_by_multiplicity[0], np.concatenate([np.mean(_, axis=1) for _ in self.intensity_by_multiplicity[1:]])))
        ires_minmax = [ires_unique.min(), ires_unique.max()]
        r_pim = r_pim_cmpt.sum() / r_denominator.sum()
        r_merge = r_merge_cmpt.sum() / r_denominator.sum()
        r_meas = r_meas_cmpt.sum() / r_denominator.sum()
        completeness_mean = self.cal_completeness(ires_unique)
        sig_rms_hkl = np.concatenate([self.sig_by_multiplicity[0],
                                     np.concatenate([np.sqrt(np.mean(_ * _, axis=1)) for _ in self.sig_by_multiplicity[1:]])])
        i_mean = i_mean_hkl.mean()
        i_over_sigma_mean = i_mean / np.sqrt(np.mean(sig_rms_hkl * sig_rms_hkl))
        redundancy_mean = np.concatenate([np.full(_.shape[0], _[0].size) for _ in self.intensity_by_multiplicity]).mean()

        # cc half
        sig_epsilon_square = cc_sig_epsilon_cmpt.mean()
        sig_y_square = \
            1 / (cc_x_i_bar_cmpt.size - 1) * ((cc_x_i_bar_cmpt * cc_x_i_bar_cmpt).sum() - np.square(cc_x_i_bar_cmpt.sum()) / cc_x_i_bar_cmpt.size)
        cc_half = (sig_y_square - 0.5 * sig_epsilon_square) / (sig_y_square + 0.5 * sig_epsilon_square)
        from ..BinnedData import BinnedStatistics
        merge_stats = BinnedStatistics().const_stats(ires_minmax, num_data, i_mean, i_over_sigma_mean, completeness_mean, redundancy_mean,
                                                     r_pim, r_merge, r_meas, cc_half)
        return merge_stats

    def merge_stats_binned(self, num_of_bins: int = 21) -> auspex.BinnedData.BinnedStatistics:
        r_pim_cmpt, r_meas_cmpt, r_merge_cmpt, r_denominator, cc_sig_epsilon_cmpt, cc_x_i_bar_cmpt = self.merge_stats_cmpt()
        ires_unique = np.concatenate(self.ires_by_multiplicity)
        intensity_hkl = np.concatenate(
            (self.intensity_by_multiplicity[0], np.concatenate([np.mean(_, axis=1) for _ in self.intensity_by_multiplicity[1:]]))
        )

        """
        sigma_hkl = np.concatenate(
            (self.sig_by_multiplicity[0], np.concatenate([np.mean(_ * _, axis=1) for _ in self.sig_by_multiplicity[1:]]))
        )
        """

        sig_nested = sum([_.tolist() for _ in self.sig_by_multiplicity], []) # concatenate nested arrays of different dimensions


        redundancy = np.concatenate([np.full(_.shape[0], _[0].size) for _ in self.intensity_by_multiplicity])
        args_binned = _get_args_binned(self.ires_by_multiplicity, num_of_bins)
        r_pim_binned = []
        r_merge_binned = []
        r_meas_binned = []
        cc_half_binned = []
        # cc_star_binned = dict()
        ires_binned = []
        i_mean_binned = []
        i_over_sigma_binned = []
        redundancy_binned = []
        completeness_binned = []
        num_data_binned = []

        for args in args_binned:
            args_redund = args[self.ires_by_multiplicity[0].size:]
            ires_binned.append(ires_unique[args])

            num_data_binned.append(np.sum(args))

            # completeness
            completeness_binned.append(self.cal_completeness(ires_binned[-1]))

            # mean intensity
            i_mean = intensity_hkl[args].mean()
            i_mean_binned.append(i_mean)

            # mean i over sigma
            """
            i_over_sigma_binned.append(np.mean(intensity_hkl[args]/sigma_hkl[args]))
            """
            sigs_in_bin = np.hstack(list(itertools.compress(sig_nested, args))) # itertools.compress(list, boolean_list), acting similar to np.take
            sig_rms = np.sqrt(np.mean(sigs_in_bin * sigs_in_bin))
            i_over_sigma_binned.append(i_mean/sig_rms)

            redundancy_binned.append(redundancy[args].mean())

            # stats
            r_pim_binned.append(r_pim_cmpt[args_redund].sum() / r_denominator[args_redund].sum())
            r_merge_binned.append(r_merge_cmpt[args_redund].sum() / r_denominator[args_redund].sum())
            r_meas_binned.append(r_meas_cmpt[args_redund].sum() / r_denominator[args_redund].sum())
            sig_epsilon_square = cc_sig_epsilon_cmpt[args_redund].mean()
            x_i_bar = cc_x_i_bar_cmpt[args_redund]
            sig_y_square = \
                1 / (x_i_bar.size - 1) * ((x_i_bar * x_i_bar).sum() - np.square(x_i_bar.sum()) / x_i_bar.size)
            cc_half_binned.append(
                (sig_y_square - 0.5 * sig_epsilon_square) / (sig_y_square + 0.5 * sig_epsilon_square)
            )
            # cc_star_binned[bin_num] = np.sqrt((2 * cc_half_binned[bin_num]) / (1 + cc_half_binned[bin_num]))
        r_pim_binned = np.array(r_pim_binned)
        r_merge_binned = np.array(r_merge_binned)
        r_meas_binned = np.array(r_meas_binned)
        cc_half_binned = np.array(cc_half_binned)
        i_mean_binned = np.array(i_mean_binned)
        i_over_sigma_binned = np.array(i_over_sigma_binned)
        redundancy_binned = np.array(redundancy_binned)
        completeness_binned = np.array(completeness_binned)
        num_data_binned = np.array(num_data_binned)
        from ..BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_binned, num_data_binned, i_mean_binned, i_over_sigma_binned, completeness_binned,
                                                    redundancy_binned, r_pim_binned, r_merge_binned, r_meas_binned, cc_half_binned)
        return merg_stats

    def merge_stats_by_range(self, max_resolution: float, min_resolution: float) \
            -> auspex.BinnedData.BinnedStatistics:
        r_pim_cmpt, r_meas_cmpt, r_merge_cmpt, r_denominator, cc_sig_epsilon_cmpt, cc_x_i_bar_cmpt = self.merge_stats_cmpt()
        ires_unique = np.concatenate(self.ires_by_multiplicity)
        intensity_hkl = np.concatenate(
            (self.intensity_by_multiplicity[0], np.concatenate([np.mean(_, axis=1) for _ in self.intensity_by_multiplicity[1:]]))
        )

        sig_nested = sum([_.tolist() for _ in self.sig_by_multiplicity], []) # concatenate nested arrays of different dimensions

        redundancy = np.concatenate([np.full(_.shape[0], _[0].size) for _ in self.intensity_by_multiplicity])
        args_by_range = _get_args_by_range(self.ires_by_multiplicity, max_resolution, min_resolution)

        args_redund = args_by_range[self.ires_by_multiplicity[0].size:]
        ires_mean = ires_unique[args_by_range].mean()
        num_data = np.sum(args_by_range)
        completeness = self.cal_completeness(ires_unique[args_by_range])
        i_mean = intensity_hkl[args_by_range]

        sigs_in_bin = np.hstack(list(
            itertools.compress(sig_nested, args_by_range)))  # itertools.compress(list, boolean_list), acting similar to np.take
        sig_rms = np.sqrt(np.mean(sigs_in_bin * sigs_in_bin))
        i_over_sigma = i_mean / sig_rms
        redundancy = np.concatenate([np.full(_.shape[0], _[0].size) for _ in self.intensity_by_multiplicity])[args_by_range].mean()

        r_pim = r_pim_cmpt[args_redund].sum() / r_denominator[args_redund].sum()
        r_merge = r_merge_cmpt[args_redund].sum() / r_denominator[args_redund].sum()
        r_meas = r_meas_cmpt[args_redund].sum() / r_denominator[args_redund].sum()
        sig_epsilon_square = cc_sig_epsilon_cmpt[args_redund].mean()
        x_i_bar = cc_x_i_bar_cmpt[args_redund]
        sig_y_square = 1 / (x_i_bar.size - 1) * ((x_i_bar * x_i_bar).sum() - np.square(x_i_bar.sum()) / x_i_bar.size)
        cc_half = (sig_y_square - 0.5 * sig_epsilon_square) / (sig_y_square + 0.5 * sig_epsilon_square)
        # cc_star_binned[bin_num] = np.sqrt((2 * cc_half_binned[bin_num]) / (1 + cc_half_binned[bin_num]))

        from ..BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_mean, num_data, i_mean, i_over_sigma, completeness,
                                                    redundancy, r_pim, r_merge, r_meas, cc_half)
        return merg_stats

    def cc_sig_y_square(self, num_of_bins: int = 21) \
            -> tuple[np.ndarray[Literal["N"], np.float32], np.ndarray[Literal["N"], np.float32], np.ndarray[Literal["N"], np.float32]]:
        r_pim_cmpt, r_meas_cmpt, r_merge_cmpt, r_denominator, cc_sig_epsilon_cmpt, cc_x_i_bar_cmpt = self.merge_stats_cmpt()
        ires_unique = np.concatenate(self.ires_by_multiplicity)

        redundancy = np.concatenate([np.full(_.shape[0], _[0].size) for _ in self.intensity_by_multiplicity])
        args_binned = _get_args_binned(self.ires_by_multiplicity, num_of_bins)
        cc_sig_y_square_binned = list()
        cc_sig_epsilon_square_binned = list()
        ires_binned = list()
        redundancy_binned = list()
        completeness_binned = list()
        num_data_binned = list()

        for args in args_binned:
            args_redund = args[self.ires_by_multiplicity[0].size:]
            ires_binned.append(ires_unique[args])

            num_data_binned.append(np.sum(args))

            # completeness
            completeness_binned.append(self.cal_completeness(ires_binned[-1]))
            redundancy_binned.append(redundancy[args].mean())

            # stats
            sig_epsilon_square = cc_sig_epsilon_cmpt[args_redund].mean()
            x_i_bar = cc_x_i_bar_cmpt[args_redund]
            sig_y_square = \
                1 / (x_i_bar.size - 1) * ((x_i_bar * x_i_bar).sum() - np.square(x_i_bar.sum()) / x_i_bar.size)
            cc_sig_y_square_binned.append(sig_y_square)
            cc_sig_epsilon_square_binned.append(sig_epsilon_square)
            # cc_star_binned[bin_num] = np.sqrt((2 * cc_half_binned[bin_num]) / (1 + cc_half_binned[bin_num]))
        cc_sig_y_square_binned = np.array(cc_sig_y_square_binned)
        cc_sig_epsilon_square_binned = np.array(cc_sig_epsilon_square_binned)
        redundancy_binned = np.array(redundancy_binned)
        completeness_binned = np.array(completeness_binned)
        num_data_binned = np.array(num_data_binned)
        return ires_binned, cc_sig_y_square_binned, cc_sig_epsilon_square_binned

    def merge_stats_binned_deprecated(self, iresbinwidth: float = 0.01) \
            -> tuple[np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32],
                     np.ndarray[Literal["N"], np.float32]]:
        ires, r_pim_cmpt, r_meas_cmpt, r_merge_cmpt, r_denominator, cc_sig_epsilon_cmpt, cc_x_i_bar_cmpt = self.merge_stats_cmpt()
        binning = _get_bins_by_binwidth(ires, iresbinwidth)
        binned_args = _get_args_binned()
        r_pim_binned = dict()
        r_merge_binned = dict()
        r_meas_binned = dict()
        cc_half_binned = dict()
        # cc_star_binned = dict()
        ires_binned = dict()
        for bin_num, binned_idx in binning.items():
            ires_binned[bin_num] = ires[binned_idx]
            if binned_idx.size <= 1:
                r_pim_binned[bin_num] = 0.
                r_merge_binned[bin_num] = 0.
                r_meas_binned[bin_num] = 0.
                cc_half_binned[bin_num] = 0.
            else:
                r_pim_binned[bin_num] = r_pim_cmpt[binned_idx].sum() / r_denominator[binned_idx].sum()
                r_merge_binned[bin_num] = r_merge_cmpt[binned_idx].sum() / r_denominator[binned_idx].sum()
                r_meas_binned[bin_num] = r_meas_cmpt[binned_idx].sum() / r_denominator[binned_idx].sum()
                sig_epsilon_square = cc_sig_epsilon_cmpt[binned_idx].mean()
                x_i_bar = cc_x_i_bar_cmpt[binned_idx]
                sig_y_square = \
                    1 / (x_i_bar.size - 1) * ((x_i_bar * x_i_bar).sum() - np.square(x_i_bar.sum()) / x_i_bar.size)
                cc_half_binned[bin_num] = \
                    (sig_y_square - 0.5 * sig_epsilon_square) / (sig_y_square + 0.5 * sig_epsilon_square)
                # cc_star_binned[bin_num] = np.sqrt((2 * cc_half_binned[bin_num]) / (1 + cc_half_binned[bin_num]))
        return ires_binned, r_pim_binned, r_merge_binned, r_meas_binned, cc_half_binned

    def cal_completeness(self, unique_ires_array: np.ndarray[Literal["N"], np.float32],
                         d_min: float = None,
                         d_max: float = None) -> float:
        """Calculate the completeness between d_min and d_max.

        :param unique_ires_array: The d-spacings (resolutions) of the given unique observations.
        :param d_min: Minimum d-spacing.
        :param d_max: Maximum d-spacing.
        :return: Completeness between d_min and d_max.
        """
        d_star_sq = self._complete_set.d_star_sq().data()
        d_star = flex.sqrt(d_star_sq)
        if d_min is None:
            d_min = unique_ires_array.min()
        if d_max is None:
            d_max = unique_ires_array.max()
        sele_theory = (d_star >= 1/d_max) & (d_star <= 1/d_min)
        theory_obs_num = self._complete_set.select(sele_theory).size()
        sele_unique = (unique_ires_array >= d_min) & (unique_ires_array <= d_max)
        unique_obs_num = sele_unique.sum()
        completeness = unique_obs_num / theory_obs_num
        return completeness

    @filename_check
    def get_space_group(self) -> str:
        """
        :return: space group
        :rtype: str
        """
        return str(self._obj.miller_set().space_group().info())

    @filename_check
    def get_cell_dimension(self):
        """
        :return: cell dimensions (a*, b*, c*, alpha, beta, gamma)
        :rtype: tuple
        """
        return self._obj.miller_set().unit_cell().parameters()

    @filename_check
    def get_max_resolution(self) -> float:
        """
        :return: maximum resolution
        :rtype: float
        """
        return self._obj.miller_set().resolution_range()[1]

    @filename_check
    def get_min_resolution(self) -> float:
        """
        :return: minimum resolution
        :rtype: float
        """
        return self._obj.miller_set().resolution_range()[0]

    @filename_check
    def get_merged_I(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._I_merged

    @filename_check
    def get_merged_hkl(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._hkl_merged

    @filename_check
    def get_merged_sig(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._sigI_merged

    @filename_check
    def get_merged_resolution(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: resolution array of merged intensity
        :rtype: 1d ndarray
        """
        return self._resolution_merged

    @filename_check
    def get_zd(self) -> np.ndarray[Literal["N"], np.float32] | None:
        """
        :return: reflection position array on z-axis
        :rtype: 1d ndarray
        """
        if self._obj.unmerged_data:
            return self._obj.zd.as_numpy_array()
        else:
            return None


def _get_bins_by_binwidth(ires: np.ndarray[Literal["N"], np.float32], bin_width: float) \
        -> Dict[str, np.ndarray[Literal["N"], np.int_]]:
    all_bins = np.floor(1. / ires / bin_width)
    # find the unique bin values (uni_vals), the indices of unique values (inv_vals)
    # and the number of times each unique item appears (counts)
    uni_vals, inv_vals, counts = np.unique(all_bins, return_inverse=True, return_counts=True)
    bins_num = uni_vals.astype(int)
    # idx_vals_repeated = uni_vals[np.where(counts > 0.)[0]]
    idx_vals_repeated = np.where(counts > 0.)[0]
    tmp_r, tmp_c = np.where(inv_vals == idx_vals_repeated[:, None])
    _, inv_tmp_r = np.unique(tmp_r, return_index=True)
    binned_idx = np.array(np.split(tmp_c, inv_tmp_r[1:]), dtype=object)
    binning = dict(zip(bins_num, binned_idx))
    return binning


def _binning_idx_even(array_size: int, num_of_bins: int) -> np.ndarray[Literal["N", 2], np.int_]:
    max_per_bin = array_size // num_of_bins
    min_per_bin = array_size // (num_of_bins+1)
    rng = np.random.default_rng(12345)
    step_redund_per_bin = np.floor(min_per_bin + (max_per_bin - min_per_bin) * rng.random(num_of_bins))
    compensation_per_bin = (array_size - step_redund_per_bin.sum()) // num_of_bins
    step_redund_per_bin += compensation_per_bin
    upper_ind_list = np.cumsum(step_redund_per_bin)
    upper_ind_list[-1] = array_size - 1
    lower_ind_list = np.insert(upper_ind_list[:-1], 0, 0)
    ind_array = np.array((lower_ind_list, upper_ind_list), dtype=int).transpose()
    return ind_array


def _binning_idx_xprep(array_size: int, num_of_bins: int = 21):
    max_per_bin = array_size // (num_of_bins-1)
    min_per_bin = array_size // num_of_bins
    rng = np.random.default_rng(12345)
    step_redund_per_bin = np.floor(min_per_bin + (max_per_bin - min_per_bin) * rng.random(num_of_bins - 1))
    compensation_per_bin = (array_size - step_redund_per_bin.sum()) // (num_of_bins - 1)
    step_redund_per_bin += compensation_per_bin
    upper_ind_list = np.cumsum(step_redund_per_bin)
    division_20 = upper_ind_list[-2] + math.floor((array_size - upper_ind_list[-2]) * 0.7)
    upper_ind_list[-1] = division_20
    upper_ind_list = np.append(upper_ind_list, array_size - 1)
    lower_ind_list = np.insert(upper_ind_list[:-1], 0, 0)
    ind_array = np.array((lower_ind_list, upper_ind_list), dtype=int).transpose()
    return ind_array


def _get_args_binned(ires_by_multiplicity: list, num_of_bins: int, method: str = 'xprep') \
        -> list[np.ndarray[Literal["N"], np.float32]]:
    """
    :param ires_by_multiplicity: a list of resolution groupped by multiplicity
    :type ires_by_multiplicity: list
    :param num_of_bins: total number of bins
    :type num_of_bins: int
    :param method: 'even' or 'xprep'. Default: xprep
    :type method: str
    :return: A list of the indices grouped by corresponding binning method.
    """
    #  sorted_args = [np.argsort(_) for _ in ires_by_multiplicity]
    #  list_ires_size = [_.size for _ in sorted_args]
    if method == 'xprep':
        binning_idx = _binning_idx_xprep
    elif method == 'even':
        binning_idx = _binning_idx_even
    else:
        binning_idx = _binning_idx_xprep
    ires_unique = np.concatenate(ires_by_multiplicity)
    sorted_args_ires_all = np.argsort(ires_unique)
    ind_range_of_binned = binning_idx(ires_unique.size, num_of_bins)
    minmax_ires_per_bin = ires_unique[sorted_args_ires_all][ind_range_of_binned]
    binned_args_by_binnum= []
    for min_ires, max_ires in minmax_ires_per_bin:
        binned_args_by_binnum.append((ires_unique <= max_ires) & (ires_unique >= min_ires))
    return binned_args_by_binnum


def _get_args_by_range(ires_by_multiplicity: list, max_resolution: float, min_resolution: float) \
        -> np.ndarray[Literal["N"], np.float32]:
    ires_unique = np.concatenate(ires_by_multiplicity)
    args_in_range = (ires_unique >= max_resolution) & (ires_unique <= min_resolution)
    return args_in_range

