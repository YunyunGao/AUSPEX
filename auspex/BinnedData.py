from ReflectionData import Observation
from IceRings import IceRing
import numpy as np
import copy


class BinnedSummaries(object):
    def __init__(self, obs_obj):
        """
        Takes observation object and creates a Binsummary object.
        Can flag an ice ring using an IceRings object and a icefinder score for each bin.

        :param obs_obj: Observation instance
        """
        super(BinnedSummaries, self).__init__()
        # assert isinstance(obs_obj, Observation), 'invalid import.' mark out due to conflicts between dev and build
        self._observation = obs_obj
        self._iresbinwidth = None  # bin width
        self._bins = None  # indices of bins
        self._binned_idx = None  # indices of observations. Each bin is a 1d ndarray.
        self._no_obs_binned = None  # number of observations in each bin.
        self._bin_args_in_icering = None  # indices of bins which are in the ice ring range
        # Defines window size (no. bins) for the filtering/smoothing, such that: window_size = 2*smoothing_parameter + 1
        self._smooth_param = 5
        # The quantiles used for filtering prior to Gaussian smoothing, thus ensuring robustness to outliers
        self._quantiles = [0.25, 0.75]
        # This means that the window will span 2 sd's from the mean, i.e. within 95%, when doing Gaussian smoothing
        self._smooth_sd_divisor = 2.
        self._lower_quantiles = list()
        self._upper_quantiles = list()
        self._stdmeans = None  # standard mean intensities of bins
        self._est_stdmeans = None  # estimated standard mean intensities of bins
        self.set_binning_rules()

    def set_binning_rules(self, iresbinwidth=None):
        """Set parameters used for binning.

        :param iresbinwidth:  Bin width, provided as inverse resolution. Default value is 0.001.
        :type iresbinwidth: float
        """
        if self._iresbinwidth is None:
            self._iresbinwidth = 0.001
        #elif not isinstance(iresbinwidth, float):
            #raise TypeError("bin width should be a float number, provided as inverse resolution")
        else:
            self._iresbinwidth = iresbinwidth
        assert self._iresbinwidth > 0.0, print("Bin width must not be negative.")
        # calculate the bin values of all reflections
        all_bins = np.floor(1. / self._observation.ires / self._iresbinwidth)
        # find the unique bin values (uni_vals), the indices of unique values (inv_vals)
        # and the number of times each unique item appears (counts)
        uni_vals, inv_vals, counts = np.unique(all_bins, return_inverse=True, return_counts=True)
        self._bins = uni_vals.astype(int)
        # idx_vals_repeated = uni_vals[np.where(counts > 0.)[0]]
        idx_vals_repeated = np.where(counts > 0.)[0]
        tmp_r, tmp_c = np.where(inv_vals == idx_vals_repeated[:, None])
        _, inv_tmp_r = np.unique(tmp_r, return_index=True)
        self._binned_idx = np.array(np.split(tmp_c, inv_tmp_r[1:]), dtype=object)
        self._no_obs_binned = counts

    def obs_in_bin(self, bin_num):
        """Return the value of all the observations at the given bin.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: An array of observations
        :rtype: ndarray of float
        """
        # add [0] due to the dimension of index array
        current_bin_idx = self._binned_idx[self._bins == bin_num][0]
        return self._observation.obs[current_bin_idx]

    def mean_obs_in_bin(self, bin_num):
        """Return the mean value of the observations at the given bin.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: The mean value of the observations at the given bin
        :rtype: float
        """
        current_bin_idx = self._binned_idx[self._bins == bin_num][0]
        return np.mean(self._observation.obs[current_bin_idx])

    def stdmean_obs_in_bin(self, bin_num):
        """Return the standardized mean of the observation in bin_num.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: The standardized mean of the observation in bin_num
        :rtype: float
        """
        current_bin_idx = self._binned_idx[self._bins == bin_num][0]
        obs_var = np.var(self._observation.obs[current_bin_idx])
        obs_mean = np.mean(self._observation.obs[current_bin_idx])
        if obs_var > 0.:
            return obs_mean/np.sqrt(obs_var)
        else:
            return np.nan

    def ires_in_bin(self, bin_num):
        """Return the resolutions of all the observations in the given bin.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: An array of resolutions
        :rtype: ndarray of float
        """
        current_bin_idx = self._binned_idx[self._bins == bin_num][0]
        return self._observation.ires[current_bin_idx]

    def mean_invresolsq_in_bin(self, bin_num):
        """Return the mean inverse resolution squares in the given bin.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: An array of resolutions
        :rtype: ndarray of float
        """
        current_bin_idx = self._binned_idx[self._bins == bin_num][0]
        ires_in_bin = self._observation.ires[current_bin_idx]
        return np.mean(1./(ires_in_bin * ires_in_bin))

    def bin_args_in_icering(self, ice_ring):
        """Return the indices of bins appearing in the ice ring range, partially included.

        :param ice_ring: IceRing instance
        :return: An array of indices
        :rtype: ndarray of int
        """
        # assert isinstance(ice_ring, IceRing), "expect an instance of IceRing"
        # use broadcasting to enhance the efficiency of logic operation
        mean_inversolsq = self.mean_invresolsq_all()
        bin_args = np.any((mean_inversolsq[:, None] > ice_ring.ice_rings[:, 0][None, :]) &
                          (mean_inversolsq[:, None] < ice_ring.ice_rings[:, 1][None, :]),
                          axis=1)
        self._bin_args_in_icering = bin_args
        return self._bin_args_in_icering

    def bins_in_icering(self, ice_ring):
        """Return the values of bins appearing in the ice ring range, partially included.

        :param ice_ring: IceRing instance
        :return: An array of bin numbers
        :rtype: ndarray of int
        """
        if self._bin_args_in_icering is None:
            self.bin_args_in_icering(ice_ring)
        # bins = np.floor(1. / self._observation.ires[bin_args] / 0.01)
        bins = self._bins[self._bin_args_in_icering]
        return bins

    def bins_in_icering_groupped(self, ice_ring):
        if self._bin_args_in_icering is None:
            self.bin_args_in_icering(ice_ring)
        bins = self._bins[self._bin_args_in_icering]
        group_indices = np.argwhere((bins[1:] - bins[:-1]) > 1).flatten() + 1
        group_indices_lower = np.insert(group_indices, 0, 0)
        group_indices_upper = np.append(group_indices, len(bins))
        groupped_bins = []
        for l, u in zip(group_indices_lower, group_indices_upper):
            groupped_bins.append(bins[l:u])
        return groupped_bins

    def bin_args_windowed(self):
        """Return a list of the indices of grouped bins using the provided smooth parameter (half window length).

        :return: A list of bin indices
        :rtype list of ndarrays
        """
        assert self._bin_args_in_icering is not None, "bins in ice ring are unknown"
        if self._stdmeans is None:
            self.get_stdmean_all()
        window_size = 2*self._smooth_param + 1
        bin_args_windowed = [np.arange(0, i+self._smooth_param+1) for i in range(0, self._smooth_param)]
        bin_args_windowed.extend([np.arange(i-self._smooth_param, i+self._smooth_param+1)
                                  for i in range(self._smooth_param, self._bins.size - self._smooth_param)])
        bin_args_windowed.extend([np.arange(self._bins.size - window_size + i, self._bins.size)
                                  for i in range(1, self._smooth_param + 1)])
        # exclude icering and invalid stdmean
        invalid_bin_args = np.concatenate((np.where(np.isnan(self._stdmeans))[0],
                                           np.where(self._bin_args_in_icering)[0]))
        bin_args_windowed = [np.setdiff1d(bin_args, invalid_bin_args) for bin_args in bin_args_windowed]
        return bin_args_windowed

    def quartile_windowed(self):
        """Calculate the lower quantile and upper quantile within each window.

        """
        bin_args_windowed = self.bin_args_windowed()
        sorted_stdmeans_windowed = [np.sort(self._stdmeans[ind]) for ind in bin_args_windowed]
        for stdmean in sorted_stdmeans_windowed:
            if stdmean.size < self._smooth_param:
                self._lower_quantiles.append(np.nan)
                self._upper_quantiles.append(np.nan)
            else:
                self._lower_quantiles.append(stdmean[int(np.ceil(self._quantiles[0]*(stdmean.size-1)))])
                self._upper_quantiles.append(stdmean[int(np.floor(self._quantiles[1]*(stdmean.size-1)))])

    def smoothing_sd_in_bin(self, bin_num):
        """Convert bin width to sd used for Gaussian smoothing and return the smoothing sd at the given bin.

        :param bin_num: the number of given bin
        :type bin_num: int
        :return: The smoothing sd at the given bin.
        :rtype: float
        """
        mean_invresolsq = self.mean_invresolsq_in_bin(bin_num)
        tmp = np.sqrt(mean_invresolsq) + self._iresbinwidth
        bin_width = (tmp*tmp) - mean_invresolsq
        smoothing_sd = bin_width * self._smooth_param / self._smooth_sd_divisor
        return smoothing_sd
    
    def get_est_stdmeans(self):
        """Calculate the estimated standardised means of each bin.

        :return: An array of the estimated standardised means for all bins.
        :rtype: ndarray of float
        """
        if self._est_stdmeans is not None:
            return self._est_stdmeans

        if not self._lower_quantiles:
            self.quartile_windowed()
        mean_invresolsq_all = self.mean_invresolsq_all()
        weighted_stdmean = list()
        for idx, bin_arg in enumerate(self.bin_args_windowed()):
            if bin_arg.size < self._smooth_param or np.isnan(self._stdmeans[idx]):
                weighted_stdmean.append(np.nan)
            else:
                # weights = np.random.rand(bin_arg.size)
                stdmean_list = self._stdmeans[bin_arg]
                invalid_args = np.isnan(stdmean_list)
                stdmean_list = stdmean_list[~invalid_args]
                out_quantile_range_bool = (stdmean_list >= self._lower_quantiles[idx]) & \
                                          (stdmean_list <= self._upper_quantiles[idx]) == False
                weight = _norm_pdf(mean_invresolsq_all[idx],
                                   mean_invresolsq_all[bin_arg][~invalid_args],
                                   self.smoothing_sd_in_bin(self._bins[idx]))
                weight[out_quantile_range_bool] = 0.
                weight_sum = np.sum(weight)
                if weight_sum == 0.:
                    weighted_stdmean.append(np.nan)
                    continue
                weight_ratio = weight/weight_sum
                weighted_stdmean.append(np.sum(stdmean_list * weight_ratio))
        self._est_stdmeans = np.array(weighted_stdmean, dtype=float)

        # fill out start
        invalid_bools = np.isnan(self._est_stdmeans)
        valid_args = np.where(~invalid_bools)[0]
        min_arg_nan = valid_args.min()
        if np.all(np.isnan(self._est_stdmeans[:min_arg_nan])):
            self._est_stdmeans[:min_arg_nan] = self._est_stdmeans[min_arg_nan+1]
        else:
            self._est_stdmeans[min_arg_nan] = self._est_stdmeans[min_arg_nan+1]

        # fill out end
        max_arg_nan = valid_args.max() + 1
        if np.all(np.isnan(self._est_stdmeans[max_arg_nan:])):
            self._est_stdmeans[max_arg_nan:] = self._est_stdmeans[max_arg_nan-1]
        else:
            self._est_stdmeans[max_arg_nan] = self._est_stdmeans[max_arg_nan-1]

        # linear interpolation for hexagonal regions
        args_in_hexagonal = np.where(np.isnan(self._est_stdmeans) & ~np.isnan(self._stdmeans))[0]
        non_continous_args = np.where(args_in_hexagonal[1:] - args_in_hexagonal[:-1] != 1)[0]
        start_idx = np.concatenate(([0], non_continous_args+1))
        end_idx = np.concatenate((non_continous_args+1, [args_in_hexagonal.size]))
        for ind_0, ind_1 in zip(start_idx, end_idx):
            args = args_in_hexagonal[ind_0:ind_1]
            start_val = self._est_stdmeans[args[0]-1]
            end_val = self._est_stdmeans[args[-1]+1]
            self._est_stdmeans[args] = np.linspace(start_val, end_val, args.size+2)[1:-1]
        return self._est_stdmeans

    def icefinder_score(self):
        """Calculate the icefinder scores of each bin.

        :return: An array of icefinder scores for all bins.
        :rtype: ndarray of float
        """
        if self._est_stdmeans is None:
            self.get_stdmean_all()
        return (self._stdmeans - self._est_stdmeans) * np.sqrt(self._no_obs_binned)

    def get_stdmean_all(self):
        """Calculate the standardised means of each bin.

        :return: An array of standardised means for all bins.
        :rtype: ndarray of float
        """
        vectorized_stdmean = np.vectorize(self.stdmean_obs_in_bin)
        self._stdmeans = vectorized_stdmean(self._bins)
        return self._stdmeans

    def mean_invresolsq_all(self):
        """Calculate the inverse resolution squares of each bin.

        :return: An array of inverse resolution squares for all bins.
        :rtype: ndarray of float
        """
        vectorized_mean_inveresolsq = np.vectorize(self.mean_invresolsq_in_bin)
        return vectorized_mean_inveresolsq(self._bins)

    @property
    def no_obs_binned(self):
        """
        :return: The number of observations of each bin.
        :rtype: ndarray of int
        """
        return self._no_obs_binned

    @property
    def bins(self):
        """
        :return: A list of bins.
        :rtype: ndarray of int
        """
        return self._bins

    @property
    def iresbinwidth(self):
        """
        :return: Bin width, provided as inverse resolution.
        :rtype: float
        """
        return self._iresbinwidth


class BinnedStatistics(object):
    def __init__(self):
        self._ires_binned = None
        self._i_mean_binned = None
        self._i_over_sigma_binned = None
        self._completeness_binned = None
        self._redundancy_binned = None
        self._r_pim_binned = None
        self._r_merge_binned = None
        self._r_meas_binned = None
        self._cc_half_binned = None
        self._num_data_binned = None

    @classmethod
    def const_stats(cls, resolution, num_data, i_mean, i_over_sigma, completeness, redundancy, r_pim, r_merge, r_meas, cc_half):
        obj = cls.__new__(cls)
        super(BinnedStatistics, obj).__init__()
        obj._ires_binned = resolution
        obj._i_mean_binned = i_mean
        obj._i_over_sigma_binned = i_over_sigma
        obj._completeness_binned = completeness
        obj._redundancy_binned = redundancy
        obj._r_pim_binned = r_pim
        obj._r_merge_binned = r_merge
        obj._r_meas_binned = r_meas
        obj._cc_half_binned = cc_half
        obj._num_data_binned = num_data
        return obj

    @property
    def ires_binned(self):
        return self._ires_binned

    @property
    def num_data_binned(self):
        return self._num_data_binned

    @property
    def i_mean_binned(self):
        return self._i_mean_binned

    @property
    def i_over_sigma_binned(self):
        return self._i_over_sigma_binned

    @property
    def completeness_binned(self):
        return self._completeness_binned

    @property
    def redundancy_binned(self):
        return self._redundancy_binned

    @property
    def r_pim_binned(self):
        return self._r_pim_binned

    @property
    def r_merge_binned(self):
        return self._r_merge_binned

    @property
    def r_meas_binned(self):
        return self._r_meas_binned

    @property
    def cc_half_binned(self):
        return self._cc_half_binned

    @ires_binned.setter
    def ires_binned(self, _):
        self._ires_binned = _

    @i_mean_binned.setter
    def i_mean_binned(self, _):
        self._i_mean_binned = _

    @num_data_binned.setter
    def num_data_binned(self, _):
        self._num_data_binned = _

    @i_over_sigma_binned.setter
    def i_over_sigma_binned(self, _):
        self._i_over_sigma_binned = _

    @completeness_binned.setter
    def completeness_binned(self, _):
        self._completeness_binned = _

    @redundancy_binned.setter
    def redundancy_binned(self, _):
        self._redundancy_binned = _

    @r_pim_binned.setter
    def r_pim_binned(self, _):
        self._r_pim_binned = _

    @r_merge_binned.setter
    def r_merge_binned(self, _):
        self._r_merge_binned = _

    @r_meas_binned.setter
    def r_meas_binned(self, _):
        self._r_meas_binned = _

    @cc_half_binned.setter
    def cc_half_binned(self, _):
        self._cc_half_binned = _

    def get_stats_as_list(self):
        return self._ires_binned, self._num_data_binned, self.i_mean_binned, self._i_over_sigma_binned, self._completeness_binned, \
            self._redundancy_binned, self._r_pim_binned, self._r_merge_binned, self._r_meas_binned, self._cc_half_binned


def _norm_pdf(x, m, s):
    inv_sqrt_2pi = 0.3989422804014327
    a = (x - m)/s
    return inv_sqrt_2pi / s * np.exp(-0.5 * a * a)


