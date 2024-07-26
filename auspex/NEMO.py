import copy
from typing import Optional

import numpy as np

import os, ctypes
from scipy import LowLevelCallable
from scipy.special import erfc
from scipy.integrate import nquad

from sklearn.cluster import HDBSCAN

from cctbx.array_family import flex
from mmtbx.scaling import absolute_scaling

import matplotlib.pyplot as plt

from ReflectionData import Mtz, Xds, PlainASCII

from mpi4py import MPI

# initiate c lib for complex integral
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'lib/int_lib.so'))
lib.f.restype = ctypes.c_double
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

class NemoHandler(object):
    def __init__(self, reso_min: float = 10.):
        """Constructor class for automatic NEMO detection. Default hyperparameters t,l,m1,m2,m3 (t_i, l_i, for intensities)
        are trained on the reflection data.

        :param reso_min: minimum resolution to check.
        :type reso_min: float.
        """
        super(NemoHandler, self).__init__()
        self._reso_low = None
        self._sorted_arg = None
        self._reso_min = reso_min
        self._refl_data = None
        self._work_obs = None
        self._work_norma_obs = None
        self._centric_flag = None
        self._acentric_flag = None
        self._prob_c = None
        self._final_nemo_ind = None
        self._original_row_ind = None
        self._detect_option = ['obs_over_sig', 'obs']
        self._t = 0.0248  # hyperparamter t: french_wilson_level, snr trained 0.0248
        self._t_i = 0.2265  # hyperparamter t for intensity: snr trained 0.2265
        self._l = 0.496  # hyperparameter l: intersection fraction, snr trained 0.496
        self._l_i = 0.598  # hyperparameter l for intensity: intersection fraction, snr trained 0.598
        self._m1 = 0.109  # recurrence rate below 30 Angstrom, snr trained 0.109.
        self._m1_i = 0.083  # recurrence rate below 30 Angstrom for intensity, snr trained 0.083.
        self._m2 = 0.519  # recurrence rate between 30-20 Angstrom, snr trained 0.519.
        self._m3 = 0.787  # recurrence rate between 20-10 Angstrom, snr trained  0.787.

    def refl_data_prepare(self, reflection_data: Mtz.MtzParser | Xds.XdsParser, observation_label: str = 'FP'):
        """Construct the initial set A. Conduct kernel normalization. The centric reflections and acentric reflections

        :param reflection_data: One of the supported ReflectionData instance.
        :param observation_label: The label of the observation. Can be 'FP' ('F', 'FMEANS') or 'I' ('IMEANS', 'IMEAN').
        :return: None.
        """
        self._refl_data = reflection_data
        self._work_obs = reflection_data.get_miller_array(observation_label)
        d_spacings = self._work_obs.d_spacings().data().as_numpy_array()
        self._sorted_arg = d_spacings.argsort()[::-1]
        self._reso_select = (d_spacings > self._reso_min).sum()
        self._reso_low = d_spacings[self._sorted_arg][:self._reso_select]
        self._obs_low = self._work_obs.data().as_numpy_array()[self._sorted_arg][:self._reso_select]
        self._sig_low = self._work_obs.sigmas().as_numpy_array()[self._sorted_arg][:self._reso_select]
        normalizer = absolute_scaling.kernel_normalisation(self._work_obs, auto_kernel=50)
        if self._work_obs.is_xray_amplitude_array():
            self._work_norma_obs = self._work_obs.customized_copy(
                data=flex.sqrt(normalizer.normalised_miller_dev_eps.data())
            )
        if self._work_obs.is_xray_intensity_array():
            self._work_norma_obs = self._work_obs.customized_copy(
                data=normalizer.normalised_miller_dev_eps.data(),
                sigmas=normalizer.normalised_miller_dev_eps.sigmas()
            )
        self._centric_flag = self._work_norma_obs.centric_flags().data().as_numpy_array()[self._sorted_arg][:self._reso_select]
        self._acentric_flag = ~self._work_norma_obs.centric_flags().data().as_numpy_array()[self._sorted_arg][:self._reso_select]
        self._centric_ind_low = self._sorted_arg[:self._reso_select][self._centric_flag]
        self._acentric_ind_low = self._sorted_arg[:self._reso_select][self._acentric_flag]

    def outliers_by_wilson(self, prob_level: float = 0.01) -> tuple[np.ndarray[np.bool_], np.ndarray[np.bool_]]:
        """Find outliers by Wilson statistics. Calculate the probability of a reflection smaller than a certain value
        to be observed according to Wilson statistics. Return those with probability smaller than prob_level.

        :param prob_level: The probability threshold to be applied. Default value 0.01.
        :return: [outlier flags for acentric reflections, outlier flags for centric reflections]
        :rtype: tuple of two ndarrays
        """
        ac_obs = self._work_norma_obs.data().as_numpy_array()[self._acentric_ind_low]
        c_obs = self._work_norma_obs.data().as_numpy_array()[self._centric_ind_low]
        if self._work_norma_obs.is_xray_amplitude_array():
            self._prob_ac = cumprob_ac_amplitude(ac_obs)
            self._prob_c = cumprob_c_amplitude(c_obs)
            ac_outlier_flag = self._prob_ac < prob_level
            c_outlier_flag = self._prob_c < prob_level
        elif self._work_norma_obs.is_xray_intensity_array():
            ac_sigs = self._work_norma_obs.sigmas().as_numpy_array()[self._acentric_ind_low]
            c_sigs = self._work_norma_obs.sigmas().as_numpy_array()[self._centric_ind_low]
            self._prob_ac = cumprob_ac_intensity(ac_obs, ac_sigs)

            ac_outlier_flag = self._prob_ac < prob_level
            if c_obs.sum() == 0:  # no centric reflections
                self._prob_c = np.array([], dtype=float)
                c_outlier_flag = np.array([], dtype=bool)
            else:
                self._prob_c = cumprob_c_intensity(c_obs, c_sigs)
                c_outlier_flag = self._prob_c < prob_level
        else:
            raise Exception("Unknown observation type: {}".format(self._work_norma_obs.observation_type()))
        return ac_outlier_flag, c_outlier_flag

    def mmtbx_beamstop_outlier(self, level: float = 0.01) -> np.ndarray[np.int16]:
        """Return the row indices of beamstop outliers identified according to the probability threshold set by
        mmtbx.scale.outlier_rejection.

        :param level: Default probability threshold set by mmtbx.scale.outlier_rejection: 0.01.
        :return: Row indices of the identified outliers.
        :rtype: Nx1 ndarray(dtype=int)
        """
        if self._work_norma_obs.is_xray_amplitude_array():
        # the default level for beamstop outliers in mmtbx.scaling::outlier_rejection is set to 0.01
            ac_outlier_flag, c_outlier_flag = self.outliers_by_wilson(level)
            ind_weak = np.concatenate((self._acentric_ind_low[ac_outlier_flag], self._centric_ind_low[c_outlier_flag]))
            ind_false_sigma = np.argwhere(self._refl_data.sigF <= 0.).flatten()
        if self._work_norma_obs.is_xray_intensity_array():
            ac_outlier_flag, c_outlier_flag = self.outliers_by_wilson(level)
            ind_weak = np.concatenate((self._acentric_ind_low[ac_outlier_flag], self._centric_ind_low[c_outlier_flag]))
            ind_false_sigma = np.argwhere(self._refl_data.sigI <= 0.).flatten()
        in_add = np.sum((ind_weak[:, None] - ind_false_sigma[None, :]) >= 0, axis=1)
        recovered_ind = ind_weak + in_add
        return np.sort(recovered_ind)

    def cluster_detect(self, y_option:  0 | 1 = 0):
        """The core algorithm for NEMO detection. Record the indices of NEMOs corresponding to the input reflection data.
        While x is fixed to d-spacing squared, choose y_option from 0 or 1 to determine what is to be used as y
        (0: signal-to-noise ratio; 1: signal only).

        :param y_option: 0 or 1 (0: signal-to-noise ratio; 1: signal only). Default value: 0.
        :return:None
        """
        # it is possible low-resolution cutoff smaller than 10 angstrom, e.g. 7r33
        if self._acentric_ind_low.size == 0 and self._centric_ind_low.size == 0:
            self._final_nemo_ind = np.empty((0,), dtype=int)
            return None

        y_option_ = self._detect_option[y_option]

        ac_outlier_flag, c_outlier_flag = self.outliers_by_wilson(0.5)
        # ac_weak = self._obs_low[self._acentric_flag][ac_outlier_flag]
        # c_weak = self._obs_low[self._centric_flag][c_outlier_flag]
        # d_ac_weak = 1./self._reso_low[self._acentric_flag][ac_outlier_flag]**2
        # d_c_weak = 1./self._reso_low[self._centric_flag][c_outlier_flag]**2
        # sig_ac_weak = self._sig_low[self._acentric_flag][ac_outlier_flag]
        # sig_c_weak = self._sig_low[self._centric_flag][c_outlier_flag]

        ind_weak = np.concatenate((self._acentric_ind_low[ac_outlier_flag], self._centric_ind_low[c_outlier_flag]))
        weak_prob = np.concatenate([self._prob_ac[ac_outlier_flag], self._prob_c[c_outlier_flag]])

        if ind_weak.size == 1:
            # if only one outlier by wilson then we become conservative. level -> 0.001
            conserv_ind_weak = ind_weak[ind_weak <= 1e-3]
            self._final_nemo_ind = conserv_ind_weak
            return conserv_ind_weak
        # i = np.concatenate((d_ac_weak, d_c_weak))
        if y_option_ == 'obs_over_sig':
            # j = np.concatenate((ac_weak/sig_ac_weak, c_weak/sig_c_weak))
            auspex_array = np.vstack((1. / (self._reso_low ** 2),
                                      np.divide(self._obs_low, self._sig_low, out=np.zeros_like(self._obs_low), where=self._sig_low != 0.))
                                     ).transpose()
        if y_option_ == 'obs':
            # j = np.concatenate((ac_weak, c_weak))
            auspex_array = np.vstack((1. / (self._reso_low ** 2), self._obs_low)).transpose()

        if self._work_obs.is_xray_amplitude_array():
            ind_weak_work = copy.deepcopy(ind_weak)[weak_prob <= self._t]
            max_search_size = np.sum(
                weak_prob <= 0.015)  # setting minimum noise level. It seems reaching an extremely low noise level is unecessary.
        if self._work_obs.is_xray_intensity_array():
            ind_weak_work = copy.deepcopy(ind_weak)[weak_prob <= self._t_i]
            max_search_size = np.sum(
                weak_prob <= 0.125)  # setting minimum noise level. It seems reaching an extremely low noise level is unecessary.

        auspex_array_for_fit = copy.deepcopy(auspex_array)
        auspex_array_for_fit[:, 0] = np.percentile(auspex_array_for_fit[:, 1], 95) / auspex_array_for_fit[:, 0].max() * auspex_array[:, 0]

        ind_cluster_by_size = []

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()

        bootstrapping_range = range(max_search_size, 1, -1)
        # for num_points in range(max_search_size, 1, -1):
        for i, task in enumerate(bootstrapping_range):
            if i % size != rank:
                continue

            detect = HDBSCAN(min_cluster_size=task)  #TODO: code performance comparison with hdbscan lib
                             #min_samples=ind_weak_work.size-num_points+1, # not needed because the noise bootstraping strategy
                             #max_cluster_size=ind_weak_work.size,  # not needed because the noise bootstraping strategy
                             #algorithm='brute') # the default works fine
            try:
                cluster_fitted = detect.fit(auspex_array_for_fit)
            except KeyError:
                continue
            cluster_labels = cluster_fitted.labels_
            cluster_prob = cluster_fitted.probabilities_
            unique_cluster_label = np.unique(cluster_labels)
            unique_cluster_label = unique_cluster_label[unique_cluster_label >= 0]
            if unique_cluster_label.size == 0:
                continue
            else:
                # initiation
                in_token = np.empty(0, dtype=int)
                in_prob = np.empty(0, dtype=float)
                for c_label in unique_cluster_label:
                    args_ = np.argwhere((cluster_labels == c_label) & (cluster_prob >= 0.49)).flatten()
                    if args_.size == 0:
                        continue
                    ind_sub_cluster = self._sorted_arg[:self._reso_select][args_]
                    wilson_filter = np.isin(ind_sub_cluster, ind_weak_work)

                    s_ij = (wilson_filter.sum() / wilson_filter.size) > self._l_i
                    if ((self._work_obs.is_xray_amplitude_array() and s_ij > self._l) or
                            (self._work_obs.is_xray_intensity_array() and s_ij > self._l_i)):
                        in_token = np.append(in_token, ind_sub_cluster)
                        in_prob = np.append(in_prob, cluster_prob[args_])

                ind_cluster_by_size.append(np.unique(in_token))

        if not ind_cluster_by_size:
            # when no cluster can be found we need to be very conservative thus level -> 0.001
            conserv_ind_weak = ind_weak[weak_prob <= 1e-3]
            self._final_nemo_ind = conserv_ind_weak
            return conserv_ind_weak

        cluster_ind_recur, cluster_counts_recur = np.unique(np.concatenate(ind_cluster_by_size), return_counts=True)
        if cluster_ind_recur.size == 0 or cluster_ind_recur.size == 1:
            # when the intersection of the cluster and wilson outliers has only one element,
            # we need to be very conservative thus level 0.01->0.001
            final_weak_ind = ind_weak[weak_prob <= 1e-3]
        else:
            # when the elements in the clusters are varying.
            if self._work_obs.is_xray_amplitude_array():
                repetitive_ind_30 = (cluster_counts_recur >= cluster_counts_recur.max() * self._m1) & \
                                    (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] >= 30.)
            if self._work_obs.is_xray_intensity_array():
                repetitive_ind_30 = (cluster_counts_recur >= cluster_counts_recur.max() * self._m1_i) & \
                                    (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] >= 30.)
            repetitive_ind_20 = (cluster_counts_recur >= cluster_counts_recur.max() * self._m2) & \
                                (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] >= 20.) & \
                                (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] < 30.)
            repetitive_ind_10 = (cluster_counts_recur >= cluster_counts_recur.max() * self._m3) & \
                                (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] >= 10.) & \
                                (self._work_obs.d_spacings().data().as_numpy_array()[cluster_ind_recur] < 20.)
            ind_weak_and_cluster = np.unique(
                np.concatenate((ind_weak[(weak_prob <= 1e-3)],
                                cluster_ind_recur[repetitive_ind_30],
                                cluster_ind_recur[repetitive_ind_20],
                                cluster_ind_recur[repetitive_ind_10]))
            )
            final_weak_ind = ind_weak_and_cluster

        self._final_nemo_ind = final_weak_ind

    def get_nemo_indices(self) -> np.ndarray[np.int16]:
        """Return the row indices of NEMOs identified in the corresponding reflection data.

        :return: Row indices of NEMOs.
        :rtype: Nx3 numpy.ndarray(dtype=int)
        """
        assert self._final_nemo_ind is not None
        return self._work_obs.indices().as_vec3_double().as_numpy_array()[self._final_nemo_ind].astype(int)

    def get_nemo_D2(self) -> np.ndarray[np.float32]:
        """Return the inverse d-spacing squared of NEMOs identified in the corresponding reflection data.

        :return: Inverse d-spacing squared of NEMOs.
        :rtype: Nx1 numpy.ndarray(dtype=float)
        """
        assert self._final_nemo_ind is not None
        return 1./self._work_obs.d_spacings().data().as_numpy_array()[self._final_nemo_ind] ** 2

    def get_nemo_data(self) -> np.ndarray["N", np.float32]:
        """Return the amplitude/intensity value of NEMOs identified in the corresponding reflection data.

        :return: Amplitude/Intensity value of NEMOs.
        :rtype: Nx1 numpy.ndarray(dtype=float)
        """
        assert self._final_nemo_ind is not None
        return self._work_obs.data().as_numpy_array()[self._final_nemo_ind]

    def get_nemo_sig(self) -> np.ndarray[np.float32]:
        """Return the sigmas of NEMOs identified in the corresponding reflection data.

        :return: Sigmas of NEMOs.
        :rtype: Nx1 numpy.ndarray(dtype=float)
        """
        assert self._final_nemo_ind is not None
        return self._work_obs.sigmas().as_numpy_array()[self._final_nemo_ind]

    def get_nemo_data_over_sig(self) -> np.ndarray[float]:
        """Return the signal-to-noise ratio of NEMOs identified in the corresponding data.

        :return: Signal-to-noise ratio of NEMOs.
        :rtype: Nx1 numpy.ndarray(dtype=float)
        """
        assert self._final_nemo_ind is not None
        return self._work_obs.data().as_numpy_array()[self._final_nemo_ind] / self._work_obs.sigmas().as_numpy_array()[self._final_nemo_ind]

    def add_false_sigma_record_back(self, return_idx: bool = False) -> Optional[np.ndarray[int]]:
        """Recover the original row number by adding back the invalid rows removed by miller_array.
        cctbx miller array automatically remove invalid observations with 0 or negative sigma. For any operation on the
        original record, the row number of the original records is needed. This function calculates the corresponding
        row number of NEMOs before the removal of invalid observations.

        :param return_idx: If True, return the original row indices of NEMOs.
        :return: Original row indices of NEMOs.
        :rtype: Nx1 numpy.ndarray(dtype=int)
        """
        if self._work_obs.is_xray_amplitude_array():
            ind_false_sigma = np.argwhere(self._refl_data.sigF <= 0.).flatten()
        if self._work_obs.is_xray_intensity_array():
            ind_false_sigma = np.argwhere(self._refl_data.sigI <= 0.).flatten()
        # indices recoverd by calculating how many indices in ind_false_sigma are smaller than any given index in final_nemo_ind
        in_add = np.sum((self._final_nemo_ind[:, None] - ind_false_sigma[None, :]) >= 0, axis=1)
        self._original_row_ind = self._final_nemo_ind + in_add
        if return_idx is True:
            return self._original_row_ind

    def get_nemo_row_ind(self):
        if self._final_nemo_ind is None:
            self.cluster_detect(0)
        if self._original_row_ind is None:
            self.add_false_sigma_record_back()
        if self._refl_data.source_data_format == 'xds_hkl':
            hkl_array = self._refl_data.get_merged_hkl()[self._final_nemo_ind]
            row_exclude = []
            for hkl in hkl_array:
                equiv_rows = self._refl_data.find_equiv_refl(*hkl)
                row_exclude.append(equiv_rows)
            self._original_row_ind = np.concatenate(row_exclude)
        return self._original_row_ind

    def ft_and_tar(self) -> tuple[np.ndarray, np.ndarray]:
        return np.arange(0, self._reso_select), np.isin(self._sorted_arg[:self._reso_select], self._final_nemo_ind).astype(int)

    def weak_by_signal_to_noise(self, level: float = 6.) -> np.ndarray[bool]:
        """Return the indices of weak observations with high errors.

        :param level: The threshold of signal-to-noise level. Default: 6.
        :return: Indices of weak observations with high errors.
        :rtype: Nx1 numpy.ndarray(dtype=int)
        """
        if self._work_obs.is_xray_amplitude_array():
            ind_weak = np.argwhere((self._refl_data.F / self._refl_data.sigF <= level)
                                   & (self._refl_data.resolution > 10.)).flatten()
        if self._work_obs.is_xray_intensity_array():
            ind_weak = np.argwhere((self._refl_data.I / self._refl_data.sigI <= level)
                                   & (self._refl_data.resolution > 10.)).flatten()
        return ind_weak

    def NEMO_removal(self, filename: str):
        """Remove NEMOs from the given dataset and write into a new one. Current supported format: mtz.

        :param filename: The output path or file name.
        """
        assert self._original_row_ind is not None
        isel = flex.size_t(self._original_row_ind)
        self._refl_data._obj.delete_reflections(isel)
        self._refl_data._obj.write(filename)

    def write_filter_hkl(self, integrate_hkl_plain: PlainASCII.IntegrateHKLPlain, hkl_array: np.ndarray):
        """Generate FILTER.HKL for XDS. FILTER.HKL will be written to pwd.

        :param integrate_hkl_plain: An IntegrateHKLPlain instance.
        :param hkl_array: Nx3 array of hkl indices.
        """
        row_exclude = []
        for hkl in hkl_array:
            equiv_rows = integrate_hkl_plain.find_equiv_refl(*hkl)
            #print(equiv_rows)
            for _ in equiv_rows:
                if integrate_hkl_plain.corr[_] < 20 and ~np.any(integrate_hkl_plain.xyz_obs[_]):
                    row_exclude.append(_)
        lines = []
        for row in row_exclude:
            lines.append("{:2d} {:2d} {:2d} {:6.1f} {:6.1f} {:6.1f} 0.5 0.5 0.5\n".format(
                *integrate_hkl_plain.hkl[row], *integrate_hkl_plain.xyz_cal[row]
            ))
        with open('FILTER.HKL', 'w') as f:
            f.writelines(lines)

def cumprob_c_amplitude(e):
    """Calculate the probability of normalised centric amplitude smaller than a value.
    Ref: READ Acta. Cryst. (1999). D55, 1759-1764

    :param e: The normalised centric amplitude value.
    :return: Probability of normalised centric amplitude smaller than e.
    """
    return 1. - erfc(e / 1.4142)


def cumprob_ac_amplitude(e):
    """Calculate the probability of normalised acentric amplitude smaller than a value.
    Ref: READ Acta. Cryst. (1999). D55, 1759-1764

    :param e: The normalised acentric amplitude value.
    :return: Probability of normalised acentric amplitude smaller than e.
    """
    return 1 - np.exp(-e*e)


def cumprob_ac_intensity(e_square, sig):
    """Calculate the probability of normalised acentric intensity smaller than a value with a given sigma.
    Ref: READ Acta. Cryst. (2016). D72, 375-387

    :param e_square: The normalised acentric intensity value.
    :param sig: The normalised acentric intensity sigma value.
    :return: Probability of normalised acentric intensity smaller than e_square with given sigma value sig.
    """
    #
    # READ Acta. Cryst. (2016). D72, 375-387
    e_square_div_sig = np.divide(e_square, sig, out=np.zeros_like(e_square), where=sig != 0.)
    sig_minus_e_square_div_sig = np.divide(sig - e_square, sig, out=np.zeros_like(e_square), where=sig != 0.)
    return 0.5 * (erfc(-e_square_div_sig / 1.4142) - np.exp((sig - 2 * e_square) / 2) * erfc(
        sig_minus_e_square_div_sig / 1.4142))

# def prob_c_intensity(e_square, sig):
#     # probability Baysian denominator for centric intensity, given read e**2 and sigma sig.
#     # READ Acta. Cryst. (2016). D72, 375-387
#     # equation 9b. analytical definite integral
#     # fast but intolerant to small values
#     p = 0.5 / np.sqrt(np.pi * sig) * np.exp(1 / 16 * (sig * sig - 4 * e_square - 4 * e_square * e_square / (sig * sig))) * \
#          pbdv(-0.5, 0.5 * sig - e_square / sig)[0]
#     return p

# def prob_c_intensity_integrand(x, e_square, sig):
#     # equation 9b. integrand
#     return 1/np.sqrt(2*np.pi*sig*sig)*np.exp(-0.5*np.square(e_square-x)/(sig*sig))/np.sqrt(2*np.pi*x)*np.exp(-0.5*x)

# def cumprob_c_intensity(e_square, sig):
#     # equation 21a. double numerical integral.
#     # very slow.
#     return nquad(prob_c_intensity_integrand, [[0, np.inf], [-np.inf, e_square]], args=(sig,))[0]

# def prob_c_intensity(e_square, sig):
#     # equation 9a. numerical integral.
#     return quad(prob_c_intensity_integrand, 0, np.inf, args=(e_square, sig))[0]

# def cumprob_c_intensity(e_square, sig):
#     # probability of centric normalised intensity smaller than e**2, given read e**2 and sigma sig.
#     # READ Acta. Cryst. (2016). D72, 375-387
#     # equation 21a. numerical integral.
#     # slowest.
#     return quad(prob_c_intensity, -np.inf, e_square, args=(sig,))[0]


@np.vectorize
def cumprob_c_intensity(e_square, sig):
    """Calculate the probability of normalised acentric intensity smaller than a value with a given sigma.
    Ref: READ Acta. Cryst. (2016). D72, 375-387

    :param e_square: The normalised centric intensity value.
    :param sig: he normalised centric intensity sigma value.
    :return:  Probability of normalised centric intensity smaller than e_square with given sigma value sig.
    """
    # based on low level c for better speed
    c = ctypes.c_double(sig+1e-9)
    user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
    prob_c_intensity_integrand = LowLevelCallable(lib.f, user_data)
    return nquad(prob_c_intensity_integrand, [[0, np.inf], [-np.inf, e_square]], opts={"limit": 301})[0]


def construct_ih_table(obs, inv_res_sqr):
    obs_ext = obs[None, :] * np.ones(obs.size)[:, None]
    inv_res_sqr_ext = inv_res_sqr[None, :] * np.ones(inv_res_sqr.size)[:, None]
    obs_ih_table = delete_diag(obs_ext)
    inv_res_sqr_ih_table = delete_diag(inv_res_sqr_ext)
    return obs_ih_table, inv_res_sqr_ih_table

def delete_diag(square_matrix):
    # inspired by https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
    # delete diagonal elements
    m = square_matrix.shape[0]
    s0, s1 = square_matrix.strides
    return np.lib.stride_tricks.as_strided(square_matrix.ravel()[1:], shape=(m-1, m), strides=(s0+s1, s1)).reshape(m, -1)

def plot(x0 , y0, x1, y1, c, path):
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    ax.scatter(x0, y0/np.percentile(y0, 95), color='grey', s=10, alpha=0.5)
    ax.scatter(x1, y1/np.percentile(y0, 95), color=c, s=5, alpha=0.5)
    ax.set_xticks([0., 0.002, 0.004, 0.006, 0.008, 0.01])
    ax.set_yticks([])
    ax.set_xticklabels(['inf', '22.4', '15.8', '12.9', '11.2', '10.0'])
    fig.tight_layout()
    fig.savefig(path, transparent=False, dpi=75)
    plt.close(fig)