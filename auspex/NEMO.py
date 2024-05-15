import copy
import warnings

import numpy as np

import os, ctypes
from scipy import LowLevelCallable
from scipy.special import erfc
from scipy.integrate import nquad

from sklearn.cluster import HDBSCAN

from cctbx.array_family import flex
from mmtbx.scaling import absolute_scaling

import matplotlib.pyplot as plt

# initiate c lib for complex integral
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'lib/int_lib.so'))
lib.f.restype = ctypes.c_double
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

warnings.filterwarnings("ignore", message="The integral is probably divergent, or slowly convergent.")


class NemoHandler(object):
    def __init__(self, reso_min=10.):
        """Constructor class for automatic NEMO detection. Default hyperparameters t,l,m1,m2,m3 are trained on the
        reflection data.

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
        self._t = 0.0248  # hyperparamter t: french_wilson_level, trained 0.0248
        self._t_i = 0.0565  # hyperparamter t for intensity: trained 0.0565
        self._l = 0.296  # hyperparameter l: intersection fraction, trained 0.296
        self._l_i = 0.598  # hyperparameter l for intensity: intersection fraction, trained 0.598
        self._m1 = 0.100  # recurrence rate below 30 Angstrom, neither important for F nor I
        self._m2 = 0.598  # recurrence rate between 30-20 Angstrom, trained 0.598. not important for F, important for I
        self._m3 = 0.787  # recurrence rate between 20-10 Angstrom, trained 0.787. neither important for F nor I

    def refl_data_prepare(self, reflection_data, observation_label='FP'):
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

    def outliers_by_wilson(self, prob_level=0.02):
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

    def mmtbx_beamstop_outlier(self, level=0.01):
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

    def cluster_detect(self, y_option=0):
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
            # if only one outlier by wilson then we become conservative. level 0.02 -> 0.005
            conserv_ind_weak = ind_weak[ind_weak <= 1e-3]
            self._final_nemo_ind = conserv_ind_weak
            return conserv_ind_weak
        # i = np.concatenate((d_ac_weak, d_c_weak))
        if y_option_ == 'obs_over_sig':
            # j = np.concatenate((ac_weak/sig_ac_weak, c_weak/sig_c_weak))
            auspex_array = np.vstack((1. / (self._reso_low ** 2), self._obs_low / self._sig_low)).transpose()
        if y_option_ == 'obs':
            # j = np.concatenate((ac_weak, c_weak))
            auspex_array = np.vstack((1. / (self._reso_low ** 2), self._obs_low)).transpose()

        if self._work_obs.is_xray_amplitude_array():
            ind_weak_work = copy.deepcopy(ind_weak)[weak_prob <= self._t]

        if self._work_obs.is_xray_intensity_array():
            ind_weak_work = copy.deepcopy(ind_weak)[weak_prob <= self._t_i]

        auspex_array_for_fit = copy.deepcopy(auspex_array)
        auspex_array_for_fit[:, 0] = np.percentile(auspex_array_for_fit[:, 1], 95) / auspex_array_for_fit[:, 0].max() * auspex_array[:, 0]

        ind_cluster_by_size = []
        max_search_size = np.sum(weak_prob <= 0.01)  # setting maximum noise level
        for num_points in range(max_search_size, 1, -1):
            detect = HDBSCAN(min_cluster_size=num_points)
                             #min_samples=ind_weak_work.size-num_points+1,
                             #max_cluster_size=ind_weak_work.size,
                             #algorithm='brute')
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
                    args_ = np.argwhere((cluster_labels == c_label) & (cluster_prob >= 0.48)).flatten()
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
            # when no cluster can be found we need to be very conservative thus level 0.02->0.005
            conserv_ind_weak = ind_weak[weak_prob <= 1e-3]
            self._final_nemo_ind = conserv_ind_weak
            return conserv_ind_weak

        cluster_ind_recur, cluster_counts_recur = np.unique(np.concatenate(ind_cluster_by_size), return_counts=True)
        if cluster_ind_recur.size == 0 or cluster_ind_recur.size == 1:
            # when the intersection of the cluster and wilson outliers has only one element,
            # we need to be very conservative thus level 0.02->0.005
            final_weak_ind = ind_weak[weak_prob <= 1e-3]
        else:
            # when the elements in the clusters are varying. only those with 0.8 occurrence rate will pass
            repetitive_ind_30 = (cluster_counts_recur >= cluster_counts_recur.max() * self._m1) & \
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

    def get_nemo_indices(self):
        assert self._final_nemo_ind is not None
        return self._work_obs.indices().as_vec3_double().as_numpy_array()[self._final_nemo_ind].astype(int)

    def get_nemo_D2(self):
        assert self._final_nemo_ind is not None
        return 1./self._work_obs.d_spacings().data().as_numpy_array()[self._final_nemo_ind] ** 2

    def get_nemo_data(self):
        assert self._final_nemo_ind is not None
        return self._work_obs.data().as_numpy_array()[self._final_nemo_ind]

    def get_nemo_sig(self):
        assert self._final_nemo_ind is not None
        return self._work_obs.sigmas().as_numpy_array()[self._final_nemo_ind]

    def get_nemo_data_over_sig(self):
        assert self._final_nemo_ind is not None
        return self._work_obs.data().as_numpy_array()[self._final_nemo_ind] / self._work_obs.sigmas().as_numpy_array()[self._final_nemo_ind]

    def add_false_sigma_record_back(self, return_idx=False):
        # miller array automatically remove invalid observations with 0 or negative sigma.
        # for any operation on the original record, the row number of the original records is needed
        # following code calculate the row number before removing invalid observations
        if self._work_obs.is_xray_amplitude_array():
            ind_false_sigma = np.argwhere(self._refl_data.sigF <= 0.).flatten()
        if self._work_obs.is_xray_intensity_array():
            ind_false_sigma = np.argwhere(self._refl_data.sigI <= 0.).flatten()
        # indices recoverd by calculating how many indices in ind_false_sigma are smaller than any given index in final_nemo_ind
        in_add = np.sum((self._final_nemo_ind[:, None] - ind_false_sigma[None, :]) >= 0, axis=1)
        self._original_row_ind = self._final_nemo_ind + in_add
        if return_idx is True:
            return self._original_row_ind

    def ft_and_tar(self):
        return np.arange(0, self._reso_select), np.isin(self._sorted_arg[:self._reso_select], self._final_nemo_ind).astype(int)

    def weak_by_signal_to_noise(self, level=6.):
        if self._work_obs.is_xray_amplitude_array():
            ind_weak = np.argwhere((self._refl_data.F / self._refl_data.sigF <= level)
                                   & (self._refl_data.resolution > 10.)).flatten()
        if self._work_obs.is_xray_intensity_array():
            ind_weak = np.argwhere((self._refl_data.I / self._refl_data.sigI <= level)
                                   & (self._refl_data.resolution > 10.)).flatten()
        return ind_weak

    def NEMO_removal(self, filename):
        assert self._original_row_ind is not None
        isel = flex.size_t(self._original_row_ind)
        self._refl_data._obj.delete_reflections(isel)
        self._refl_data._obj.write(filename)

    def write_filter_hkl(self, integrate_hkl_plain, hkl_array):
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
    # probability of normalised centric amplitude smaller than e, given read e and sigma sig.
    # READ Acta. Cryst. (1999). D55, 1759-1764
    return 1. - erfc(e / 1.4142)


def cumprob_ac_amplitude(e):
    # probability of normalised acentric amplitude smaller than e, given read e and sigma sig.
    # READ Acta. Cryst. (1999). D55, 1759-1764
    return 1 - np.exp(-e*e)


def cumprob_ac_intensity(e_square, sig):
    # probability of normalised acentric intensity smaller than e**2, given read e**2 and sigma sig.
    # READ Acta. Cryst. (2016). D72, 375-387
    return 0.5 * (erfc(-e_square / 1.4142 / sig) - np.exp((sig - 2 * e_square) / 2) * erfc((sig - e_square) / 1.4142 / sig))

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
    # probability of centric normalised intensity smaller than e**2, given read e**2 and sigma sig.
    # READ Acta. Cryst. (2016). D72, 375-387
    # based on low level c, faster
    c = ctypes.c_double(sig)
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