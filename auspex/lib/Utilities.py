import numpy as np
from scipy.linalg import svd

def construct_hl_table(obs):
    obs_ext = obs[None, :] * np.ones(obs.size)[:, None]
    # inv_res_sqr_ext = inv_res_sqr[None, :] * np.ones(inv_res_sqr.size)[:, None]
    obs_hl_table = delete_diag(obs_ext)
    # inv_res_sqr_ih_table = delete_diag(inv_res_sqr_ext)
    return obs_hl_table


def delete_diag(square_matrix):
    # inspired by https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
    # delete diagonal elements
    m = square_matrix.shape[0]
    s0, s1 = square_matrix.strides
    return np.lib.stride_tricks.as_strided(square_matrix.ravel()[1:], shape=(m-1, m), strides=(s0+s1, s1)).reshape(m, -1)


def cal_scaling_outlier(A, i_hl, sig_hl, SdRej=6.):
    U, s, Vh = svd(A)
    whl = 1. / (sig_hl * sig_hl)
    ghl = U[:, 0] * Vh[0, :]
    # i_sum_denominator = np.sum(whl * ghl * ghl)
    # i_sum_numerator = np.sum(whl * ghl * ihl)
    ihl_table = construct_hl_table(i_hl)
    whl_table = construct_hl_table(whl)
    ghl_table = construct_hl_table(ghl)
    i_sum_numerator_table = np.sum(whl_table * ghl_table * ihl_table, 1)
    i_sum_denominator_table = np.sum(whl_table * ghl_table * ghl_table, 1)
    i_avrg_others = i_sum_numerator_table / i_sum_denominator_table
    norm_dev = (i_hl - ghl * i_avrg_others) / np.sqrt(1. / whl + np.square(ghl * i_avrg_others))
    ind_filter = norm_dev > SdRej
    return ind_filter