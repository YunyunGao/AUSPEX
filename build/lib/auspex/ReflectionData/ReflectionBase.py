import numpy as np
from collections import namedtuple

from cctbx import miller

_obs_names = ['F', 'sig', 'I', 'F_ano', 'I_ano', 'sigF_ano', 'sigI_ano']

column_type_as_miller_array_type_hints = {
    "H": "miller_index",
    "J": "intensity",
    "F": "amplitude",
    "D": "anomalous_difference",
    "Q": "standard_deviation",
    "G": "amplitude",
    "L": "standard_deviation",
    "K": "intensity",
    "M": "standard_deviation",
    "E": "normalized_amplitude",
    "P": "phase_degrees",
    "W": "weight",
    "A": "hendrickson_lattman",
    "B": "batch_number",
    "Y": "m_isym",
    "I": "integer",
    "R": "real",
}

_merge_stats_names = ['res', 'r_pim', 'r_meas', 'r_merge', 'cc_sig_epsilon_squared', 'cc_x_i_bar']

class Observation(object):
    """
    A conceptual class representation of an observation array.

    :param obs: observation array
    :type obs: 1d ndarray
    :param sigma: deviation array
    :type sigma: 1d ndarray
    :param ires: resolution array
    :type ires: 1d ndarray
    """

    def __init__(self, obs, sigma, ires):
        self._obs = obs
        self._sigma = sigma
        self._ires = ires
        self.omit_invalid_sigmas()

    def omit_invalid_sigmas(self):
        """
        Function to omit observations with negative sigmas.
        """
        valid_sigmas_idx = self._sigma > 0.
        self._obs = self._obs[valid_sigmas_idx]
        self._sigma = self._sigma[valid_sigmas_idx]
        self._ires = self._ires[valid_sigmas_idx]

    @property
    def obs(self):
        """
        :return: observations
        :rtype: 1d ndarray
        """
        return self._obs

    @property
    def sigma(self):
        """
        :return: deviations
        :rtype: 1d ndarray
        """
        return self._sigma

    @property
    def ires(self):
        """
        :return: resolutions
        :rtype: 1d ndarray
        """
        return self._ires

    def size(self):
        """
        :return: number of observations
        :rtype: int
        """
        if self._obs.size == self._sigma.size and self._obs.size == self._ires.size:
            return self._obs.size

    def invresolsq(self):
        """
        :return: inverse resolution squared
        :rtype: 1d ndarray
        """
        return 1. / (self.ires * self.ires)


class ReflectionParser(object):
    """
    A conceptual class for reflections. It is the basic class for the actual file parser.
    """

    def __init__(self):
        """
        constructor method
        """
        self._filename = None
        self._source_data_format = None
        self._obj = None
        self._hkl = None
        self._F = None
        self._sigF = None
        self._I = None
        self._sigI = None
        self._background = None
        self._F_ano = None
        self._I_ano = None
        self._sigF_ano = None
        self._sigI_ano = None
        self._resolution = None
        self._resolutionI = None
        self._resolutionI_ano = None
        self._resolutionF = None
        self._resolutionF_ano = None
        self._space_group = None

    @property
    def file_name(self):
        """
        :return: file or path to file
        :rtype: str
        """
        return self._filename

    @property
    def size(self):
        return self._resolution.size

    @property
    def hkl(self):
        """
        :return: hkl indices
        :rtype: nx3 ndarray
        """
        return self._hkl

    @property
    def resolution(self):
        """
        :return: resolutions
        :rtype: 1d ndarray
        """
        if any(res is not None
               for res in [self._resolutionI, self._resolutionF, self._resolutionI_ano, self._resolutionF_ano]):
            raise RuntimeError("Multiple data arrays exist. Call observation wrapper instead.")
        else:
            return self._resolution

    @property
    def F(self):
        """
        :return: amplitude
        :rtype: 1d ndarray
        """
        return self._F

    @property
    def sigF(self):
        """
        :return: standard deviation of amplitude
        :rtype: 1d ndarray
        """
        return self._sigF

    @property
    def I(self):
        """
        :return: intensity
        :rtype: 1d ndarray
        """
        return self._I

    @property
    def sigI(self):
        """
        :return: standard deviation of intensity
        :rtype: 1d ndarray
        """
        return self._sigI

    @property
    def background(self):
        """
        :return: reflection background
        :rtype: 1d ndarray
        """
        return self._background

    @property
    def F_ano(self):
        """
        :return: anomalous amplitude
        :rtype: 1d ndarray
        """
        return self.F_ano

    @property
    def sigF_ano(self):
        """
        :return: standard deviation of anomalous amplitude
        :rtype: 1d ndarray
        """
        return self.sigF_ano

    @property
    def sigI_ano(self):
        """
        :return: standard deviation of anomalous intensity
        :rtype: 1d ndarray
        """
        return self.sigI_ano

    @property
    def source_data_format(self):
        return self._source_data_format

    @source_data_format.setter
    def source_data_format(self, _):
        self._source_data_format = _

    def observation(self, idx):
        """Return all the not None observations at the given index, as a namedtuple.

        :return ObsTuple with fields of not None observations.
        :rtype: namedtuple
        """
        assert isinstance(idx, (list, np.ndarray)), 'invalid bin number(s)'
        obs_list = [self._F, self._sigF, self._I, self._sigI, self._F_ano, self._I_ano, self._sigF_ano, self._sigI_ano]
        obs_iter = zip(obs_list, _obs_names)
        obs_dict = {}
        for value, name in obs_iter:
            if value is not None:
                obs_dict[name] = value[idx]
        # convert dict to namedtuple. make sure it is unmutable
        ObsTuple = namedtuple('ObsTuple', sorted(obs_dict))
        obs_at_idx = ObsTuple(**obs_dict)
        return obs_at_idx

    def get_amplitude_data(self):
        """Return the wrapped amplitude data

        :return: Observation instance
        """
        if self._F is None:
            return None
        valid_args = self._F.nonzero()
        obs = self._F[valid_args]
        sigma = self._sigF[valid_args]
        try:
            ires = self._resolutionF[valid_args]
        except (AttributeError, TypeError):
            ires = self._resolution[valid_args]
        return Observation(obs=obs, sigma=sigma, ires=ires)

    def get_intensity_data(self):
        """Return the wrapped intensity data

        :return: Observation instance
        """
        if self._I is None:
            return None
        valid_args = self._I.nonzero()
        obs = self._I[valid_args]
        sigma = self._sigI[valid_args]
        try:
            ires = self._resolutionI[valid_args]
        except (AttributeError, TypeError):
            ires = self._resolution[valid_args]
        return Observation(obs=obs, sigma=sigma, ires=ires)

    def get_amplitude_anom_data(self):
        """Return the wrapped anomalous amplitude data

        :return: Observation instance
        """
        if self._F_ano is None:
            return None
        valid_args = self._F_ano.nonzero()
        obs = self._F_ano[valid_args]
        sigma = self._sigF_ano[valid_args]
        try:
            ires = self._resolutionF_ano[valid_args]
        except (AttributeError, TypeError):
            ires = np.c_[self._resolution, self._resolution].flatten()[valid_args]
        return Observation(obs=obs, sigma=sigma, ires=ires)

    def get_intensity_anom_data(self):
        """Return the wrapped anomalous intensity data.

        :return: Observation instance
        """
        if self._I_ano is None:
            return None
        valid_args = self._I_ano.nonzero()
        obs = self._I_ano[valid_args]
        sigma = self._sigI_ano[valid_args]
        try:
            ires = self._resolutionI_ano[valid_args]
        except (AttributeError, TypeError):
            ires = np.c_[self._resolution, self._resolution].flatten()[valid_args]
        return Observation(obs=obs, sigma=sigma, ires=ires)

    def get_equiv_index(self, hkl):
        """Return the equivalent indices of given hkl based on Laue class

        :param hkl: Miller index.
        :return: A list of symmetry-equivalent reflections
        :rtype: list
        """
        sym_operator = miller.sym_equiv_indices(self._space_group, list(hkl))
        equiv_indices = [_.h() for _ in sym_operator.indices()]
        return equiv_indices

    def get_miller_array(self, observation_type):
        """

        :param observation_type: Can be either 'FP' or 'I'.
        :return: Miller array corresponding to the given column label
        """
        try:
            ma = self._obj.as_miller_arrays()
        except Exception as e:
            print(e.message, e.args)
        if observation_type == 'FP':
            if self.source_data_format == 'mtz':
                for ma_type in ['FP', 'F', 'FMEANS']:
                    try:
                        ma_select = np.argwhere([ma_type in _.info().labels for _ in ma])[0][0]
                        return_ma = ma[ma_select]
                    except IndexError:
                        continue
            elif self.source_data_format == 'cif':
                miller_arrays = self._obj.build_miller_arrays()
                for model in miller_arrays.keys():
                    for key in miller_arrays[model].keys():
                        if 'F_meas' in key:
                            return_ma = miller_arrays[model][key]
        if observation_type == 'I':
            if self.source_data_format == 'mtz':
                for ma_type in ['I', 'IMEANS', 'IMEAN']:
                    try:
                        ma_select = np.argwhere([ma_type in _.info().labels for _ in ma])[0][0]
                        return_ma = ma[ma_select]
                    except IndexError:
                        continue
            elif self.source_data_format == 'cif':
                miller_arrays = self._obj.build_miller_arrays()
                wavelength_id = np.array(miller_arrays[model]['_refln.wavelength_id'].data())[0]
                for model in miller_arrays.keys():
                    for key in miller_arrays[model].keys():
                        if ('intensity_meas' in key) and ('wavelength_id='+wavelength_id in key):
                            if miller_arrays[model][key].anomalous_flag():
                                continue
                            else:
                                return_ma = miller_arrays[model][key]
        try:
            return_ma
        except NameError:
            raise ValueError('Non-standard colum label')
        return return_ma


def namedtuplify(keys, values):
    obs_iter = zip(values, keys)
    obs_dict = {}
    for value, name in obs_iter:
        if value is not None:
            obs_dict[name] = value
    # convert dict to namedtuple. make sure it is unmutable
    ObsTuple = namedtuple('ObsTuple', sorted(obs_dict))
    obs = ObsTuple(**obs_dict)
    return obs


def filename_check(func):
    def check(obj):
        if obj._filename is None:
            raise ValueError('No file has been read.')
        return check

    return func


def unique_redundancies(miller_array):
    # get redundancy of each reflection in merged data
    if not miller_array.is_unmerged_intensity_array():
        raise ValueError('is not an unmerged intensity array'.format())
    redund = miller.merge_equivalents(
        miller_array.map_to_asu()).redundancies().data().as_numpy_array()
    return np.unique(redund)