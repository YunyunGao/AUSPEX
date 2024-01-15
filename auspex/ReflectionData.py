import math

from iotbx import mtz, cif, reflection_file_reader
from iotbx.xds import read_ascii
import iotbx.scalepack.merge as sca_merge
from cctbx import miller, uctbx, crystal, sgtbx
from cctbx.array_family import flex as af_flex
import scitbx_array_family_flex_ext as flex

import numpy as np
import itertools
from collections import namedtuple
import msgpack
import json
import copy

from iotbx.cif import CifParserError

import os

from dxtbx.model import Crystal


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

_obs_names = ['F', 'sig', 'I', 'F_ano', 'I_ano', 'sigF_ano', 'sigI_ano']

_merge_stats_names = ['res', 'r_pim', 'r_meas', 'r_merge', 'cc_sig_epsilon_squared', 'cc_x_i_bar']

_dials_strong = ['bbox', 'flags', 'id', 'intensity.sum.value', 'intensity.sum.variance', 'n_signal', 'panel', 'shoebox',
                 'xyzobs.px.value', 'xyzobs.px.variance']

_dials_indexed = ['bbox', 'entering', 'flags', 'id', 'imageset_id', 'intensity.sum.value', 'intensity.sum.variance',
                  'miller_index', 'n_signal', 'panel', 'rlp', 's1', 'shoebox',
                  'xyzcal.mm', 'xyzcal.px',
                  'xyzobs.mm.value', 'xyzobs.mm.variance', 'xyzobs.px.value', 'xyzobs.px.variance']

_dials_integrated_diamond = ['background.mean', 'background.sum.value', 'background.sum.variance', 'bbox', 'd',
                             'entering',
                             'flags', 'id',
                             'intensity.prf.value', 'intensity.prf.variance', 'intensity.sum.value',
                             'intensity.sum.variance',
                             'lp', 'miller_index',
                             'num_pixels.background', 'num_pixels.background_used', 'num_pixels.foreground',
                             'num_pixels.valid',
                             'panel', 'partial_id', 'partiality', 'profile.correlation', 'qe', 's1',
                             'xyzcal.mm', 'xyzcal.px', 'xyzobs.mm.value', 'xyzobs.mm.variance',
                             'xyzobs.px.value', 'xyzobs.px.variance', 'zeta']

_dials_integrated_aps = ['background.dispersion', 'background.mean', 'background.mse', 'background.sum.value',
                         'background.sum.variance', 'bbox', 'd', 'entering',
                         'flags', 'id', 'imageset_id',
                         'intensity.prf.value', 'intensity.prf.variance', 'intensity.sum.value', 'intensity.sum.variance',
                         'lp', 'miller_index', 'num_pixels.background', 'num_pixels.background_used', 'num_pixels.foreground',
                         'num_pixels.valid', 'panel', 'partial_id', 'partiality', 'profile.correlation', 's1',
                         'xyzcal.mm', 'xyzcal.px', 'xyzobs.mm.value', 'xyzobs.mm.variance',
                         'xyzobs.px.value', 'xyzobs.px.variance', 'zeta']

_dials_integrated_ssrl = ['background.mean', 'background.sum.value', 'background.sum.variance',
                          'bbox', 'd', 'entering',
                          'flags', 'id', 'imageset_id',
                          'intensity.prf.value', 'intensity.prf.variance', 'intensity.sum.value', 'intensity.sum.variance',
                          'lp', 'miller_index', 'num_pixels.background', 'num_pixels.background_used', 'num_pixels.foreground',
                          'num_pixels.valid', 'panel', 'partial_id', 'partiality', 'profile.correlation',
                          'qe', 's1',
                          'xyzcal.mm', 'xyzcal.px', 'xyzobs.mm.value', 'xyzobs.mm.variance',
                          'xyzobs.px.value', 'xyzobs.px.variance', 'zeta']

_dials_scaled = ['background.mean', 'background.sum.value', 'background.sum.variance', 'bbox', 'd', 'entering',
                 'flags', 'id',
                 'intensity.prf.value', 'intensity.prf.variance',
                 'intensity.scale.value', 'intensity.scale.variance',
                 'intensity.sum.value', 'intensity.sum.variance',
                 'inverse_scale_factor', 'inverse_scale_factor_variance',
                 'lp', 'miller_index',
                 'num_pixels.background', 'num_pixels.background_used', 'num_pixels.foreground', 'num_pixels.valid',
                 'original_index',
                 'panel', 'partial_id', 'partiality', 'phi', 'profile.correlation', 'qe', 's1',
                 'xyzcal.mm', 'xyzcal.px', 'xyzobs.mm.value', 'xyzobs.mm.variance',
                 'xyzobs.px.value', 'xyzobs.px.variance', 'zeta']


def filename_check(func):
    def check(obj):
        if obj._filename is None:
            raise ValueError('No file has been read.')
        return check

    return func


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
        return self._sigF

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

        :param observation_type: Column label.
        :return: Miller array corresponding to the given column label
        """
        try:
            ma = self._obj.as_miller_arrays()
        except Exception as e:
            print(e.message, e.args)
        ma_select = np.argwhere([observation_type in _.info().labels for _ in ma])[0][0]
        return ma[ma_select]


class MtzParser(ReflectionParser):
    """
    The Parser class to process mtz files.
    """

    def __init__(self):
        super(MtzParser, self).__init__()
        self._Fobs_refmac = None
        self._Fcalc_refmac = None

    def read(self, filename):
        """Read the given mtz file

        :param filename: File name or path to the file
        :return: None
        """
        if not os.path.exists(filename):
            raise FileNotFoundError('{0} does not exist'.format(filename))
        # use iotbx.mtz to read mtz file
        self._obj = mtz.object(file_name=filename)
        #self._space_group = self._obj.space_group()
        columns = [_ for _ in self._obj.columns()]
        cidx = self.sort_column_types(self._obj.column_types(), self._obj.column_labels())
        if self._batch_exits():
            self._read_batch()
        # read h, k, l
        # h = columns[cidx[0][0]].extract_values().as_numpy_array()
        # k = columns[cidx[0][1]].extract_values().as_numpy_array()
        # l = columns[cidx[0][2]].extract_values().as_numpy_array()
        # self._hkl = np.array([h, k, l]).T
        self._hkl = np.array(self._obj.extract_miller_indices())
        # one line
        # read structure amplitude
        if self.column_exits(cidx['F']):
            self._F = columns[cidx['F'][0]].extract_values().as_numpy_array()
            if cidx['F'][0] + 1 in cidx['sig']:  # read F standard deviation if exist
                self._sigF = columns[cidx['F'][0] + 1].extract_values().as_numpy_array()
        # read  F(+)/F(-)
        if self.column_exits(cidx['F_ano']):
            # Here assumes that F(+) column and F(-) column are adjacent
            pair_1 = columns[cidx['F_ano'][0]].extract_values().as_numpy_array()
            pair_2 = columns[cidx['F_ano'][1]].extract_values().as_numpy_array()
            self._F_ano = np.c_[pair_1, pair_2].flatten()
        # read standard deviation for  F(+)/F(-)
        if self.column_exits(cidx['sigF_ano']):
            pair_1 = columns[cidx['sigF_ano'][0]].extract_values().as_numpy_array()
            pair_2 = columns[cidx['sigF_ano'][1]].extract_values().as_numpy_array()
            self._sigF_ano = np.c_[pair_1, pair_2].flatten()
        # read intensity
        if self.column_exits(cidx['I']):
            self._I = columns[cidx['I'][0]].extract_values().as_numpy_array()
            if cidx['I'][0] + 1 in cidx['sig']:  # read I standard deviation if exist
                self._sigI = columns[cidx['I'][0] + 1].extract_values().as_numpy_array()
        # read I(+)/I(-)
        if self.column_exits(cidx['I_ano']):
            # Here assumes that I(+) column and I(-) column are adjacent
            pair_1 = columns[cidx['I_ano'][0]].extract_values().as_numpy_array()
            pair_2 = columns[cidx['I_ano'][1]].extract_values().as_numpy_array()
            self._I_ano = np.c_[pair_1, pair_2].flatten()
        # read standard deviation for I(+)/I(-)
        if self.column_exits(cidx['sigI_ano']):
            pair_1 = columns[cidx['sigI_ano'][0]].extract_values().as_numpy_array()
            pair_2 = columns[cidx['sigI_ano'][1]].extract_values().as_numpy_array()
            self._sigI_ano = np.c_[pair_1, pair_2].flatten()
        # read refmac sfcalc output
        if self.column_exits(cidx['Fobs_refmac']):
            self._Fobs_refmac = columns[cidx['Fobs_refmac'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['Fcalc_refmac']):
            self._Fcalc_refmac = columns[cidx['Fcalc_refmac'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['FP']):
            self._FP_refmac = columns[cidx['FP'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['FC']):
            self._FC_refmac = columns[cidx['FC'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['FC_ALL']):
            self._FC_ALL_refmac = columns[cidx['FC_ALL'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['FC_ALL_LS']):
            self._FC_ALL_LS_refmac = columns[cidx['FC_ALL_LS'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['FOM']):
            self._FOM_refmac = columns[cidx['FOM'][0]].extract_values().as_numpy_array()
        # read phenix.refinement output
        if self.column_exits(cidx['Fobs_phenix']):
            self._Fobs_phenix = columns[cidx['Fobs_phenix'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['Fcalc_phenix']):
            self._Fcalc_phenix = columns[cidx['Fcalc_phenix'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['Fobs_meta_phenix']):
            self._Fobs_meta_phenix = columns[cidx['Fobs_meta_phenix'][0]].extract_values().as_numpy_array()
        if self.column_exits(cidx['Fmodel_phenix']):
            self._Fmodel_phenix = columns[cidx['Fmodel_phenix'][0]].extract_values().as_numpy_array()
        # read resolution
        self._resolution = np.array(self._obj.crystals()[0].unit_cell().d(self._obj.extract_miller_indices()))
        self._filename = filename

    def _batch_exits(self):
        """
        :return: Check whether the batch data exists.
        :rtype: bool
        """
        if self._obj.batches().size() == 0:
            return False
        else:
            return True

    def _read_batch(self):
        # TODO:
        pass

    @staticmethod
    def column_exits(cidx):
        """
        :return: Check the existence of a column of the certain data type.
        :rtype: bool
        """
        if cidx.size != 0:
            return True
        else:
            return False

    @filename_check
    def get_column_types(self):
        """
        :return: a list of column types
        :rtype: list
        """
        return self._obj.column_types()

    @filename_check
    def get_column_list(self):
        """
        :return: a list of column labels
        :rtype: list
        """
        return self._obj.column_labels()

    @filename_check
    def get_space_group(self):
        """
        :return: space group
        :rtype: str
        """
        return str(self._obj.space_group().info())

    @filename_check
    def get_max_resolution(self):
        """
        :return: Maximum resolution
        :rtype: float
        """
        return self._obj.max_min_resolution()[1]

    @filename_check
    def get_min_resolution(self):
        """
        :return: minimum resolution
        :rtype: float
        """
        return self._obj.max_min_resolution()[0]

    @staticmethod
    def sort_column_types(column_types_list, column_labels_list):
        """
        :return: indices of columns containing corresponding data
        :rtype: dict
        """
        # standard columns
        column_types = np.array(column_types_list, dtype=str)
        column_labels = np.array(column_labels_list, dtype=str)
        miller_cidx = np.argwhere(column_types == 'H').flatten()
        F_cidx = np.argwhere(column_types == 'F').flatten()
        I_cidx = np.argwhere(column_types == 'J').flatten()
        sig_cidx = np.argwhere(column_types == 'Q').flatten()
        F_ano_cidx = np.argwhere(column_types == 'G').flatten()
        sigF_ano_cidx = np.argwhere(column_types == 'L').flatten()
        I_ano_cidx = np.argwhere(column_types == 'K').flatten()
        sigI_ano_cidx = np.argwhere(column_types == 'M').flatten()
        bg_cidx = np.argwhere((column_types == 'R') & (column_labels == 'BG')).flatten()
        sigbg_cidx = np.argwhere((column_types == 'R') & (column_labels == 'BG')).flatten()
        # non-standard columns
        # refmac
        fobs_refmac_cidx = np.argwhere(column_labels == 'Fobs').flatten()  # refmac sfcalc mode output
        fcalc_refmac_cidx = np.argwhere(column_labels == 'Fcalc').flatten()  # refmac sfcalc mode output
        fp_cidx = np.argwhere(column_labels == 'FP').flatten()  # refmac general output
        fc_cidx = np.argwhere(column_labels == 'FC').flatten()  # refmac general output
        fc_all_cidx = np.argwhere(column_labels == 'FC_ALL').flatten()  # refmac general output
        fc_all_ls_cidx = np.argwhere(column_labels == 'FC_ALL_LS').flatten()  # refmac general output
        fom_refmac_cidx = np.argwhere(column_labels == 'FOM').flatten()  # refmac general output
        fom_refmac_cidx = np.argwhere(column_labels == 'FOM').flatten()  # refmac general output
        # phenix.refinement
        fobs_meta_phenix_cidx = np.argwhere(column_labels == 'F-obs').flatten()
        fmodel_phenix_cidx = np.argwhere(column_labels == 'F-model').flatten() or \
                             np.argwhere(column_labels == 'F-model_xray').flatten()
        fobs_phenix_cidx = np.argwhere(column_labels == 'FOBS').flatten()
        fcalc_phenix_cidx = np.argwhere(column_labels == 'FCALC').flatten()
        cidx = {'indices': miller_cidx,
                'F': F_cidx,
                'sig': sig_cidx,
                'F_ano': F_ano_cidx,
                'sigF_ano': sigF_ano_cidx,
                'I': I_cidx,
                'I_ano': I_ano_cidx,
                'sigI_ano': sigI_ano_cidx,
                'bg': bg_cidx,
                'sigbg': sigbg_cidx,
                'Fobs_refmac': fobs_refmac_cidx,
                'Fcalc_refmac': fcalc_refmac_cidx,
                'FP': fp_cidx,
                'FC': fc_cidx,
                'FC_ALL': fc_all_cidx,
                'FC_ALL_LS': fc_all_ls_cidx,
                'FOM': fom_refmac_cidx,
                'Fobs_phenix': fobs_phenix_cidx,
                'Fcalc_phenix': fcalc_phenix_cidx,
                'Fobs_meta_phenix': fobs_meta_phenix_cidx,
                'Fmodel_phenix': fmodel_phenix_cidx
                }
        return cidx


class XdsParser(ReflectionParser):
    """The Parser class to process xds files.

    """

    def __init__(self):
        super(XdsParser, self).__init__()

    def read_hkl(self, filename=None, merge_equivalents=True):
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

    def unique_redundancies(self):
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

    def merge_stats_cmpt(self):
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

    def merge_stats_overall(self):
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
        from BinnedData import BinnedStatistics
        merge_stats = BinnedStatistics().const_stats(ires_minmax, num_data, i_mean, i_over_sigma_mean, completeness_mean, redundancy_mean,
                                                     r_pim, r_merge, r_meas, cc_half)
        return merge_stats

    def merge_stats_binned(self, num_of_bins=21):
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
        from BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_binned, num_data_binned, i_mean_binned, i_over_sigma_binned, completeness_binned,
                                                    redundancy_binned, r_pim_binned, r_merge_binned, r_meas_binned, cc_half_binned)
        return merg_stats

    def merge_stats_by_range(self, max_resolution, min_resolution):
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

        from BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_mean, num_data, i_mean, i_over_sigma, completeness,
                                                    redundancy, r_pim, r_merge, r_meas, cc_half)
        return merg_stats

    def cc_sig_y_square(self, num_of_bins=21):
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

    def merge_stats_binned_deprecated(self, iresbinwidth=0.01):
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

    def cal_completeness(self, unique_ires_array, d_min=None, d_max=None):
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
    def get_space_group(self):
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
    def get_max_resolution(self):
        """
        :return: maximum resolution
        :rtype: float
        """
        return self._obj.miller_set().resolution_range()[1]

    @filename_check
    def get_min_resolution(self):
        """
        :return: minimum resolution
        :rtype: float
        """
        return self._obj.miller_set().resolution_range()[0]

    @filename_check
    def get_merged_I(self):
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._I_merged

    @filename_check
    def get_merged_hkl(self):
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._hkl_merged

    @filename_check
    def get_merged_sig(self):
        """
        :return: merged intensity array
        :rtype: 1d ndarray
        """
        return self._sigI_merged

    @filename_check
    def get_merged_resolution(self):
        """
        :return: resolution array of merged intensity
        :rtype: 1d ndarray
        """
        return self._resolution_merged

    @filename_check
    def get_zd(self):
        """
        :return: reflection position array on z-axis
        :rtype: 1d ndarray
        """
        if self._obj.unmerged_data:
            return self._obj.zd.as_numpy_array()
        else:
            return None


class CifParser(ReflectionParser):
    """The Parser class to process cif files.

    """

    def __init__(self):
        super(CifParser, self).__init__()
        self._resolutionF = None
        self._resolutionI = None
        self._resolutionI_ano = None

    def read(self, filename=None):
        """Read the given cif file.

        :param filename: File or path to file
        :type filename: str
        :return: None
        """
        self._obj = cif.reader(file_path=filename)
        miller_arrays = self._obj.build_miller_arrays()
        for model in miller_arrays.keys():
            for key in miller_arrays[model].keys():
                wavelength_id = np.array(miller_arrays[model]['_refln.wavelength_id'].data())[0]
                if 'F_meas' in key:
                    self._hkl = np.array(miller_arrays[model][key].indices(), dtype=int)
                    self._F = np.array(miller_arrays[model][key].data(), dtype=float)
                    self._sigF = np.array(miller_arrays[model][key].sigmas(), dtype=float)
                    self._resolutionF = np.array(miller_arrays[model][key].
                                                 unit_cell().d(miller_arrays[model][key].indices()), dtype=float)
                    self._space_group = miller_arrays[model][key].space_group()
                    self._unit_cell = miller_arrays[model][key].unit_cell()
                if 'intensity_meas' in key:
                    if np.array(miller_arrays[model]['_refln.wavelength_id'].data())[0] == wavelength_id:
                        if miller_arrays[model][key].anomalous_flag():
                            self._I_ano = np.array(miller_arrays[model][key].data(), dtype=float)
                            self._sigI_ano = np.array(miller_arrays[model][key].sigmas(), dtype=float)
                            self._resolutionI_ano = np.array(miller_arrays[model][key].
                                                             unit_cell().d(miller_arrays[model][key].indices()),
                                                             dtype=float)
                        else:
                            self._I = np.array(miller_arrays[model][key].data(), dtype=float)
                            self._sigI = np.array(miller_arrays[model][key].sigmas(), dtype=float)
                            self._resolutionI = np.array(miller_arrays[model][key].
                                                         unit_cell().d(miller_arrays[model][key].indices()),
                                                         dtype=float)
            self._filename = filename

    @filename_check
    def get_space_group(self):
        """
        :return: Space group text.
        :rtype: str
        """
        return str(self._space_group.info())

    @filename_check
    def get_cell_dimension(self):
        """
        :return: Unit cell parameters (a, b, c, alpha, beta, gamma).
        :rtype: list
        """
        return self._unit_cell.parameters()


class ScaParser(ReflectionParser):
    """The Parser class to process sca files.

    """

    def __init__(self):
        super(ScaParser, self).__init__()

    def read(self, filename):
        """Read the given cif file.

        :param filename: File or path to file
        :type filename: str
        :return: None
        """
        try:
            with open(filename) as f:
                self._obj = sca_merge.reader(f)
            if self._obj.anomalous:
                self._I_ano = np.array(self._obj.iobs, dtype=float)
                self._sigI_ano = np.array(self._obj.sigmas, dtype=float)
                self._hkl = np.array(self._obj.miller_indices, dtype=int)
                self._resolutionI_ano = np.array(self._obj.unit_cell.d(self._obj.miller_indices), dtype=float)
            else:
                self._I = np.array(self._obj.iobs, dtype=float)
                self._sigI = np.array(self._obj.sigmas, dtype=float)
                self._hkl = np.array(self._obj.miller_indices, dtype=int)
                self._resolutionI = np.array(self._obj.unit_cell.d(self._obj.miller_indices), dtype=float)
        except sca_merge.FormatError:
            try:
                import iotbx.scalepack.no_merge_original_index as sca_unmerge
                sca_unmerge.reader(filename)
                print('Lacking the unit cell parameters. Cannot load unmerged intensities.')
            except AssertionError:
                print('Not a readable scalepack file.')
        self._filename = filename


class ShlexParser(ReflectionParser):
    """

    """
    def __init__(self):
        super(ShlexParser, self).__init__()
        self.miller_set = None
        self.crystal_symmetry = None

    def read(self, filename, unit_cell, space_group_number):
        try:
            self._obj = reflection_file_reader.any_reflection_file(filename+'=intensities')
        except:
            pass
        self.crystal_symmetry = crystal.symmetry().customized_copy(
            uctbx.unit_cell(unit_cell),
            sgtbx.space_group(sgtbx.space_group_symbols(space_group_number)).info()
        )
        self._space_group = self.crystal_symmetry.space_group()
        self.miller_set = self._obj.as_miller_arrays(self.crystal_symmetry, merge_equivalents=False)[0].map_to_asu()
        self._hkl = np.array(self.miller_set.indices())
        self._I = np.array(self.miller_set.data())
        self._sigI = np.array(self.miller_set.sigmas())
        self._resolutionI = np.array(self.miller_set.unit_cell().d(self.miller_set.indices()))
        self._resolution = self._resolutionI
        self._filename = filename
        self._merge()

    def _merge(self):
        """Record the merged data.

        :return: None
        """
        # merged_miller = self._obj.as_miller_arrays(self.crystal_symmetry, merge_equivalents=True)[0] won't merge
        assert self.miller_set.is_unmerged_intensity_array()
        merged_miller = self.miller_set.merge_equivalents().array()
        self._I_merged = np.array(merged_miller.data())
        self._hkl_merged = np.array(merged_miller.indices())
        self._sigI_merged = np.array(merged_miller.sigmas())
        self._resolution_merged = np.array(merged_miller.unit_cell().d(merged_miller.indices()))
        self._multiplicity_merged = merged_miller.multiplicities().data().as_numpy_array()
        self._complete_set = merged_miller.complete_set()

    def unique_redundancies(self):
        """Get redundancy of each reflection in merged data.

        :return: array of redundancy
        :rtype: 1d ndarray
        """
        redund = miller.merge_equivalents(self.miller_set).redundancies().data().as_numpy_array()
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
        redund = miller.merge_equivalents(self.miller_set).redundancies().data().as_numpy_array()


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

    def merge_stats_cmpt(self):
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

    def merge_stats_overall(self):
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
        from BinnedData import BinnedStatistics
        merge_stats = BinnedStatistics().const_stats(ires_minmax, num_data, i_mean, i_over_sigma_mean, completeness_mean, redundancy_mean,
                                                     r_pim, r_merge, r_meas, cc_half)
        return merge_stats

    def merge_stats_binned(self, num_of_bins=21):
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
        from BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_binned, num_data_binned, i_mean_binned, i_over_sigma_binned, completeness_binned,
                                                    redundancy_binned, r_pim_binned, r_merge_binned, r_meas_binned, cc_half_binned)
        return merg_stats

    def merge_stats_by_range(self, max_resolution, min_resolution):
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

        from BinnedData import BinnedStatistics
        merg_stats = BinnedStatistics().const_stats(ires_mean, num_data, i_mean, i_over_sigma, completeness,
                                                    redundancy, r_pim, r_merge, r_meas, cc_half)
        return merg_stats

    def cc_sig_y_square(self, num_of_bins=21):
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

    def merge_stats_binned_deprecated(self, iresbinwidth=0.01):
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

    def cal_completeness(self, unique_ires_array, d_min=None, d_max=None):
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

class DialsParser(ReflectionParser):
    """The Parser class to process dials files.

    """
    flex_func = {'int': flex.int_from_byte_str, 'double': flex.double_from_byte_str}

    def __init__(self):
        super(DialsParser, self).__init__()
        self._filename = None
        self._expt = None
        self._nrows = None
        self._type_reflection_table = None
        self._identifiers = None
        self._id = None
        self._id_bool = []
        self._xyzcal_mm = None
        self._xyzcal_px = None
        self._xyzobs_px = None
        self._xyzobs_var_px = None
        self._crystals = []

    def smart_read(self, filename):
        """Read dials spots files.

        :param filename: File or path to spots file
        :type filename: str
        :return: None
        """
        self._filename = filename
        with open(self._filename, 'rb') as f:
            self._obj = msgpack.unpack(f, raw=False, strict_map_key=False)
        assert self._obj[0] == 'dials::af::reflection_table', 'Not a dials reflection table'
        self._nrows = int(self._obj[2]['nrows'])
        self._identifiers = self._obj[2]['identifiers']

        data_dict = list(self._obj[2]['data'].keys())
        if data_dict == _dials_strong:
            self._type_reflection_table = 'spots'
        elif data_dict == _dials_indexed:
            self._type_reflection_table = 'indexed'
        elif (data_dict == _dials_integrated_aps) or \
                (data_dict == _dials_integrated_diamond) or \
                (data_dict == _dials_integrated_ssrl):
            self._type_reflection_table = 'integrated'
        elif data_dict == _dials_scaled:
            self._type_reflection_table = 'scaled'
        else:
            raise AssertionError('Not a standard DIALS data file.')
        self.read_columns()

    def read_columns(self):
        """Read and record data by keys.

        """
        # experimental data
        self._I_sum = self.column_to_array('intensity.sum.value', 'double', False)
        self._I_sum_var = self.column_to_array('intensity.sum.variance', 'double', False)
        self._xyzobs_px = self.column_to_array('xyzobs.px.value', 'double', True)
        self._xyzobs_px_var = self.column_to_array('xyzobs.px.variance', 'double', True)
        self._id = self.column_to_array('id', 'int', False)

        for identifier in self._identifiers.keys():
            self._id_bool.append(self._id == identifier)

        # indexed data
        if self._type_reflection_table in ('indexed', 'integrated', 'scaled'):
            self._hkl = self.column_to_array('miller_index', 'int', True)
            self._xyzcal_mm = self.column_to_array('xyzcal.mm', 'double', True)
            self._xyzcal_px = self.column_to_array('xyzcal.px', 'double', True)
            self._xyzobs_mm = self.column_to_array('xyzobs.mm.value', 'double', True)
            self._xyzobs_mm_var = self.column_to_array('xyzobs.mm.variance', 'double', True)

        # integrated data
        if self._type_reflection_table in ('integrated', 'scaled'):
            self._I_prf = self.column_to_array('intensity.prf.value', 'double', False)
            self._I_prf_var = self.column_to_array('intensity.prf.variance', 'double', False)
            self._background = self.column_to_array('background.mean', 'double', False)
            self._background_sum_var = self.column_to_array('background.sum.value', 'double', False)
            self._background_sum_var = self.column_to_array('background.sum.variance', 'double', False)
            self._backgroud_pixels = self.column_to_array('num_pixels.background_used', 'int', False)
            self._foreground_pixels = self.column_to_array('num_pixels.foreground', 'int', False)

        # scaled data
        if self._type_reflection_table == 'scaled':
            self._I_sca = self.column_to_array('intensity.scale.value', 'double', False)
            self._I_var_sca = self.column_to_array('intensity.scale.variance', 'double', False)
            self._inv_sca_factor = self.column_to_array('inverse_scale_factor', 'double', False)
            self._inv_sca_factor_var = self.column_to_array('inverse_scale_factor_variance', 'double', False)

    def read_expt(self, filename):
        """Read expt using dxtbx crystal model.

        :param filename: File or path to expt file
        """
        self._expt = 'filename'
        with open(filename, 'r') as inline:
            dict_crystals = json.loads(inline.read())['crystal']
        for d in dict_crystals:
            real_space_a = d['real_space_a']
            real_space_b = d['real_space_b']
            real_space_c = d['real_space_c']
            space_group = str('Hall:{0}'.format(d['space_group_hall_symbol']))
            xl = Crystal(real_space_a, real_space_b, real_space_c, space_group_symbol=space_group)

            recalculated_unit_cell = d.get("recalculated_unit_cell")
            if recalculated_unit_cell is not None:
                xl.set_recalculated_unit_cell(uctbx.unit_cell(recalculated_unit_cell))
            self._crystals.append(xl)

    def cal_resolution(self):
        """Get resolutions.

        :return: resoltuion array
        :rtype: 1d ndarray
        """
        if not self._expt:
            raise RuntimeError('No experiment list file. Please load corresponding expt.')
        if len(self._crystals) != len(self._identifiers):
            raise RuntimeError('Mismatched data file and experiment list file.')
        if self._resolution is None:
            self._resolution = np.zeros(self._nrows)
            for i, k in zip(self._id_bool, self._identifiers.keys()):
                self._resolution[i] = self._crystals[k].get_unit_cell().d(af_flex.miller_index(self._hkl[i].tolist()))
        return self._resolution

    def as_miller_array(self, identifier_key, intensity='sum'):
        """Convert dials metadata to cctbx miller array.

        :param identifier_key: dials experiment identifier
        :type identifier_key: str
        :param intensity: can be 'sum' or 'prf', default 'sum'
        :type intensity: str
        :return: A cctbx miller array
        :rtype: cctbx.miller_array
        """
        if intensity == 'sum':
            intensities, variances = \
                self._I_sum[self._id_bool[identifier_key]], self._I_sum_var[self._id_bool[identifier_key]]
        elif intensity == 'prf':
            intensities, variances = \
                self._I_prf[self._id_bool[identifier_key]], self._I_prf_var[self._id_bool[identifier_key]]
        crystal_symmetry = crystal.symmetry().customized_copy(self._crystals[identifier_key].get_unit_cell(),
                                                              self._crystals[identifier_key].get_space_group().info())
        miller_set = miller.set(crystal_symmetry=crystal_symmetry,
                                indices=af_flex.miller_index(self._hkl[self._id_bool[identifier_key]]),
                                anomalous_flag=False)
        i_obs = miller.array(miller_set, data=af_flex.double(intensities))
        i_obs.set_observation_type_xray_intensity()
        i_obs.set_sigmas(af_flex.sqrt(flex.double(variances)))
        i_obs.set_info(miller.array_info(source='DIALS', source_type='reflection_tables'))
        return i_obs

    def column_to_array(self, dict_key, d_type, reshape=False):
        """An universal data dict reader.

        :param dict_key: key used by dials
        :type dict_key: str
        :param d_type: data type of the chosen column
        :type d_type: type
        """
        array = self.flex_func[d_type](self._obj[2]['data'][dict_key][1][1])
        array = array.as_numpy_array()
        if reshape:
            return array.reshape(self._nrows, 3)
        else:
            return array

    @property
    def data_type(self):
        """
        :return: type of reflection table
        :rtype: str
        """
        return self._type_reflection_table

    def get_zd(self):
        """
        :return: positions of observations on z-axis
        :rtype: 1d ndarray
        """
        return self._xyzobs_px[:, 2]

    def get_background(self):
        """
        :return: background
        :rtype: 1d ndarray
        """
        return self._background

    def get_background_var(self):
        """
        :return: variance of background
        :rtype: 1d ndarray
        """
        return self._background_sum_var


def unique_redundancies(miller_array):
    # get redundancy of each reflection in merged data
    if not miller_array.is_unmerged_intensity_array():
        raise ValueError('is not an unmerged intensity array'.format())
    redund = miller.merge_equivalents(
        miller_array.map_to_asu()).redundancies().data().as_numpy_array()
    return np.unique(redund)


def FileReader(file_name, file_type=None, *args):
    if file_name[-3:] == 'mtz' or (file_type in ('xds', 'mtz', 'MTZ', 'mrg', 'binary')):
        try:
            reflection_data = MtzParser()
            reflection_data.read(file_name)
            reflection_data.source_data_format = 'mtz'
        except AssertionError:
            print('Failed to read the mtz file. Check the data format or specify the input type using --input-type.')
    elif file_name[-3:] == 'HKL' or (file_type in ('xds', 'HKL', 'xds_HKL')):
        try:
            reflection_data = XdsParser()
            reflection_data.read_hkl(file_name, merge_equivalents=False)
            reflection_data.source_data_format = 'xds_hkl'
        except AssertionError:
            print('Failed to read the XDS HKL file. Check the data format or specify the input type using --input-type.')
    elif file_name[-3:] == 'cif' or (file_type in ['cif', 'mmcif', 'CIF']):
        try:
            reflection_data = CifParser()
            reflection_data.read(file_name)
            reflection_data.source_data_format = 'cif'
        except CifParserError or AssertionError:
            print('Failed to read the cif file. Check the data format or specify the input type using --input-type.')
    elif file_name[-4:] == 'refl' or (file_type in ['refl', 'dials', 'msgpack']):
        try:
            reflection_data = DialsParser()
            reflection_data.smart_read(file_name)
            reflection_data.source_data_format = 'refl'
        except AssertionError:
            print('Failed to read the DIALS file. Check the data format or specify the input type using --input-type')
    elif file_name[-3:] == 'hkl' or (file_type in ['shlex', 'hkl', 'shlex_hkl']):
        try:
            if len(args) == 2:
                unit_cell, space_group_number = args
            reflection_data = ShlexParser()
            reflection_data.read(file_name, unit_cell, space_group_number)
            reflection_data.source_data_format = 'shlex_hkl'
        except RuntimeError as err:
            print('An error occurred when parsing shlex hkl: ', err)
    return reflection_data


def group_by_redundancies(miller_array, hkl, observations, resolution):
    """

    :return: A list of containers.
        indices_container: list
        obs_container: list
        resolution_container: list
    """
    if not miller_array.is_unmerged_intensity_array():
        raise ValueError('is not an unmerged intensity array'.format())

    # get redundancy of each reflection in merged data
    merged = miller.merge_equivalents(miller_array)
    redund = merged.redundancies().data().as_numpy_array()

    # get multiplicity of each reflection in merged data
    multi = merged.array().multiplicities().data().as_numpy_array()

    # get unique redundancies
    uni_redund = np.unique(redund)

    # creat containers for indices, obs and resolution grouped by the number of redundancies
    indices_container = [list() for _ in range(uni_redund.size)]
    obs_container = [list() for _ in range(uni_redund.size)]
    resolution_container = [list() for _ in range(uni_redund.size)]

    # shrinkable shallow copy for unmerged indices, obs and resolution
    tmp = hkl.astype(int)
    tmp_obs = observations
    tmp_resol = resolution

    for idx, redund_num in enumerate(uni_redund):  # loop through unique redundancy
        args_redund = np.where(redund == redund_num)[0]
        multi_of_args_redund = multi[args_redund]
        # separate the args_redund by the multiplicities of corresponding reflections
        args_redund_separated = [args_redund[multi_of_args_redund == uni] for uni in
                                 np.unique(multi_of_args_redund)]
        for args in args_redund_separated:  # loop through args_redund separated by multiplicity
            # create iterator for merged data with specific multiplicity and redundancy
            it = np.array(merged.array().indices(), int)[args]
            hkl_view = tmp.view([('a', int), ('b', int), ('c', int)])  # 1d view of Nx3 matrix

            set_by_multiplicity = list()
            for hkl_index in it:
                sym_operator = miller.sym_equiv_indices(miller_array.space_group(), hkl_index.tolist())
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
            # shrink reflections
            tmp = copy.deepcopy(tmp[~logic_or])
            tmp_obs = tmp_obs[~logic_or]
            tmp_resol = tmp_resol[~logic_or]
        indices_container[idx] = np.concatenate(indices_container[idx]).reshape(args_redund.size, redund_num, 3)
        obs_container[idx] = np.concatenate(obs_container[idx]).reshape(args_redund.size, redund_num)
        resolution_container[idx] = np.concatenate(resolution_container[idx]).reshape(args_redund.size, redund_num)[
                                    :, 0]
    return indices_container, obs_container, resolution_container


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


def _get_bins_by_binwidth(ires, bin_width):
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


def _get_binned_by_multiplicity(quantity_array, sorted_args_by_ires, total_bin_number):
    # sorted_args = [np.argsort(_) for _ in quantity_array]
    # find the index of multiplicity with the maximum observation
    idx_multi_max_obs = np.argmax([_.size for _ in sorted_args_by_ires])
    #ind_list = [_binning_idx(_.size, total_bin_number) for _ in quantity_array]
    ind_list = _binning_idx(quantity_array[idx_multi_max_obs].size, total_bin_number)
    binned_quantity_by_multiplicity = []
    for i, arg, ind in zip(quantity_array, sorted_args_by_ires, ind_list):
        binned_quantity_by_multiplicity.append([i[arg][lower: upper] for lower, upper in ind])
    return binned_quantity_by_multiplicity


def _binning_idx_even(array_size, num_of_bins):
    max_per_bin = array_size // num_of_bins
    min_per_bin = array_size // (num_of_bins+1)
    step_redund_per_bin = np.floor(min_per_bin + (max_per_bin - min_per_bin) * np.random.rand(num_of_bins))
    compensation_per_bin = (array_size - step_redund_per_bin.sum()) // num_of_bins
    step_redund_per_bin += compensation_per_bin
    upper_ind_list = np.cumsum(step_redund_per_bin)
    upper_ind_list[-1] = array_size - 1
    lower_ind_list = np.insert(upper_ind_list[:-1], 0, 0)
    ind_array = np.array((lower_ind_list, upper_ind_list), dtype=int).transpose()
    return ind_array


def _binning_idx_xprep(array_size, num_of_bins=21):
    max_per_bin = array_size // (num_of_bins-1)
    min_per_bin = array_size // num_of_bins
    step_redund_per_bin = np.floor(min_per_bin + (max_per_bin - min_per_bin) * np.random.rand(num_of_bins - 1))
    compensation_per_bin = (array_size - step_redund_per_bin.sum()) // (num_of_bins - 1)
    step_redund_per_bin += compensation_per_bin
    upper_ind_list = np.cumsum(step_redund_per_bin)
    division_20 = upper_ind_list[-2] + math.floor((array_size - upper_ind_list[-2]) * 0.7)
    upper_ind_list[-1] = division_20
    upper_ind_list = np.append(upper_ind_list, array_size - 1)
    lower_ind_list = np.insert(upper_ind_list[:-1], 0, 0)
    ind_array = np.array((lower_ind_list, upper_ind_list), dtype=int).transpose()
    return ind_array


def _get_args_binned(ires_by_multiplicity, num_of_bins, method='xprep'):
    """
    :param ires_by_multiplicity: a list of resolution groupped by multiplicity
    :type ires_by_multiplicity: list
    :param num_of_bins: total number of bins
    :type num_of_bins: int
    :param method: 'even' or 'xprep'. Default: xprep
    :type method: str
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


def _get_args_by_range(ires_by_multiplicity, max_resolution, min_resolution):
    ires_unique = np.concatenate(ires_by_multiplicity)
    args_in_range = (ires_unique >= max_resolution) & (ires_unique <= min_resolution)
    return args_in_range







