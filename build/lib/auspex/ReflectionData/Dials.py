import msgpack
import json

import scitbx_array_family_flex_ext as flex
from cctbx.array_family import flex as af_flex
from cctbx import uctbx, crystal
from dxtbx.model import Crystal

from .ReflectionBase import *

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