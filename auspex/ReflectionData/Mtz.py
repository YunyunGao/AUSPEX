import os
import copy
import numpy as np
from iotbx import mtz

from .ReflectionBase import *


class MtzParser(ReflectionParser):
    """
    The Parser class to process mtz files.
    """

    def __init__(self):
        super(MtzParser, self).__init__()
        # refmac columns
        self._Fobs_refmac = None
        self._Fcalc_refmac = None
        self._FP_refmac = None
        self._FC_refmac = None
        self._FC_ALL_refmac = None
        self._FC_ALL_LS_refmac = None
        self._FOM_refmac = None
        # phenix columns
        self._Fobs_phenix = None
        self._Fcalc_phenix = None
        self._Fobs_meta_phenix = None
        self._Fmodel_phenix = None
        # ESS mtz
        self._lam = None

    def read(self, filename: str = None):
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
        # l = columns[cidx[0][2]].extract_values().as_numpy_arrayF()
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
        # read ESS mtz output
        if self.column_exits(cidx['LAM']):
            self._lam = columns[cidx['LAM'][0]].extract_values().as_numpy_array()
        # read resolution
        self._resolution = np.array(self._obj.crystals()[0].unit_cell().d(self._obj.extract_miller_indices()))
        self._filename = filename

    def _batch_exits(self) -> bool:
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
    def column_exits(cidx) -> bool:
        """
        :return: Check the existence of a column of the certain data type.
        :rtype: bool
        """
        if cidx.size != 0:
            return True
        else:
            return False

    @filename_check
    def get_column_types(self) -> list[str, ...]:
        """
        :return: a list of column types
        :rtype: list
        """
        return self._obj.column_types()

    @filename_check
    def get_column_list(self) -> list[str, ...]:
        """
        :return: a list of column labels
        :rtype: list
        """
        return self._obj.column_labels()

    @filename_check
    def get_space_group(self) -> str:
        """
        :return: space group
        :rtype: str
        """
        return str(self._obj.space_group().info())

    @filename_check
    def get_max_resolution(self) -> float:
        """
        :return: Maximum resolution
        :rtype: float
        """
        return self._obj.max_min_resolution()[1]

    @filename_check
    def get_min_resolution(self) -> float:
        """
        :return: minimum resolution
        :rtype: float
        """
        return self._obj.max_min_resolution()[0]

    @staticmethod
    def sort_column_types(column_types_list: list[str, ...], column_labels_list: list[str, ...]) \
            -> Dict[str, np.ndarray[Literal["N"], np.int_]]:
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
        # ESS neutron mtz
        lam = np.argwhere((column_types == 'R') & (column_labels == 'LAM')).flatten()
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
                'Fmodel_phenix': fmodel_phenix_cidx,
                'LAM': lam
                }
        return cidx

    def group_by_redundancies(self):
        """Get the lists of indices/observations/resolutions grouped by the number of redundancy.

        :returns: tuple(indices_container, obs_container, resolution_container)
            WHERE
            list indices_container: lists of indices
            list obs_container: lists of observations
            list resolution_container: lists of resolutions
        """

        merged_array = next((_ for _ in iter(self._obj.as_miller_arrays()) if 'I' in _.info().labels), None)
        if merged_array is None:
            raise AttributeError("The mtz {0} has no intensity column.".format(self.file_name))
        unmerged_array = next((_ for _ in iter(self._obj.as_miller_arrays(merge_equivalents=False)) if 'I' in _.info().labels), None)
        if unmerged_array.info().merged is True:
            raise AttributeError("The mtz {0} are merged.".format(self.file_name))

        # prepare data arrays
        self._space_group = merged_array.crystal_symmetry().space_group()
        self._hkl_merged = np.array(merged_array.indices())

        # get redundancy of each reflection in merged data
        redund = miller.merge_equivalents(unmerged_array.map_to_asu()).redundancies().data().as_numpy_array()

        # get multiplicity of each reflection in merged data
        multi = merged_array.multiplicities().data().as_numpy_array()

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

        for idx, redund_num in enumerate(uni_redund[::-1]):  # loop through unique redundancy
            args_redund = np.where(redund == redund_num)[0]
            multi_of_args_redund = multi[args_redund]
            # separate the args_redund by the multiplicities of corresponding reflections
            args_redund_separated = [args_redund[multi_of_args_redund == uni] for uni in
                                     np.unique(multi_of_args_redund)]
            demension_reduction = 0
            for args in args_redund_separated:  # loop through args_redund separated by multiplicity
                # create iterator for merged data with specific multiplicity and redundancy
                it = self._hkl_merged[args]
                hkl_view = tmp.view([('a', int), ('b', int), ('c', int)])  # 1d view of Nx3 matrix

                set_by_multiplicity = list()
                for hkl_index in it:
                    if redund_num == 1:
                        set_by_multiplicity.append([hkl_index])
                        continue
                    sym_operator = miller.sym_equiv_indices(self._space_group, hkl_index.tolist())
                    if redund_num == 2:
                        if len(sym_operator.indices()) < redund_num:
                            demension_reduction += 1
                            continue
                    set_by_multiplicity.append([_.h() for _ in sym_operator.indices()])

                # set_by_multiplicity: NxMx3 array,
                # N: number of obs with specific multiplicity and redundancy
                # M: multiplicity
                set_by_multiplicity = np.array(set_by_multiplicity, dtype=int)
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
            indices_container[idx] = np.concatenate(indices_container[idx]).reshape(args_redund.size-demension_reduction, redund_num, 3)
            obs_container[idx] = np.concatenate(obs_container[idx]).reshape(args_redund.size-demension_reduction, redund_num)
            resolution_container[idx] = np.concatenate(resolution_container[idx]).reshape(args_redund.size-demension_reduction, redund_num)[:, 0]
            sigma_container[idx] = np.concatenate(sigma_container[idx]).reshape(args_redund.size-demension_reduction, redund_num)
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

    def scaling_outliers(self):
        assert self.hkl_by_multiplicity is not None
        from scipy.linalg import svd

        for i_redun, sig_redun in zip(self.intensity_by_multiplicity, self.sig_by_multiplicity):
            AA = i_redun[:, :, None] * i_redun[:, None, :]
            for A, ihl, sighl in zip(AA, i_redun, sig_redun):
                U, s, Vh = svd(A)
                whl = 1./(sighl*sighl)
                ghl = U[:, 0] * Vh[0, :]
                i_sum_denominator = np.sum(whl * ghl * ghl)
                i_sum_numerator = np.sum(whl * ghl * ihl)
                i_hl = i_sum_numerator / i_sum_denominator
                ghl_table =






