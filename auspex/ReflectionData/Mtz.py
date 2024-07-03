import os

import numpy as np
from iotbx import mtz

from .ReflectionBase import *


class MtzParser(ReflectionParser):
    """
    The Parser class to process mtz files.
    """

    def __init__(self):
        super(MtzParser, self).__init__()
        self._Fobs_refmac = None
        self._Fcalc_refmac = None

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