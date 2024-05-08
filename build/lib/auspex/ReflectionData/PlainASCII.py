from cctbx import sgtbx

from .ReflectionBase import *


class IntegrateHKLPlain(ReflectionParser):
    def __init__(self):
        super(IntegrateHKLPlain, self).__init__()
        self._data_dict = None
        self._space_group = None
        self._xyz_cal = None
        self._xyz_obs = None
        self._corr_peak = None
        self._hkl_view = None

    def read_hkl(self, filename):
        data = {
            "H": [],  # 0
            "K": [],  # 1
            "L": [],  # 2
            "IOBS": [],  # 3
            "SIGMA": [],  # 4
            "XCAL": [],  # 5
            "YCAL": [],  # 6
            "ZCAL": [],  # 7
            "RLP": [],  # 8
            "PEAK": [],  # 9
            "CORR": [],  # 10
            "MAXC": [],  # 11
            "XOBS": [],  # 12
            "YOBS": [],  # 13
            "ZOBS": [],  # 14
            "ALF0": [],  # 15
            "BET0": [],  # 16
            "ALF1": [],  # 17
            "BET1": [],  # 18
            "PSI": [],  # 19
            "ISEG": []  # 20
        }
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('!SPACE_GROUP_NUMBER='):
                    self._space_group=sgtbx.space_group(sgtbx.space_group_symbols(
                        int(line.lstrip('!SPACE_GROUP_NUMBER=').strip())
                    ))
                if not line.startswith('!'):
                    columns = line.strip().split()
                    if len(columns) == 21:
                        for col_name, col_value in zip(data.keys(), columns):
                            data[col_name].append(float(col_value))
        self._data_dict = data
        self._hkl = np.array([data['H'], data['K'], data['L']], dtype=int).transpose()
        self._hkl_view = self._hkl.copy().view([('a', int), ('b', int), ('c', int)])
        self._xyz_cal = np.array([data['XCAL'], data['YCAL'], data['ZCAL']]).transpose()
        self._xyz_obs = np.array([data['XOBS'], data['YOBS'], data['ZOBS']]).transpose()
        self._corr_peak = np.array(data['CORR'], dtype=int)
        self._I = np.array(data['IOBS'])
        self._sigI = np.array(data['SIGMA'])

    def find_equiv_refl(self, h, k, l):
        bool_pos = np.full(self.size, False)
        sym_operator = miller.sym_equiv_indices(self._space_group, [int(h), int(k), int(l)])
        for miller_idx in sym_operator.indices():
            search_view = np.array(miller_idx.h()).view([('a', int), ('b', int), ('c', int)])
            bool_pos |= np.in1d(self._hkl_view, search_view)
        return np.argwhere(bool_pos).flatten()

    @property
    def size(self) -> int:
        """
        :return:
        :rtype: int
        """
        return self._I.size

    @property
    def corr(self) -> np.ndarray:
        """
        :return: correlation factor between observed and expected reflection profile
        :rtype: 1d ndarray
        """
        return self._corr_peak

    @property
    def xyz_obs(self) -> np.ndarray:
        return self._xyz_obs

    @property
    def xyz_cal(self) -> np.ndarray:
        return self._xyz_cal