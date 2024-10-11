from iotbx import cif

from .ReflectionBase import *


class CifParser(ReflectionParser):
    """The Parser class to process cif files.

    """

    def __init__(self):
        super(CifParser, self).__init__()
        self._resolutionF = None
        self._resolutionI = None
        self._resolutionI_ano = None
        self._resolution = None

    def read(self, filename: str = None):
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
            if self._resolutionF is not None and self._resolutionI is None:
                self._resolution = self._resolutionF
            if self._resolutionI is not None and self._resolutionF is None:
                self._resolution = self._resolutionI
            if self._resolutionI_ano is not None:
                self._resolution = self._resolutionI_ano

    @filename_check
    def get_space_group(self) -> str:
        """
        :return: Space group text.
        :rtype: str
        """
        return str(self._space_group.info())

    @filename_check
    def get_cell_dimension(self) -> list[float, float, float, float, float, float]:
        """
        :return: Unit cell parameters (a, b, c, alpha, beta, gamma).
        :rtype: list
        """
        return self._unit_cell.parameters()

    @filename_check
    def get_max_resolution(self) -> float:
        """
        :return: maximum resolution
        :rtype: float
        """
        return self._resolution.min()

