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
