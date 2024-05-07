import iotbx.scalepack.merge as sca_merge

from .ReflectionBase import *

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