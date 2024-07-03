from iotbx import reflection_file_reader
from cctbx import uctbx, crystal, sgtbx

from .ReflectionBase import *


class ShlexParser(ReflectionParser):
    """

    """
    def __init__(self):
        super(ShlexParser, self).__init__()
        self.miller_set = None
        self.crystal_symmetry = None

    def read(self, filename: str,
             unit_cell: list[float, float, float, float, float, float],
             space_group_number: int):
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

