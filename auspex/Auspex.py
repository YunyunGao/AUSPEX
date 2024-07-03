from ReflectionData import Mtz, Dials, Cif, Xds

from BinnedData import BinnedSummaries
from IceRings import IceRing

from Helcaraxe import cnn_predict

import numpy as np
import copy

from typing import Literal


class IceFinder(object):
    """The class for calculating ice ring scores.

    :param reflection_data: an instance of parsed reflection data
    :type reflection_data: ReflectionData subclass
    :param ice_ring: an instance of IceRing
    :type ice_ring: IceRing
    :param use_anom_if_present: Use anomalous data if present, default to True
    :type use_anom_if_present: bool
    """
    def __init__(self,
                 reflection_data: Mtz.MtzParser | Dials.DialsParser | Cif.CifParser | Xds.XdsParser = None,
                 ice_ring: IceRing = None,
                 use_anom_if_present: bool = True):
        """
        Constructor method.
        """
        self._reflection_data = reflection_data
        self._use_anom_if_present = use_anom_if_present
        self._intensity_data = self._reflection_data.get_intensity_data()
        self._amplitude_data = self._reflection_data.get_amplitude_data()
        self._intensity_ano_data = self._reflection_data.get_intensity_anom_data()
        self._amplitude_ano_data = self._reflection_data.get_amplitude_anom_data()
        self._file_name = self._reflection_data.file_name
        if ice_ring is None:
            self._ice_ring = IceRing()
        else:
            self._ice_ring = ice_ring
        self._bool_ranges_in_ice = None
        self._icefinder_scores = None
        self._helcaraxe_status = False
        self._cnn_predicted_i = None
        self._cnn_predicted_f = None
        self._has_ice_rings = False

    def binning(self,
                obs_type: str = 'F',
                binning: float = 0.001):
        """Construct a binned dataset.

        :param obs_type: observation type to be used, can be 'F' or 'I', default to 'F'
        :param binning: the bin width, default to 0.001
        """
        # TODO: assert obs_type in list ...
        if obs_type == 'I':
            self._binned_summaries = BinnedSummaries(self._intensity_data)
        elif obs_type == 'F':
            self._binned_summaries = BinnedSummaries(self._amplitude_data)
        elif obs_type == 'I_ano' and self._use_anom_if_present:
            assert self._intensity_ano_data is not None, 'No available anomalous intensity data.'
            self._binned_summaries = BinnedSummaries(self._intensity_ano_data)
        elif obs_type == 'F_ano' and self._use_anom_if_present:
            assert self._amplitude_ano_data is not None, 'No available anomalous amplitude data.'
            self._binned_summaries = BinnedSummaries(self._amplitude_ano_data)
        else:
            raise TypeError('Wrong observation type. Need to be one of I, F, I_ano and F_ano.')
        self._binned_summaries.set_binning_rules(binning)
        self._binned_summaries.bins_in_icering(self._ice_ring)

    def is_in_ice_ring(self) -> np.ndarray[Literal["N"], np.int16]:
        """
        :return: list of boolean values representing whether a bin is within the range of ice ring
        :rtype: Nx1 ndarray(dtype=int)
        """
        return self._binned_summaries.bin_args_in_icering(self._ice_ring)

    def icefinder_scores(self) -> np.ndarray[Literal["N"], np.float32]:
        """Calculate icefinder scores for all bins.
        :return: icefinder scores of all bins
        :rtype: Nx1 ndarray(dtype=int)
        """
        if self._icefinder_scores is None:
            self._binned_summaries.get_est_stdmeans()
            self._icefinder_scores = self._binned_summaries.icefinder_score()
        else:
            pass
        return self._icefinder_scores

    def run_helcaraxe(self):
        """Run HELCARAXE prediction based on intensity and/or amplitude.
        :return: None
        """
        if (self._amplitude_ano_data is not None) and self._use_anom_if_present:
            fres, fobs = self._amplitude_ano_data.ires, self._amplitude_ano_data.obs
        elif self._amplitude_data is not None:
            fres, fobs = self._amplitude_data.ires, self._amplitude_data.obs
        else:
            fres, fobs = None, None

        if (self._intensity_ano_data is not None) and self._use_anom_if_present:
            ires, iobs = self._intensity_ano_data.ires, self._intensity_ano_data.obs
        elif self._intensity_data is not None:
            ires, iobs = self._intensity_data.ires, self._intensity_data.obs
        else:
            ires, iobs = None, None

        self._cnn_predicted_i, self._cnn_predicted_f \
            = cnn_predict(ires, iobs,
                          fres, fobs)
        self._cnn_predicted_i = np.nan_to_num(self._cnn_predicted_i, nan=1.)
        self._cnn_predicted_f = np.nan_to_num(self._cnn_predicted_f, nan=1.)
        if (self._cnn_predicted_i is not None) or (self._cnn_predicted_f is not None):
            self._helcaraxe_status = True

    def mean_ires_squared(self) -> np.ndarray[Literal["N"], np.float32]:
        """Return the mean d-spacing squared for all bins.
        :return: mean d-spacing squared (inverse resolution squared) of all bins
        :rtype: Nx1 ndarray(dtype=float)
        """
        return self._binned_summaries.mean_invresolsq_all()

    def ice_range_by_icefinderscore(self, cutoff: float = 5.) -> np.ndarray[Literal["N", 2], np.float32]:
        """Calculate the ice ring range based on icefinder scores. Return the lower and upper bounds.
        :param cutoff: Threshold for peak identification in icefinder_score. Default: 5.0.
        :return: Lower and upper d-spacing squared where ice rings are potentially present
                 based on icefinderscore.
        :rtype: Nx2 ndarray
        """
        with np.errstate(invalid='ignore'):
            args_possible_ice = np.abs(self.icefinder_scores()) >= cutoff
        self._args_ice_by_icefinderscore = np.logical_and(args_possible_ice, self.is_in_ice_ring())
        self._bool_ranges_in_ice = np.any((self.mean_ires_squared()[self._args_ice_by_icefinderscore][:, None]
                                           >= self._ice_ring.ice_rings[:, 0][None, :]) &
                                          (self.mean_ires_squared()[self._args_ice_by_icefinderscore][:, None]
                                           <= self._ice_ring.ice_rings[:, 1][None, :]),
                                          axis=0)
        if np.any(self._args_ice_by_icefinderscore):
            self._has_ice_rings = True
        return self._ice_ring.ice_rings[self._bool_ranges_in_ice]

    def ice_range_by_helcaraxe(self, cutoff: float = .02) -> np.ndarray[Literal["N", 2], np.float32]:
        """
        :return: lower and upper resolutions in inverse squared angstrom where ice rings are potentially present
                based on HELCARAXE prediction
        :rtype: Nx2 ndarray
        """
        if (self._cnn_predicted_f is not None) and (self._cnn_predicted_i is None):
            self._bool_ranges_in_ice = self._cnn_predicted_f >= cutoff
        elif (self._cnn_predicted_f is None) and (self._cnn_predicted_i is not None):
            self._bool_ranges_in_ice = self._cnn_predicted_i >= cutoff
        elif (self._cnn_predicted_f is not None) and (self._cnn_predicted_i is not None):
            self._bool_ranges_in_ice = (self._cnn_predicted_f >= cutoff) & (self._cnn_predicted_i >= cutoff)
        else:
            raise Exception("No Helcaraxe prediction. Please try to rerun Helcaraxe.")
        if np.any(self._bool_ranges_in_ice > cutoff):
            self._has_ice_rings = True
        return self._ice_ring.ice_rings[self._bool_ranges_in_ice]

    def quantitative_score(self) -> np.ndarray[Literal["N"], np.float32]:
        """Quantitative score based on HELCARAXE
        :return: The scores for each potential ice range in a scale of 0 to 1.
        """
        if (self._cnn_predicted_f is not None) and (self._cnn_predicted_i is None):  # Helcaraxe F
            args_minus = self._cnn_predicted_f == -1.
            scores = copy.deepcopy(self._cnn_predicted_f)
            scores[args_minus] = 0.
            scores_in_ice_range = scores[self._bool_ranges_in_ice]
        elif (self._cnn_predicted_f is None) and (self._cnn_predicted_i is not None):  # Helcaraxe I
            args_minus = self._cnn_predicted_i == -1.
            scores = copy.deepcopy(self._cnn_predicted_i)
            scores[args_minus] = 0.
            scores_in_ice_range = scores[self._bool_ranges_in_ice]
        elif (self._cnn_predicted_f is not None) and (self._cnn_predicted_i is not None):
            scores = np.maximum(self._cnn_predicted_f, self._cnn_predicted_i)
            args_minus = scores == -1
            scores[args_minus] = 0.
            scores_in_ice_range = scores[self._bool_ranges_in_ice]
        else:  # normalized icefinderscore
            icefinder_scores_in_ice = self.icefinder_scores()[self._args_ice_by_icefinderscore]
            # groupped_bin_args = self._binned_summaries.bins_in_icering_groupped(self._ice_ring)
            # scores_in_ice_range = []
            # for args in groupped_bin_args:
            #    scores_in_ice_range.append(self.icefinder_scores()[args].max())
            sorted_indices = sorted(np.argsort(icefinder_scores_in_ice)[-np.sum(self._bool_ranges_in_ice):])
            scores_in_ice_range = icefinder_scores_in_ice[sorted_indices]
            scores_in_ice_range /= icefinder_scores_in_ice.max()
        if np.any(self._bool_ranges_in_ice):
            self._has_ice_rings = True
        return scores_in_ice_range

    @property
    def file_name(self) -> str:
        """
        :return: path to data file
        :rtype: str
        """
        return self._file_name

    @property
    def ice_ring(self) -> IceRing:
        """
        :return: ice ring instance
        :rtype: IceRings.IceRing
        """
        return self._ice_ring

    @property
    def iobs(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: intensity data
        :rtype: ReflectionData.Observation
        """
        if self._use_anom_if_present is True and self._intensity_ano_data is not None:
            return self._intensity_ano_data
        else:
            return self._intensity_data

    @property
    def fobs(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: amplitude data
        :rtype: ReflectionData.Observation
        """
        if self._use_anom_if_present is True and self._amplitude_ano_data is not None:
            return self._amplitude_ano_data
        else:
            return self._amplitude_data

    @property
    def cnn_predicted_i(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: helcaraxe prediction based on intensity
        :rtype: 1d ndarray
        """
        return self._cnn_predicted_i

    @property
    def cnn_predicted_f(self) -> np.ndarray[Literal["N"], np.float32]:
        """
        :return: helcaraxe prediction based on amplitude
        :rtype: 1d ndarray
        """
        return self._cnn_predicted_f

    @property
    def helcaraxe_status(self) -> bool:
        """
        :return: True is helcaraxe has succeeded
        :rtype: Bool
        """
        return self._helcaraxe_status

    @property
    def has_ice_rings(self) -> bool:
        """
        :return: True if data have ice rings
        :rtype: Bool
        """
        return self._has_ice_rings

    @property
    def background(self):
        return None





