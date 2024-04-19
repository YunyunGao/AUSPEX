import numpy as np
import os


class IceRing(object):
    """Constructor class for ice rings. Default ice ring ranges to the values in [1].
    [1] Acta Cryst D73, 729-737

    :param filename: path to customized ice ring ranges file.
    :type filename: str
    :param inverse_sqrt: whether the input is in inverse resolution squared. "True" if yes, "False" otherwise.
    :type inverse_sqrt: bool
    """
    def __init__(self, filename=None, inverse_sqrt=False):
        super(IceRing, self).__init__()
        self._ice_ring = None
        if filename is None:
            self._default_ice_ring()
        elif os.path.exists(filename):
            pass  # need to be edited
        if inverse_sqrt is True:
            self._ice_ring = 1./np.sqrt(self._ice_ring)
            # flip since the inverse
            # self._ice_ring[:, [0, 1]] = self._ice_ring[:, [1, 0]]

    def _default_ice_ring(self):
        self._ice_ring = np.array([[0.064, 0.069],
                                   [0.071, 0.078],
                                   [0.0825, 0.088],
                                   [0.138, 0.144],
                                   [0.19, 0.205],
                                   [0.228, 0.240],
                                   [0.262, 0.266],
                                   [0.267, 0.278],
                                   [0.280, 0.288],
                                   [0.337, 0.341],
                                   [0.429, 0.435],
                                   [0.459, 0.466],
                                   [0.478, 0.486],
                                   [0.531, 0.537],
                                   [0.587, 0.599],
                                   [0.606, 0.643],
                                   [0.650, 0.675],
                                   [0.711, 0.740],
                                   [0.775, 0.799],
                                   [0.828, 0.879],
                                   [0.904, 0.945],
                                   [0.967, 0.979],
                                   [1.001, 1.032],
                                   [1.039, 1.051],
                                   [1.057, 1.071]], dtype=float)

    def ice_ring_reader(self, filename):
        with open(filename) as infile:
            txt = infile.readlines()
            txt = [_.strip('\n').split() for _ in txt if not _.startswith('\n')]
            try:
                self._ice_ring = np.array(txt, dtype=float)
            except ValueError:
                raise ValueError('wrong file format for ice ring.')

    def is_in_ice_ring(self, res):
        assert isinstance(res, [float, int]), "invalid resolution number"
        return np.any((res >= self._ice_ring[:, 0]) & (res <= self._ice_ring[:, 1]))

    @property
    def ice_rings(self):
        return self._ice_ring


def IceRingTextReader(filename):
    """Helper function to read ice ring text file

    """
    #Read in the text file
    with open(filename) as infile:
        lower = []
        upper = []
        for line in infile.readlines():
            tokens = line.split()
            assert len(tokens) == 2
            lower.append(float(tokens[0]))
            upper.append(float(tokens[1]))
    ice_ring = np.array([lower, upper])
    return ice_ring.T
