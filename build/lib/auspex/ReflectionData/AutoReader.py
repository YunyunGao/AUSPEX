from iotbx.cif import CifParserError

from . import Cif, Mtz, Xds, Dials, Shlex


def FileReader(file_name: str, file_type: str = None, *args):
    """A universal format parser to popular data formats.

    :param file_name: The name or path of the input file.
    :param file_type: Optional.

    :param args: (unit cell, space group number). Needed only when the input file does not include cell information.
    :return: Parsed reflection data.
    """
    if file_name[-3:] == 'mtz' or (file_type in ('xds', 'mtz', 'MTZ', 'mrg', 'binary')):
        try:
            reflection_data = Mtz.MtzParser()
            reflection_data.read(file_name)
            reflection_data.source_data_format = 'mtz'
        except AssertionError:
            print('Failed to read the mtz file. Check the data format or specify the input type using --input-type.')
    elif file_name[-3:] == 'HKL' or (file_type in ('xds', 'HKL', 'xds_HKL')):
        try:
            reflection_data = Xds.XdsParser()
            reflection_data.read_hkl(file_name, merge_equivalents=False)
            reflection_data.source_data_format = 'xds_hkl'
        except AssertionError:
            print('Failed to read the XDS HKL file. Check the data format or specify the input type using --input-type.')
    elif file_name[-3:] == 'cif' or (file_type in ['cif', 'mmcif', 'CIF']):
        try:
            reflection_data = Cif.CifParser()
            reflection_data.read(file_name)
            reflection_data.source_data_format = 'cif'
        except CifParserError or AssertionError:
            print('Failed to read the cif file. Check the data format or specify the input type using --input-type.')
    elif file_name[-4:] == 'refl' or (file_type in ['refl', 'dials', 'msgpack']):
        try:
            reflection_data = Dials.DialsParser()
            reflection_data.smart_read(file_name)
            reflection_data.source_data_format = 'refl'
        except AssertionError:
            print('Failed to read the DIALS file. Check the data format or specify the input type using --input-type')
    elif file_name[-3:] == 'hkl' or (file_type in ['shlex', 'hkl', 'shlex_hkl']):
        try:
            if len(args) == 2:
                unit_cell, space_group_number = args
            reflection_data = Shlex.ShlexParser()
            reflection_data.read(file_name, unit_cell, space_group_number)
            reflection_data.source_data_format = 'shlex_hkl'
        except RuntimeError as err:
            print('An error occurred when parsing shlex hkl: ', err)
    else:
        reflection_data = None
    return reflection_data