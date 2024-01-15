import sys
import os
import argparse
from argparse import RawTextHelpFormatter
from os.path import exists, basename, splitext
from Plotter import PlotGenerator
from Auspex import IceFinder
from IceRings import IceRing
from ReflectionData import FileReader
from Verbose import MergeStatistics

command_line = ' '.join(sys.argv[1:])

parser = argparse.ArgumentParser(
    description='AUSPEX',
    formatter_class=RawTextHelpFormatter
)

parser.add_argument(
    'hklin',
    metavar='HKLIN',
    type=str,
    nargs=1,
    help='The file to be analyzed. Can be one of mtz, XDS HKL, cif, scalepack and SHLEX hkl.'
)

parser.add_argument(
    '--helcaraxe',
    dest='helcaraxe',
    action="store_false",
    default=True,
    help='Use CNN model to predict ice contamination. The default option'
)

parser.add_argument(
    '--directory',
    dest='directory',
    type=str,
    default='.',
    help='The output directory.'
)

parser.add_argument(
    '--ylim',
    dest='ylim',
    type=str,
    default='auto',
    help='''Specify the y limit mode for the plots. Options:
minmax\tPlot everything.
auto\tPlot only core of distribution automatically (default).
auto_low\tPlot only core of distribution automatically, with a focus of lower values.
        low\tPlot only values below mean.'''
)

parser.add_argument(
    '--dmin',
    dest='dmin',
    type=float,
    default=None,
    help='Specify the maximum resolution to show in the plots.'
)

parser.add_argument(
    '--no-filename-in-title',
    dest='no_filename_in_title',
    action='store_true',
    default=False,
    help='Should the filename be shown in the title? (Options: true / false)'
)

parser.add_argument(
    '--single-figure',
    dest='single_figure',
    action='store_true',
    default=False,
    help='Should the images be generated in separate png files? (Default: No.)'
)

parser.add_argument(
    '--score-figure',
    dest='score_figure',
    action='store_true',
    default=False,
    help='Should only a scoring image be generated instead of usual output? (Default: No.)'
)

parser.add_argument(
    '--no-individual',
    dest='no_individual_figures',
    action='store_true',
    default=False,
    help='Should individual figures not be made for each plot?  (Default: No; set option for Yes.)'
)

parser.add_argument(
    '--no-automatic',
    dest='no_automatic',
    action='store_true',
    default=False,
    help='If set, no ice rings will be flagged by red bars.'
)

parser.add_argument(
    '--cutoff',
    dest='cutoff',
    type=float,
    default=5,
    help='Specify the cut off for IceFinderScore (default: 5).'
)

parser.add_argument(
    '--binning',
    dest='binning',
    type=float,
    default=0.001,
    help='Specify the bin size for individual bins, in 1/Angstroem (default: 0.001).'
)

parser.add_argument(
    '--text-output',
    dest='text_filename',
    default=None,
    help='Write out a text file with the mtz columns'
)

parser.add_argument(
    '--dont-use-anom-if-present',
    dest='use_anom_if_present',
    action="store_false",
    default=True,
    help='Use I(+)/I(-) and F(+)/F(-) for plots if present'
)

parser.add_argument(
    '--unit-cell',
    dest='unit_cell',
    default=None,
    nargs=6,
    help='Specify the unit cell parameters for the input file. '
         'The format should be: a b c alpha beta gamma (e.g. 100.0 100.0 100.0 90.0 90.0 90.0)'
)

parser.add_argument(
    '--space-group-number',
    dest='space_group_number',
    default=None,
    type=int,
    help='Specify the space group number for the input file. Only integer is accepted'
         'For space group number, refer to https://strucbio.biologie.uni-konstanz.de/xdswiki/index.php/Space_group_determination'
)

parser.add_argument(
    '--input-type',
    dest='input_type',
    default=None,
    help='Specify the input file type or integration software used .'
)

args = parser.parse_args()
filename = args.hklin[0]
output_directory = args.directory
version = "2.0.0"

print("")
print("            ######################################################## ")
print("           #                    _   _   _ ____  ____  _______  __   #")
print("           #       A           / \ | | | / ___||  _ \| ____\ \/ /   #")
print("           #   /MmmOmmM\      / _ \| | | \___ \| |_) |  _|  \  /    #")
print("           #       #         / ___ \ |_| |___) |  __/| |___ /  \    #")
print("           #      /#\       /_/   \_\___/|____/|_|   |_____/_/\_\   #")
print("           #                                        {:>13s}   #".format("Version " + str(version)))
print("            ######################################################## ")
print("\nCOMMAND LINE: auspex {0}".format(command_line))

if exists(filename):
    # Original line
    # share = os.path.join(sysconfig.PREFIX, 'share')
    # For CCP4:
    #share = os.path.join(os.environ['AUSPEX_INSTALL_BASE'], 'share')
    #auspex_package_dir = os.path.join(share, 'auspex')
    #auspex_package_data_dir = os.path.join(auspex_package_dir, 'data')

    ice = IceRing()
    reflection_data = FileReader(filename, args.input_type, args.unit_cell, args.space_group_number)
    print(reflection_data.source_data_format)
    if reflection_data.source_data_format in ('xds_hkl', 'shlex_hkl'):
        #try:
            reflection_data.group_by_redundancies()
            merge_stats = MergeStatistics(reflection_data.merge_stats_binned(), reflection_data.merge_stats_overall())
            merge_stats.print_stats_table()
        #except:
         #   pass
    ice_info = IceFinder(reflection_data, ice, use_anom_if_present=args.use_anom_if_present)
    if args.helcaraxe is True:
        ice_info.run_helcaraxe()
    elif ice_info.fobs is not None and args.use_anom_if_present:
        try:
            ice_info.binning('F_ano', binning=args.binning)
        except AssertionError:
            ice_info.binning('F', binning=args.binning)
    elif (ice_info.iobs is not None) and (ice_info.fobs is None):
        try:
            ice_info.binning('I_ano', binning=args.binning)
        except AssertionError:
            ice_info.binning('I', binning=args.binning)

    # Write a text file
    if args.text_filename is not None:
        ice_info.WriteTextFile(args.text_filename)

    plot = PlotGenerator(
        ice_info,
        output_directory=args.directory,
        ylim=args.ylim,
        dmin=args.dmin,
        filename_in_title=not args.no_filename_in_title,
        single_figure=args.single_figure,
        score_figure=args.score_figure,
        no_individual_figures=args.no_individual_figures,
        cutoff=args.cutoff,
        no_automatic=args.no_automatic)
    name_stub = splitext(basename(filename))[0]

    plot.name_stub = name_stub  # = join(output_directory, "%s.png" % )

    plot.generate(ice_info)

    have_ice_rings_been_flagged = ice_info.has_ice_rings

    if have_ice_rings_been_flagged:
        if os.path.exists("mtz_with_ice_ring.txt"):
            os.remove("mtz_with_ice_ring.txt")
        outfile = open("mtz_with_ice_rings.txt", "a")
        print(os.path.split(filename)[1], file=outfile)

else:
    print("File {0} does not exist.".format(filename))


