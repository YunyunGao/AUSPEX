from __future__ import print_function
from __future__ import division
import os
import matplotlib
import mpl_scatter_density
import Verbose
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Set the font size and then set the maths font to san-serif
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['mathtext.fontset'] = 'stixsans'


class GenericPlot(object):
    """A class to do a generic plot for Auspex
    Initialize with the ice ring resolutions.

    :param ice_rings: instance of IceRing.IceRing
    :type ice_rings: IceRings.IceRing
    :param filename: file or filename to reflection data
    :type filename: str
    :param output_directory: file or filename to which the data is saved.
    :type output_directory: str
    :param ylim: 'auto', 'auto_low', 'low' or 'maxmin'
    :type ylim: str
    :param dmin: minimum resolution to plot
    :type dmin: float
    :param filename_in_title: whether to show file name on plot
    :type filename_in_title: bool
    :param cutoff: cutoff for icefinder score
    :type cutoff: float
    :param num_xticks: number of x ticks
    :type num_xticks: int
    :param no_automatic: whether to show the vertical red bars
    :type no_automatic: bool
    :param ax: matplotlib axes
    :type ax: matplotlib.pyplot.axes
    """

    def __init__(self, 
                 ice_rings,
                 filename=None,
                 output_directory=None,
                 ylim=None,
                 dmin=None,
                 filename_in_title=False,
                 cutoff=5,
                 num_xticks=10,
                 no_automatic=False,
                 ax=None):
        self.ice_rings = ice_rings
        self.mtzfilename = filename
        self.ylim = ylim
        self.dmin = dmin
        self.filename_in_title = filename_in_title
        # Set the plot size
        self.plotwidth = 8
        self.dpi = 300
        self.num_xticks = num_xticks
        # Set the label padding
        self.xlabelpad = 4
        self.ylabelpad = 3
        self.titlepos = 1.02
        # Own figure
        self.ax = ax
        # Set cutoff
        self.cutoff = cutoff

        # Set if red bars should be shown.
        self.no_automatic = no_automatic

        # Set the output directory
        self.output_directory = output_directory
        self.name_stub = "output"

    def get_ymax(self, y_data, multiplier):
        ymax = min(y_data.max(),
                   ((np.nansum(y_data)/y_data.size) + multiplier*np.nanstd(y_data)))
        print(np.std(y_data))
        return ymax

    def generate(self, icefinder_handle, resolution, y_data):
        """
        :param icefinder_handle:
        :param resolution:
        :param y_data:
        :return:
        """
        # Setup the plot 1
        if self.ax is None:
            figure = plt.figure(figsize=(self.plotwidth, np.sqrt(2)/3*self.plotwidth))  # TODO: Implement adjustable ratio
            ax1 = figure.add_subplot(111, projection='scatter_density')
        else:
            ax1 = self.ax

        # Define function that calculates 1/x^2
        def formatter(x, p):
            if x <= 0:
                return ''
            else:
                return '{:.2f}'.format(np.sqrt(1.0/x))

        # Label the ticks in angstroms
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(formatter))

        # Set the number of x ticks
        ax1.set_xticks(
            np.arange(
                resolution.min(),
                resolution.max(),
                0.99*(resolution.max() - resolution.min()) / (self.num_xticks - 1)))

        # Label the axes 
        ax1.set_xlabel(self.xlabel, labelpad=self.xlabelpad)
        ax1.set_ylabel(self.ylabel, labelpad=self.ylabelpad)
        if self.filename_in_title:
            ax1.set_title("{0}\n {1}".format(self.title, self.mtzfilename), y=self.titlepos)
        else:
            ax1.set_title(self.title, y=self.titlepos)

        # Set the plot limits
        xmax = resolution.max()
        xmin = min(0, resolution.min())
        if self.ylim == 'auto':
            ymax = self.get_ymax(y_data, 3)
            ymin = min(0, max(y_data.min(), -ymax))
        elif self.ylim == 'auto_low':
            ymax = self.get_ymax(y_data, 2)
            ymin = min(0, y_data.min())
        elif self.ylim == 'low':
            ymax = self.get_ymax(y_data, 0)
            ymin = min(0, y_data.min())
        elif self.ylim == 'minmax':
            ymax = max(y_data)
            ymin = min(0, y_data.min())
        if self.dmin is not None:
            xmax = 1.0 / self.dmin**2

        # shrink y_data to an array with values from ymin to ymax only and shrink resolution correspondingly
        args_shrink = (y_data <= ymax) & (y_data >= ymin)
        y_data_shrinked = y_data[args_shrink]
        resolution_shrinked = resolution[args_shrink]

        # Plot the points
        point_size = 4
        ax1.scatter(
            resolution_shrinked,
            y_data_shrinked,
            point_size, 
            zorder=2, 
            color="#0d0d0d", 
            alpha=0.7*np.exp(-y_data.size*0.00004)+0.020,
            linewidth=0)

        # Plot a teal line at y = 0
        ax1.axhline(0, color='#415a55', alpha=0.5)

        #ax1.grid(True)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)

        # Loop through ice ring resolutions and plot
        # now the transparency of the ice ring patch is a function of icefinder score
        for lb, ub in self.ice_rings:
            rectangle = matplotlib.patches.Rectangle(
                (lb, ymin),
                ub - lb,
                ymax - ymin, 
                color="grey", 
                zorder=1, 
                alpha=0.5*self.cutoff/15,
                linewidth=0)
            ax1.add_patch(rectangle)

        have_ice_rings_been_flagged = False

        # Core functions to calculate icefinder_score and helcaraxe_score
        if not self.no_automatic:
            if icefinder_handle.helcaraxe_status:
                ice_spike_range = icefinder_handle.ice_range_by_helcaraxe()
            else:
                ice_spike_range = icefinder_handle.ice_range_by_icefinderscore(cutoff=self.cutoff)
            ice_spike_scores = icefinder_handle.quantitative_score()
            for ranges, scores in zip(ice_spike_range, ice_spike_scores):
                lb, ub = ranges
                rectangle = matplotlib.patches.Rectangle(
                    (lb, ymin),
                    ub - lb,
                    ymax - ymin,
                    color="red",
                    zorder=1,
                    alpha=0.5*scores,
                    linewidth=0)
                ax1.add_patch(rectangle)

        if have_ice_rings_been_flagged:
            if os.path.exists("mtz_with_ice_ring.txt"):
                os.remove("mtz_with_ice_ring.txt")
            outfile = open("mtz_with_ice_rings.txt", "a")
            print(os.path.split(self.mtzfilename)[1], file=outfile)


class IPlot(GenericPlot):
    """Plot the I vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(IPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "I_plot.png"
        self.title = r'$\mathrm{I_{obs}}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{I_{obs}}$'
        self.data_type = 'I'


class SigIPlot(GenericPlot):
    """Plot the Sig(I) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(SigIPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "SigI_plot.png"
        self.title = r'$\mathrm{\sigma(I_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{\sigma(I_{obs})}$'
        self.data_type = 'SigI'


class IoverSigIPlot(GenericPlot):
    """Plot the I / Sig(I) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(IoverSigIPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "ISigI_plot.png"
        self.title = r'$\mathrm{I_{obs} / \sigma(I_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{I_{obs} / \sigma(I_{obs})}$'
        self.data_type = 'ISigI'


class FPlot(GenericPlot):
    """Plot the F vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(FPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "F_plot.png"
        self.title = r'$\mathrm{F_{obs}}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{F_{obs}}$'
        self.data_type = 'F'


class SigFPlot(GenericPlot):
    """Plot the Sig(F) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(SigFPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "SigF_plot.png"
        self.title = r'$\mathrm{\sigma(F_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{\sigma(F_{obs})}$'
        self.data_type = 'SigF'

class FoverSigFPlot(GenericPlot):
    """ Plot the F / Sig(F) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(FoverSigFPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "FSigF_plot.png"
        self.title = r'$\mathrm{F_{obs} / \sigma(F_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{F_{obs} / \sigma(F_{obs})}$'
        self.data_type = 'FSigF'


class BPlot(GenericPlot):
    """Plot the B vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(BPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "B_plot.png"
        self.title = r'$\mathrm{B_{obs}}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{B_{obs}}$'
        self.data_type = 'B'


class SigBPlot(GenericPlot):
    """Plot the Sig(B) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(SigBPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "SigB_plot.png"
        self.title = r'$\mathrm{\sigma(B_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{\sigma(B_{obs})}$'
        self.data_type = 'SigB'


class BoverSigBPlot(GenericPlot):
    """Plot the B / Sig(B) vs resolution
    """
    def __init__(self, ice_rings, **kwargs):
        super(BoverSigBPlot, self).__init__(ice_rings, **kwargs)
        self.filename = "BSigB_plot.png"
        self.title = r'$\mathrm{B_{obs} / \sigma(B_{obs})}$ vs. resolution'
        self.xlabel = r'$1/\mathrm{resolution} (\mathrm{\AA})$' # r'$1/\mathrm{resolution}^{2} (\mathrm{\AA}^{-2})$'
        self.ylabel = r'$\mathrm{B_{obs} / \sigma(B_{obs})}$'
        self.data_type = 'BSigB'


class PlotGenerator(object):
    """ Initialise the generator.

    :param icefinder_handle: ice finder instance
    :type: Auspex.Icefinder
    :param ylim: list of y-axis range (lower, upper)
    :type: list
    :param output_directory: path to output directory
    :type: str
    :param dmin: minimum resolution to plot
    :type dmin: float
    :param filename_in_title: whether to show file name on plot
    :type filename_in_title: bool
    :param single_figure: whether to plot integrated figure. "True" if yes
    :type single_figure: bool
    :param score_figure: whether to plot icefinder score figure. "True" if yes
    :type score_figure: bool
    :param no_individual_figures: whether to NOT plot individual figure. "True" if NOT to plot.
    :type no_individual_figures: bool
    :param num_xticks: number of ticks on x-axis
    :type num_xticks: int
    :param cutoff: y-axis cutoff for icefinder score figure
    :type cutoff: float, int
    :param no_automatic:
    :type no_automatic: bool
    """
    def __init__(self,
                 icefinder_handle,
                 ylim=None,
                 output_directory=None,
                 dmin=None,
                 filename_in_title=False,
                 single_figure=False,
                 score_figure=False,
                 no_individual_figures=False,
                 num_xticks=10,
                 cutoff=5,
                 no_automatic=False):
        # Figure properties
        self._plotwidth = 10
        self._dpi = 300
        # Save some parameters
        self.filename = icefinder_handle.file_name
        self.output_directory = output_directory
        self.ylim = ylim
        self.dmin = dmin
        self.filename_in_title = filename_in_title
        # Set num x ticks
        self.num_xticks = num_xticks
        self.icefinder_handle = icefinder_handle
        # Save the ice ring resolutions
        self._ice_rings = icefinder_handle.ice_ring.ice_rings
        # A single figure or separate
        self._single_figure = single_figure
        # Extra mode for icefinder score
        self._score_figure = score_figure
        # Do individual figures:
        self._no_individual_figures = no_individual_figures
        # Save the cutoff value
        self._cutoff = cutoff
        # Save automatic ice ring flagging yes/no
        self._no_automatic = no_automatic

    def generate_I_plot(self, I, SigI, D2, ax=None):
        """Generate the I plot.

        :param I:
        :param SigI:
        :param D2:
        :param ax:
        :return:
        """
        plot = IPlot(self._ice_rings,
                     ax=ax,
                     filename=self.filename,
                     output_directory=self.output_directory,
                     cutoff=self._cutoff,
                     ylim=self.ylim,
                     dmin=self.dmin,
                     num_xticks=self.num_xticks,
                     filename_in_title=self.filename_in_title,
                     no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, I)

    def generate_SigI_plot(self, I, SigI, D2, ax=None):
        """Generate the SigI plot
        """
        plot = SigIPlot(self._ice_rings,
                        ax=ax,
                        filename=self.filename,
                        output_directory=self.output_directory,
                        cutoff=self._cutoff,
                        ylim=self.ylim,
                        dmin=self.dmin,
                        num_xticks=self.num_xticks,
                        filename_in_title=self.filename_in_title,
                        no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, SigI)

    def generate_ISigI_plot(self, I, SigI, D2, ax=None):
        """Generate the I / SigI plot
        """
        #D2, ISigI = list(zip(*[(d, i / sigi) for d, i, sigi in zip(D2, I, SigI) if sigi > 0]))
        arg_sig_valid = SigI > 0.
        ISigI = I[arg_sig_valid] / SigI[arg_sig_valid]
        D2 = D2[arg_sig_valid]

        plot = IoverSigIPlot(self._ice_rings,
                             ax=ax,
                             filename=self.filename,
                             output_directory=self.output_directory,
                             cutoff=self._cutoff,
                             ylim=self.ylim,
                             dmin=self.dmin,
                             num_xticks=self.num_xticks,
                             filename_in_title=self.filename_in_title,
                             no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, ISigI)

    def generate_standardised_mean_plot(self, I, D2, ax=None):
        xmax = max(D2)
        xmin = min(0, min(D2))

        standardised_mean = self.icefinder_handle._binned_summaries.get_stdmean_all()
        estimated_standardised_mean = self.icefinder_handle._binned_summaries.get_est_stdmeans()
        i_res_squared = self.icefinder_handle._binned_summaries.mean_invresolsq_all()
        # for idx in range(self.icefinder_handle.Size()):
        #     i_res_squared.append(self.icefinder_handle._binned_summaries.MeanIResSquared(idx))
        #     standardised_mean.append(self.icefinder_handle._binned_summaries.ObsStandardisedMean(idx))
        #     estimated_standardised_mean.append(self.icefinder_handle._binned_summaries.EstimatedStandardisedMean(idx))

        #Define function that calculates 1/x^2
        def formatter(x, p):
            if x <= 0:
                return ''
            else:
                return '%.2f' % np.sqrt(1.0/x)

        # Set the number of x ticks
        ax.set_xticks(
            np.arange(
                min(D2), 
                max(D2), 
                0.99*(max(D2) - min(D2)) / (self.num_xticks-1)))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(formatter))
        ax.set_ylabel("observed standardized mean")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-2, 4)
        ax.axhline(1, color='#415a55', alpha=0.5)
        ax.plot(i_res_squared, estimated_standardised_mean,  color="Red", lw=1)    
        ax.plot(i_res_squared, standardised_mean,  color="Black", lw=1)

    def generate_icefinderscore_plot(self, I, D2, ax=None):

        i_res_squared = self.icefinder_handle._binned_summaries.mean_invresolsq_all()
        icefinderscore = self.icefinder_handle.icefinder_scores()
        # for idx in range(self.icefinder_handle.Size()):
        #     i_res_squared.append(self.icefinder_handle.MeanIResSquared(idx))
        #     icefinderscore.append(self.icefinder_handle.IcefinderScore(idx))

        xmax = max(D2)
        xmin = min(0, min(D2))
        #Define function that calculates 1/x^2
        def formatter(x, p):
            if x <= 0:
                return ''
            else:
                return '%.2f' % np.sqrt(1.0/x)

        # Set the number of x ticks
        ax.set_xticks(
            np.arange(
                min(D2), 
                max(D2), 
                0.99*(max(D2) - min(D2)) / (self.num_xticks-1)))

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(formatter))
        # Set the number of x ticks

        ax.set_ylabel("icefinderscore")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-10, 10)
        ax.axhline(self._cutoff, color='red', alpha=0.5)
        ax.axhline(-self._cutoff, color='red', alpha=0.5)
        ax.axhline(0, color='#415a55', alpha=0.5)
        ax.plot(i_res_squared, icefinderscore,  color="Black", lw=1)

    def generate_F_plot(self, F, SigF, D2, ax=None):
        """Generate the F plot
        """
        plot = FPlot(self._ice_rings,
                     ax=ax,
                     filename=self.filename,
                     output_directory=self.output_directory,
                     cutoff=self._cutoff,
                     ylim=self.ylim,
                     dmin=self.dmin,
                     num_xticks=self.num_xticks,
                     filename_in_title=self.filename_in_title,
                     no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, F)

    def generate_SigF_plot(self, F, SigF, D2, ax=None):
        """Generate the SigF plot
        """
        plot = SigFPlot(self._ice_rings,
                        ax=ax,
                        filename=self.filename,
                        output_directory=self.output_directory,
                        cutoff=self._cutoff,
                        ylim=self.ylim,
                        dmin=self.dmin,
                        num_xticks=self.num_xticks,
                        filename_in_title=self.filename_in_title,
                        no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, SigF)

    def generate_FSigF_plot(self, F, SigF, D2, ax=None):
        """Generate the F / SigF plot
        """
        # D2, FSigF = list(zip(*[(d, f / sigf) for d, f, sigf in zip(D2, F, SigF) if sigf > 0]))
        arg_sig_valid = SigF > 0.
        FSigF = F[arg_sig_valid] / SigF[arg_sig_valid]
        D2 = D2[arg_sig_valid]
        plot = FoverSigFPlot(self._ice_rings,
                             ax=ax,
                             filename=self.filename,
                             output_directory=self.output_directory,
                             cutoff=self._cutoff,
                             ylim=self.ylim,
                             dmin=self.dmin,
                             num_xticks=self.num_xticks,
                             filename_in_title=self.filename_in_title,
                             no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, FSigF)
    
    def generate_B_plot(self, B, SigB, D2, ax=None):
        """Generate the B plot
        """
        plot = BPlot(
            self._ice_rings,
            ax=ax,
            filename=self.filename,
            output_directory=self.output_directory,
            cutoff=self._cutoff,
            ylim=self.ylim,
            dmin=self.dmin,
            num_xticks=self.num_xticks,
            filename_in_title=self.filename_in_title,
            no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, B)

    def generate_SigB_plot(self, B, SigB, D2, ax=None):
        """
        Generate the SigB plot

        """
        plot = SigBPlot(self._ice_rings,
                        ax=ax,
                        filename=self.filename,
                        output_directory=self.output_directory,
                        cutoff=self._cutoff,
                        ylim=self.ylim,
                        dmin=self.dmin,
                        num_xticks=self.num_xticks,
                        filename_in_title=self.filename_in_title,
                        no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, SigB)

    def generate_BSigB_plot(self, B, SigB, D2, ax=None):
        """
        Generate the B / SigB plot

        """
        D2, BSigB = list(zip(*[(d, b / sigb) for d, b, sigb in zip(D2, B, SigB) if sigb > 0]))
        plot = BoverSigBPlot(self._ice_rings,
                             ax=ax,
                             filename=self.filename,
                             output_directory=self.output_directory,
                             cutoff=self._cutoff,
                             ylim=self.ylim,
                             dmin=self.dmin,
                             num_xticks=self.num_xticks,
                             filename_in_title=self.filename_in_title,
                             no_automatic=self._no_automatic)
        plot.generate(self.icefinder_handle, D2, BSigB)

    def generate(self, icefinder_handle, nemo_handle_F=None, nemo_handle_I=None):
        """
        Generate all the plots

        """
        # Get the data
        i_data = icefinder_handle.iobs
        f_data = icefinder_handle.fobs
        b_data = icefinder_handle.background
        Verbose.generate_plot()

        if i_data is not None and i_data.size() > 0:
            print('Set of plots is generated with {0} intensities.'.format(i_data.size()))
            iobs = i_data.obs
            isigma = i_data.sigma
            reso_data = i_data.invresolsq()

            if self._single_figure:
                figure = plt.figure(figsize=(self._plotwidth, np.sqrt(2) * self._plotwidth))
                ax1 = figure.add_subplot(3, 1, 1, projection='scatter_density')
                ax2 = figure.add_subplot(3, 1, 2, projection='scatter_density')
                ax3 = figure.add_subplot(3, 1, 3, projection='scatter_density')
                self.generate_I_plot(iobs, isigma, reso_data, ax=ax1)
                self.generate_SigI_plot(iobs, isigma, reso_data, ax=ax2)
                self.generate_ISigI_plot(iobs, isigma, reso_data, ax=ax3)

                #  plot nemo
                if nemo_handle_I is not None:
                    generate_nemo_plot(ax1, nemo_handle_I, 'I')
                    generate_nemo_plot(ax2, nemo_handle_I, 'sigI')
                    generate_nemo_plot(ax3, nemo_handle_I, 'I_over_sigI')

                filename = os.path.join(self.output_directory, "intensities.png")
                plt.tight_layout()
                plt.savefig(filename, dpi=self._dpi, bbox_inches='tight')
                plt.clf()
                #print "single figure: %s" % (self.single_figure)

            if not self._no_individual_figures:
                fig1, ax1 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                fig2, ax2 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                fig3, ax3 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                # Generate intensity plots
                self.generate_I_plot(iobs, isigma, reso_data, ax=ax1)
                self.generate_SigI_plot(iobs, isigma, reso_data, ax=ax2)
                self.generate_ISigI_plot(iobs, isigma, reso_data, ax=ax3)

                #plot nemo
                if nemo_handle_I is not None:
                    generate_nemo_plot(ax1, nemo_handle_I, 'I')
                    generate_nemo_plot(ax2, nemo_handle_I, 'sigI')
                    generate_nemo_plot(ax3, nemo_handle_I, 'I_over_sigI')

                plt.tight_layout()
                fig1.savefig(os.path.join(self.output_directory, "I_plot.png"), dpi=self._dpi, bbox_inches='tight')
                fig2.savefig(os.path.join(self.output_directory, "SigI_plot.png"), dpi=self._dpi, bbox_inches='tight')
                fig3.savefig(os.path.join(self.output_directory, "IOverSigI_plot.png"), dpi=self._dpi, bbox_inches='tight')
                plt.clf()

            if self._score_figure:
                figure = plt.figure(figsize=(self._plotwidth, np.sqrt(2) * self._plotwidth))
                ax1 = figure.add_subplot(3, 1, 1, projection='scatter_density')
                ax2 = figure.add_subplot(3, 1, 2, projection='scatter_density')
                ax3 = figure.add_subplot(3, 1, 3, projection='scatter_density')
                self.generate_standardised_mean_plot(iobs, reso_data, ax=ax1)
                self.generate_icefinderscore_plot(iobs, reso_data, ax=ax2)
                self.generate_I_plot(iobs, isigma, reso_data, ax=ax3)
                filename = os.path.join(self.output_directory, "score.png")
                plt.tight_layout()
                plt.savefig(filename, dpi=self._dpi, bbox_inches='tight')
                plt.clf()
                print("\nGenerating a score figure: {0}\n".format(self._score_figure))

        if f_data is not None and f_data.size() > 0:
            print('Set of plots is generated with {0} amplitudes.'.format(f_data.size()))
            fobs = f_data.obs
            fsigma = f_data.sigma
            reso_data = f_data.invresolsq()

            if self._single_figure:
                figure = plt.figure(figsize=(self._plotwidth, np.sqrt(2) * self._plotwidth))
                ax1 = figure.add_subplot(3, 1, 1, projection='scatter_density')
                ax2 = figure.add_subplot(3, 1, 2, projection='scatter_density')
                ax3 = figure.add_subplot(3, 1, 3, projection='scatter_density')
                # Generate amplitude plots
                self.generate_F_plot(fobs, fsigma, reso_data, ax=ax1)
                self.generate_SigF_plot(fobs, fsigma, reso_data, ax=ax2)
                self.generate_FSigF_plot(fobs, fsigma, reso_data, ax=ax3)

                # plot nemo
                if nemo_handle_F is not None:
                    #nemo_handle_F.refl_data_prepare(self.icefinder_handle._reflection_data, 'FP')
                    #nemo_handle_F.cluster_detect(0)
                    generate_nemo_plot(ax1, nemo_handle_F, 'F')
                    generate_nemo_plot(ax2, nemo_handle_F, 'sigF')
                    generate_nemo_plot(ax3, nemo_handle_F, 'F_over_sigF')

                filename = os.path.join(self.output_directory, "amplitudes.png")
                plt.tight_layout()
                plt.savefig(filename, dpi=self._dpi, bbox_inches='tight')
                plt.clf()

            if not self._no_individual_figures:
                fig1, ax1 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                fig2, ax2 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                fig3, ax3 = plt.subplots(1, 1, subplot_kw={'projection': 'scatter_density'},
                                         figsize=(self._plotwidth, np.sqrt(2)/3 * self._plotwidth))
                # Generate intensity plots
                self.generate_F_plot(fobs, fsigma, reso_data, ax=ax1)
                self.generate_SigF_plot(fobs, fsigma, reso_data, ax=ax2)
                self.generate_FSigF_plot(fobs, fsigma, reso_data, ax=ax3)

                #plot nemo
                if nemo_handle_F is not None:
                    generate_nemo_plot(ax1, nemo_handle_F, 'F')
                    generate_nemo_plot(ax2, nemo_handle_F, 'sigF')
                    generate_nemo_plot(ax3, nemo_handle_F, 'F_over_sigF')

                plt.tight_layout()
                fig1.savefig(os.path.join(self.output_directory, "F_plot.png"), dpi=self._dpi, bbox_inches='tight')
                fig2.savefig(os.path.join(self.output_directory, "SigF_plot.png"), dpi=self._dpi, bbox_inches='tight')
                fig3.savefig(os.path.join(self.output_directory, "FOverSigF_plot.png"), dpi=self._dpi, bbox_inches='tight')
                plt.clf()

        if b_data is not None and b_data.size() > 0:
            print('Set of plots is generated with {0} backgrounds.'.format(b_data.size()))
            reso_data = [x.ires_squared for x in b_data]
            bobs    =  [y.obs for y in b_data]
            bsigma  =  [y.sigma for y in b_data]

            if self._single_figure:
                figure = plt.figure(figsize=(self._plotwidth, np.sqrt(2) * self._plotwidth))
                ax1 = figure.add_subplot(3, 1, 1)
                ax2 = figure.add_subplot(3, 1, 2)
                ax3 = figure.add_subplot(3, 1, 3)
                self.generate_B_plot(bobs, bsigma, reso_data, ax=ax1)
                self.generate_SigB_plot(bobs, bsigma, reso_data, ax=ax2)
                self.generate_BSigB_plot(bobs, bsigma, reso_data, ax=ax3)
                filename = os.path.join(self.output_directory, "backgrounds.png")
                plt.tight_layout()
                plt.savefig(filename, dpi=self._dpi, bbox_inches='tight')
                plt.clf()
                #print "single figure: %s" % (self.single_figure)

            if not self._no_individual_figures:
                ax1 = None
                ax2 = None
                ax3 = None
                # Generate intensity plots
                self.generate_B_plot(bobs, bsigma, reso_data, ax=ax1)
                self.generate_SigB_plot(bobs, bsigma, reso_data, ax=ax2)
                self.generate_BSigB_plot(bobs, bsigma, reso_data, ax=ax3)


def generate_nemo_plot(ax, nemo_handler, data_type):
    nemo_D2 = nemo_handler.get_nemo_D2()
    if data_type in ['I', 'F']:
        nemo_y = nemo_handler.get_nemo_data()
    elif data_type in ['sigI', 'sigF']:
        nemo_y = nemo_handler.get_nemo_sig()
    elif data_type in ['I_over_sigI', 'F_over_sigF']:
        nemo_y = nemo_handler.get_nemo_data_over_sig()
    ax.scatter(nemo_D2,
               nemo_y,
               8,
               zorder=3,
               color="#d60019",
               alpha=0.9,
               linewidth=0
               )

    # bbox_props = dict(boxstyle="round", fc=None, ec="#d60019", lw=1)
    # bbox = FancyBboxPatch((nemo_resolution.max()*2, nemo_y-0.5),
    #                       width=nemo_resolution.max() - nemo_resolution.min() + (nemo_resolution.max() - nemo_resolution.min())*0.05,
    #                       height=nemo_y.max() - nemo_y.min() + (nemo_y.max() - nemo_y.min())*0.05,
    #                       **bbox_props)
    # self.ax.add_patch(bbox)