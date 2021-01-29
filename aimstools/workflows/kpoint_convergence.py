from pandas.core.groupby import grouper
from aimstools.misc import *

from aimstools.structuretools import Structure
from aimstools.preparation import FHIAimsSetup
from aimstools.preparation.utilities import monkhorstpack2kptdensity
from aimstools.postprocessing import FHIAimsOutputReader, FHIAimsControlReader

from matplotlib import gridspec
from matplotlib.pyplot import Line2D

import pandas as pd
import numpy as np
from pathlib import Path

from collections import namedtuple

from ase.calculators.calculator import kptdensity2monkhorstpack

import shutil, fileinput


class KPointConvergence:
    """Workflow for k-point convergence.

    Automatically sets up the calculations needed to check for k-point convergence and analyses the results.

    Note:
        This workflow is wrapped in the utility `aims_workflow converge_kpoints`.

    Args:
        geometry (str): Path to input geometry for preparation mode.
        result_dir (str): Path to result directory for evaluation mode.
        **preparation_kwargs (dict): Keyword arguments are passed to :class:`~aimstools.preparation.aims_setup.FHIAimsSetup`.
    """

    def __init__(self, geometryfile=None, result_dir=None, **preparation_kwargs):
        dirname = "aimstools_kpoint_convergence"
        # setting up file paths
        assert (geometryfile == None) or (
            result_dir == None
        ), "You have to specify either a geometry input for preparation or a result directory for evaluation."
        if (str(Path(geometryfile).absolute().parts[-1]) == dirname) and (
            result_dir == None
        ):
            result_dir = Path(geometryfile)
            geometryfile = None
        if Path().cwd().joinpath(dirname).exists() and geometryfile != None:
            raise Exception(
                "Result directory already exists in current working directory. Not attempting an overwrite."
            )
        elif Path().cwd().joinpath(dirname).exists() and geometryfile == None:
            logger.info("Result directory found in current working directory.")
            result_dir = Path().cwd().joinpath(dirname)

        mode = None
        # setting up the mode
        if geometryfile != None:
            self.maindir = Path().cwd()
            self.geometryfile = geometryfile
            mode = "prepare"
            logger.info(
                "Entering preparation mode with geometryfile {} .".format(
                    str(geometryfile)
                )
            )
        elif result_dir != None:
            mode = "evaluate"
            logger.info("Entering evaluation mode...")
            p = Path(result_dir).absolute()
            if str(p.parts[-1]) == dirname:
                self.maindir = Path(result_dir).parent
            elif Path(result_dir).is_dir():
                assert (
                    Path(result_dir).joinpath(dirname).exists()
                ), "Result directory not found."
                self.maindir = Path(result_dir)
            else:
                logger.error("Something went wrong.")
                raise Exception("Input not recognized.")
        else:
            logger.error("Something went horribly wrong.")
            raise Exception("Input not recognized.")

        self.dirname = dirname
        self.mode = mode

        if mode == "prepare":
            self.prepare_k_point_convergence(**preparation_kwargs)
        elif mode == "evaluate":
            self.results = self.evaluate_results()
            logger.info("Results are shown below:")
            self.log_results()
            self.thresholds = list(self.interpret_results())

    def prepare_k_point_convergence(self, **preparation_kwargs):
        """ Prepares files for calculations in directory `aimstools_kpoint_convergence/`. """
        logger.info("Preparing files for calculation ...")
        prep_kwargs = preparation_kwargs
        _ = prep_kwargs.pop("k_density", 0)
        _ = prep_kwargs.pop("k_grid", None)

        setup = FHIAimsSetup(self.geometryfile, **prep_kwargs)
        if setup.dirpath.joinpath("geometry.in").exists():
            logger.info(
                "Using geometry from {} .".format(
                    str(setup.dirpath.joinpath("geometry.in"))
                )
            )
        else:
            setup.setup_geometry()
        if setup.dirpath.joinpath("control.in").exists():
            logger.info(
                "Using calculation setup from {} .".format(
                    str(setup.dirpath.joinpath("control.in"))
                )
            )
        else:
            setup.setup_control()
        if setup.dirpath.joinpath("submit.sh").exists():
            logger.info(
                "Using submission script from {} .".format(
                    str(setup.dirpath.joinpath("submit.sh"))
                )
            )
        else:
            setup.write_submission_file()
        geometry = setup.dirpath.joinpath("geometry.in")
        control = setup.dirpath.joinpath("control.in")
        submit = setup.dirpath.joinpath("submit.sh")
        files_to_copy = (geometry, control, submit)
        k_grids = np.array(
            [
                kptdensity2monkhorstpack(setup.structure, kptdensity=i, even=False)
                for i in range(1, 12)
            ],
            dtype=int,
        )
        k_grids = np.unique(k_grids, axis=0)
        logger.info(
            "Preparing {} calculations for k-point convergence.".format(len(k_grids))
        )
        for i, k in enumerate(k_grids):
            kstring = "{}x{}x{}".format(*k)
            targetdir = setup.dirpath.joinpath(self.dirname).joinpath(kstring)
            targetdir.mkdir(parents=True, exist_ok=True)
            for n in files_to_copy:
                shutil.copy(str(n), str(targetdir.joinpath(n.parts[-1])))
            for line in fileinput.input(
                str(targetdir.joinpath("control.in")), inplace=True
            ):
                if "k_grid" in line:
                    print("k_grid \t \t {} {} {}\n".format(*k), end="")
                else:
                    print(line, end="")

    def evaluate_results(self):
        """ Evaluates results from finished calculations. """
        data = {}
        dirs = self.maindir.joinpath(self.dirname).glob("*")
        dirs = [d for d in dirs if d.is_dir()]
        outs = [FHIAimsOutputReader(d) for d in dirs]
        for i, result in enumerate(outs):
            k_density = monkhorstpack2kptdensity(
                result.structure, result.control.k_grid
            )
            bg = (
                result.bandgap.soc
                if result.control.include_spin_orbit
                else result.bandgap.scalar
            )
            te = result.total_energy / len(result.structure)
            data[i] = {
                "k-grid": "{}x{}x{}".format(*result.control.k_grid),
                "k-point density [points/Angström]": k_density,
                "total energy [eV/atom]": te,
                "band gap [eV]": bg,
                "number of SCF cycles": result.nscf_steps,
                "converged": result.is_converged,
            }
        data = pd.DataFrame(data).transpose()
        data.sort_values(
            "k-point density [points/Angström]", ascending=True, inplace=True
        )
        data.reset_index(inplace=True, drop=True)
        return data

    def interpret_results(self):
        """ Interprets total energy convergence with respect to commonly used accuracy thresholds."""
        results = self.results
        results = results[results["converged"] == True]
        thresh = namedtuple(
            "convergence", ["limit", "grid", "energy", "kdensity", "bandgap"]
        )
        ref = results["total energy [eV/atom]"].iloc[-1]
        for lim in [1e-4, 1e-5, 1e-6]:
            conv = results[(results["total energy [eV/atom]"] - ref).abs() < lim]
            gridconv = conv.loc[conv.index[0]]["k-grid"]
            econv = conv.loc[conv.index[0]]["total energy [eV/atom]"]
            densconv = conv.loc[conv.index[0]]["k-point density [points/Angström]"]
            gapconv = conv.loc[conv.index[0]]["band gap [eV]"]
            logger.info(
                "The k-kgrid is converged within {: 1.1E} eV/atom for a grid of {} after {} SCF cycles.".format(
                    lim, gridconv, conv.loc[conv.index[0]]["number of SCF cycles"]
                )
            )
            if conv.index[0] == results.index[-1]:
                logger.warning("You might need to check denser k-grids.")
            yield thresh(lim, gridconv, econv, densconv, gapconv)

    def log_results(self):
        """ Logs results as table formatted with rich. """
        try:
            from rich.console import Console
            from rich.table import Table

            results = self.results.copy()
            results["total energy [eV/atom]"] = [
                "{:.6f}".format(k) for k in results["total energy [eV/atom]"]
            ]
            results["band gap [eV]"] = [
                "{:.2f}".format(k) for k in results["band gap [eV]"]
            ]

            console = Console()

            table = Table(show_header=True)
            for col in results.columns:
                table.add_column(col)
            for index, row in results.iterrows():
                r = list(row.astype(str))
                table.add_row(*r)

            console.print(table)
        except Exception as expt:
            print(expt)
            print(self.results)

    def _plot_energy(self, axes):
        results = self.results
        ref = results["total energy [eV/atom]"].iloc[-1]
        x = np.array(results["k-point density [points/Angström]"])
        y = (np.array(results["total energy [eV/atom]"]) - ref) * 1000
        axes.plot(x, y, color="royalblue", zorder=1)
        axes.scatter(x, y, facecolor="white", edgecolor="royalblue", alpha=1, zorder=2)
        axes.axhline(0, linestyle="--", color="lightgray", zorder=0, alpha=0.5)

        colors = ["gold", "orange", "crimson"]
        for i, thresh in enumerate(self.thresholds):
            axes.axvline(
                x=thresh.kdensity,
                color=colors[i],
                alpha=0.85,
                linestyle=":",
            )

        axes.set_xlim([np.min(x) * 0.95, np.max(x) * 1.05])
        axes.set_ylim([-50, 50])
        axes.set_xlabel(r"k-point density [points/Angström]")
        axes.set_ylabel(r"$E(k) - E(k_{max})$ [meV /atom]")
        axes.set_title("total energy convergence")
        return axes

    def _plot_gaps(self, axes):
        x = np.array(self.results["k-point density [points/Angström]"])
        y = np.array(self.results["band gap [eV]"])
        axes.plot(x, y, color="royalblue", zorder=1)
        axes.scatter(
            x,
            y,
            facecolors="royalblue",
            edgecolor="royalblue",
            zorder=2,
        )
        colors = ["gold", "orange", "crimson"]
        for i, thresh in enumerate(self.thresholds):
            axes.axvline(
                x=thresh.kdensity,
                color=colors[i],
                alpha=0.85,
                linestyle=":",
            )

        colors = ["gold", "orange", "crimson"]
        axes.set_ylim([np.min(y) - 0.1, np.max(y) + 0.1])
        axes.set_xlabel(r"k-point density [points/Angström]")
        axes.set_ylabel(r"band gap [eV]")
        axes.set_title("band gap convergence")
        return axes

    def plot_results(self, show=True):
        """Plots total energy convergence and band gap convergence.

        Args:
            show (bool): Shows plot via matplotlib if True, else saves to file kconv.png .

        """
        # this is the perfect job for plotly
        fig = plt.figure(
            constrained_layout=True,
            figsize=(8, 4),
        )
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0])
        ax2 = fig.add_subplot(spec[1])

        ax1 = self._plot_energy(ax1)
        ax2 = self._plot_gaps(ax2)
        handles = []
        handles.append(Line2D([0], [0], color="gold", label="< 1e-4 eV/atom", lw=1.5))
        handles.append(Line2D([0], [0], color="orange", label="< 1e-5 eV/atom", lw=1.5))
        handles.append(Line2D([0], [0], color="red", label="< 1e-6 eV/atom", lw=1.5))
        ax1.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.8,
        )
        if show:
            plt.show()
        else:
            plt.savefig(
                str(self.maindir.joinpath("kconv.png")),
                dpi=300,
                transparent=False,
                facecolor="white",
                bbox_inches="tight",
            )
            logger.info("Results have been saved to kconv.png.")

    def plot_interactive(self):
        """Plots results interactively via plotly. Requires a browser."""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ModuleNotFoundError as err:
            logger.error(err)
            raise Exception(
                "You need to install plotly to view the plot interactively."
            )

        df = self.results.copy()
        df["total energy [eV/atom]"] = df["total energy [eV/atom]"] - np.min(
            df["total energy [eV/atom]"]
        )

        texts = [
            "k-grid: {}<br>band gap [eV]: {:.4f}".format(i, j)
            for i, j in zip(df["k-grid"].astype(str), df["band gap [eV]"])
        ]
        trace = go.Scatter(
            x=df["k-point density [points/Angström]"],
            y=df["total energy [eV/atom]"],
            marker=dict(
                size=10,
                line=dict(width=2),
            ),
            mode="lines+markers",
            text=texts,
        )

        shapes = list()
        colors = ["Orange", "Darkorange", "Crimson"]
        for n, i in enumerate(self.thresholds):
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": i.kdensity,
                    "y0": 0,
                    "x1": i.kdensity,
                    "y1": 1,
                    "line": dict(
                        color=colors[n],
                        width=3,
                    ),
                    "name": i.limit,
                }
            )
        layout = go.Layout(shapes=shapes)
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()