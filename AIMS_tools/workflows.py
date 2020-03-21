import ase.io, ase.cell

from AIMS_tools.structuretools import structure
from AIMS_tools.preparation import prepare
from AIMS_tools.postprocessing import postprocess
from AIMS_tools.misc import *

from matplotlib import gridspec
from matplotlib.pyplot import Line2D

import pandas as pd

from collections import namedtuple


class k_convergence:
    def __init__(self, geometry, *args, **kwargs):
        self.geometry = geometry
        self.submit = kwargs.get("submit", None)
        self.klimits = kwargs.get("k_limits", [4, 24])

        if os.path.isdir(self.geometry):
            self.path = Path(self.geometry)
        elif os.path.isfile(self.geometry):
            self.path = Path(self.geometry).parent
        else:
            logging.error("Input geometry or directory not recognised.")

        if len(list(self.path.glob("kconv_*"))) != 0:
            self.mode = "evaluate"
        else:
            self.mode = "prepare"

        if self.mode == "prepare":
            logging.info("Preparing calculations for k-grid convergence ...")
            self.init_prepare()
            kgrids = self.setup_kgrids()
            self.setup_calcs(kgrids)
        elif self.mode == "evaluate":
            logging.info("Evaluating calculations for k-grid convergence ...")
            successes = list(self.check_success())
            self.results = self.evaluate_energies(successes)
            self.data = dict(
                zip(["first", "second", "third"], list(self.interpret_results()),)
            )
            self.plot_results()

    def init_prepare(self):
        logging.info("Preparing files for calculation ...")
        if self.path.joinpath("geometry.in").exists():
            logging.info(
                "Reading geometry from {} ...".format(
                    str(self.path.joinpath("geometry.in"))
                )
            )
            prep = prepare(str(self.path.joinpath("geometry.in")))
        else:
            try:
                prep = prepare(self.geometry)
            except:
                logging.critical("No structure found!")

        if self.path.joinpath("control.in").exists():
            logging.info(
                "Reading calculation setup from {} ...".format(
                    str(self.path.joinpath("control.in"))
                )
            )
        else:
            prep.setup_calculator()
            prep.adjust_control()
        self.prep = prep

    def setup_kgrids(self):
        kx, ky, kz = [
            1 / i for i in self.prep.structure.atoms.cell.lengths()
        ]  # reciprocal lengths
        ksteps = [i for i in np.linspace(0.005, 0.2, 1000)][::-1]
        ka = list(set([int(kx / j) for j in ksteps if int(kx / j) != 0]))
        kb = list(set([int(ky / j) for j in ksteps if int(ky / j) != 0]))
        ka = [i for i in ka if (i >= self.klimits[0]) and (i < self.klimits[1])]
        kb = [i for i in kb if (i >= self.klimits[0]) and (i < self.klimits[1])]
        if self.prep.structure.is_2d(self.prep.structure.atoms) == False:
            kc = list(set([int(kz / j) for j in ksteps if int(kz / j) != 0]))
            kc = [i for i in kc if (i >= self.klimits[0]) and (i < self.klimits[1])]
        else:
            kc = [1]
        maxlen = max([len(i) for i in [ka, kb, kc]])

        # this section assures that all k-lists have same length
        def find_closest(inp, fi):
            index = int(fi // 1)
            return 1 * inp[index]

        delta = (len(ka) - 1) / (maxlen - 1)
        ka = [find_closest(ka, i * delta) for i in range(maxlen)]
        delta = (len(kb) - 1) / (maxlen - 1)
        kb = [find_closest(kb, i * delta) for i in range(maxlen)]
        delta = (len(kc) - 1) / (maxlen - 1)
        kc = [find_closest(kc, i * delta) for i in range(maxlen)]

        try:
            kgrids = np.array([ka, kb, kc], dtype=int).T
        except:
            logging.error("K-grids were not set up with equal lengths.")
        return kgrids

    def setup_calcs(self, kgrids):
        import fileinput

        for i, j, k in kgrids:
            name = "kconv_{:02d}_{:02d}_{:02d}".format(i, j, k)
            self.path.joinpath(name).mkdir(exist_ok=True)
            shutil.copy(
                str(self.path.joinpath("control.in")), str(self.path.joinpath(name))
            )
            shutil.copy(
                str(self.path.joinpath("geometry.in")), str(self.path.joinpath(name))
            )
            if self.submit != None:
                shutil.copy(
                    str(self.path.joinpath(self.submit)), str(self.path.joinpath(name))
                )

            for line in fileinput.input(
                str(self.path.joinpath(name, "control.in")), inplace=True
            ):
                if "k_grid" in line:
                    line = "k_grid \t \t {} {} {}\n".format(i, j, k)
                sys.stdout.write(line)

    def check_success(self):
        logger = logging.getLogger()
        logger.disabled = True
        kconvs = [str(j) for j in list(self.path.glob("kconv_*"))]
        successes = []
        if len(kconvs) != 0:
            for job in kconvs:
                try:
                    pp = postprocess(job)
                    successes.append(pp.success)
                except:
                    successes.append(False)
        logger.disabled = False
        for p, m in dict(zip(kconvs, successes)).items():
            suc = "SUCCESS" if m == True else "FAILED"
            logging.info("{} : {}".format(p, suc))
            if m == True:
                yield p

    def evaluate_energies(self, successes):
        data = {}
        logger = logging.getLogger()
        logger.disabled = True
        for suc in successes:
            pp = postprocess(suc)
            k_dens = (
                pp.k_grid[0] * pp.k_grid[1] * pp.k_grid[2]
            ) / pp.structure.atoms.cell.volume
            data["{}_{}_{}".format(*pp.k_grid)] = {
                "kdens [nkpoints/Angström^3]": k_dens,
                "total_energy [eV/atom]": pp.total_energy / len(pp.structure.atoms),
                "gap [eV]": pp.band_gap,
            }
        logger.disabled = False
        data = pd.DataFrame(data).transpose()
        return data

    def interpret_results(self):
        results = self.results
        thresh = namedtuple("convergence", ["grid", "energy", "kdensity", "gap"])
        ref = results["total_energy [eV/atom]"].iloc[-1]
        for lim in [1e-4, 1e-5, 1e-6]:
            conv = results[(results["total_energy [eV/atom]"] - ref).abs() < lim]
            gridconv = conv.index[0].replace("_", "x")
            econv = conv.loc[conv.index[0]]["total_energy [eV/atom]"]
            densconv = conv.loc[conv.index[0]]["kdens [nkpoints/Angström^3]"]
            gapconv = conv.loc[conv.index[0]]["gap [eV]"]
            yield thresh(gridconv, econv, densconv, gapconv)

    def analyze_gaps(self):
        results = self.results
        bins = results["gap [eV]"].round(2).value_counts()
        if bins.max() > 2:
            reps = results[
                results["gap [eV]"].round(2) == bins[bins == bins.max()].index[0]
            ]
            magics = []
            for grid in list(reps.index):
                magics.append(grid)
                for stored in self.data.values():
                    if grid == stored.grid.replace("x", "_"):
                        logging.info(
                            "Total energy and band gap appear converged with a k-grid of {grid}.".format(
                                grid=grid.replace("_", "x")
                            )
                        )

    def _plot_energy(self, fig, axes):
        results = self.results
        ref = results["total_energy [eV/atom]"].iloc[-1]
        x = np.array(results["kdens [nkpoints/Angström^3]"]) ** (1 / 3)
        y = np.array(results["total_energy [eV/atom]"]) - ref
        axes.scatter(x, y, color="black", alpha=0.8, facecolor="none")
        axes.plot(x, y, color="blue")

        colors = {"first": "gold", "second": "orange", "third": "crimson"}
        i = 0
        for thresh, entries in self.data.items():
            axes.axvline(
                x=entries.kdensity ** (1 / 3),
                color=colors[thresh],
                alpha=0.85,
                linestyle=":",
            )
            axes.annotate(
                entries.grid,
                xy=(entries.kdensity ** (1 / 3), entries.energy - ref),
                xycoords="data",
                xytext=(0.8, 0.90 - i * 0.1),
                textcoords="axes fraction",
                arrowprops=dict(
                    facecolor=colors[thresh],
                    edgecolor="none",
                    width=1,
                    shrink=0.05,
                    headwidth=5,
                    headlength=5,
                ),
                horizontalalignment="right",
                verticalalignment="top",
            )
            i += 1

        axes.set_xlim([np.min(x) * 0.95, np.max(x) * 1.05])
        axes.set_ylim([-0.0005, 0.0005])
        handles = []
        handles.append(Line2D([0], [0], color="gold", label="< 1e-4 eV/atom", lw=1.5))
        handles.append(Line2D([0], [0], color="orange", label="< 1e-5 eV/atom", lw=1.5))
        handles.append(Line2D([0], [0], color="red", label="< 1e-6 eV/atom", lw=1.5))
        axes.legend(handles=handles, loc="lower right")
        axes.set_xlabel(r"$(nkpoints/volume)^{1/3}$ [1/Angström]")
        axes.set_ylabel(r"$E(k) - E(k_{max})$ [eV /atom]")
        axes.set_title("total energy convergence")
        return axes

    def _plot_gaps(self, fig, axes):
        x = np.array(self.results["kdens [nkpoints/Angström^3]"]) ** (1 / 3)
        y = np.array(self.results["gap [eV]"])
        axes.scatter(x, y, color="black", alpha=0.8, facecolor="none")
        axes.plot(x, y, color="blue")

        axes.set_xlabel(r"$(nkpoints/volume)^{1/3}$ [1/Angström]")
        axes.set_ylabel(r"Band gap [eV]")
        axes.set_title("band gap convergence")
        return axes

    def plot_results(self):
        fig = plt.figure(constrained_layout=True, figsize=(8, 4),)
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0])
        ax2 = fig.add_subplot(spec[1])

        ax1 = self._plot_energy(fig, ax1)
        ax2 = self._plot_gaps(fig, ax2)

        plt.show()
