import ase.io, ase.cell

from AIMS_tools.structuretools import structure
from AIMS_tools.preparation import prepare
from AIMS_tools.postprocessing import postprocess
from AIMS_tools.misc import *

import pandas as pd


class k_convergence:
    def __init__(self, *args, **kwargs):
        self.geometry = kwargs.get("input")
        self.submit = kwargs.get("submit", None)
        self.klimits = kwargs.get("k_limits", [4, 30])

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
            logging.mode("Preparing calculations for k-grid convergence ...")
            self.init_prepare()
            kgrids = self.setup_kgrids()
            self.setup_calcs(kgrids)
        elif self.mode == "evaluate":
            logging.info("Evaluating calculations for k-grid convergence ...")
            successes = list(self.check_success())
            results = self.evaluate_energies(successes)

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

    def plot_results(self, results):
        fig = plt.figure(figsize=(4, 4))
        axes = plt.gca()
        x = np.array(results["kdens [nkpoints/Angström^3]"])[1:] ** (1 / 3)
        y = np.array(results["total_energy [eV/atom]"])
        y = np.diff(y)
        x = x[np.abs(y) < 5e-4]
        y = y[np.abs(y) < 5e-4]

        first = np.min(x[np.abs(y) < 1e-4])
        second = np.min(x[np.abs(y) < 1e-5])
        third = np.min(x[np.abs(y) < 1e-6])

        yrange = (-5e-4, 5e-4)
        axes.fill_between(x, -1e-4, 1e-4, color="yellow", alpha=0.2)
        axes.fill_between(x, -1e-5, 1e-5, color="orange", alpha=0.25)
        axes.fill_between(x, -1e-6, 1e-6, color="red", alpha=0.3)
        axes.scatter(x, y, color="black", alpha=0.8, facecolor="none")
        axes.plot(x, y, color="blue")

        axes.axvline(
            x=first, color="yellow", alpha=0.5, linestyle=":",
        )
        axes.axvline(
            x=second, color="orange", alpha=0.5, linestyle=":",
        )
        axes.axvline(
            x=third, color="red", alpha=0.5, linestyle=":",
        )
        axes.set_xlim([np.min(x), np.max(x)])
        axes.set_ylim(yrange)
        print(
            results.loc[
                np.abs(results["kdens [nkpoints/Angström^3]"] - (third ** 3)) < 1e-5
            ]
        )

        plt.show()

