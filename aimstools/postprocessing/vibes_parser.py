from aimstools.misc import *
from aimstools.structuretools import Structure

from pathlib import Path

import re, os

from collections import namedtuple

testdir = Path("/mnt/c/Users/rkempt/Dropbox/Repositories/aimstools/tests/vibes_test")


class FHIVibesParser:
    def __init__(self, directory) -> None:
        self.maindir = self._find_main_directory(directory)
        self.structure = Structure(self.maindir.joinpath("geometry.in"))
        relaxdir = self.maindir.joinpath("relaxation")

        self.tasks = self._get_tasks()
        self._check_output()

    def _find_main_directory(self, directory):
        d = Path(directory).absolute()
        assert d.exists(), "Path {} does not exist.".format(str(d))
        if str(d.parts[-1]) in ["relaxation", "phonopy"]:
            maindir = d.parent
        else:
            maindir = d
        logger.info("Found FHI-vibes main directory: {}".format(str(maindir)))
        assert maindir.joinpath(
            "geometry.in"
        ).exists(), (
            "FHI-vibes main directory does not seem to contain geometry.in file."
        )
        return maindir

    def _get_tasks(self):
        md = self.maindir
        tasks = set()
        if md.joinpath("relaxation.in").exists():
            tasks.add("relaxation")
        if md.joinpath("phonopy.in").exists():
            tasks.add("phonopy")

        return tasks

    def _check_output(self):
        if "relaxation" in self.tasks:
            assert self.maindir.joinpath(
                "log.relaxation"
            ).exists(), "FHI-vibes main directory does not contain log.relaxation file."
            with open(self.maindir.joinpath("log.relaxation"), "r") as file:
                lines = [l.strip() for l in file.readlines()]
            is_converged = False
            for l in lines:
                if "Relaxation converged." in l:
                    is_converged = True
            logger.info("FHI-vibes relaxation converged: {}".format(is_converged))
            self._relaxation_converged = True
        if "phonopy" in self.tasks:
            assert self.maindir.joinpath(
                "log.phonopy"
            ).exists(), "FHI-vibes main directory does not contain log.phonopy file."
            with open(self.maindir.joinpath("log.phonopy"), "r") as file:
                is_done = [l.strip() for l in file.readlines()][-1]
            is_done = bool(re.search(r"\[vibes\]\s+done\.", is_done))
            logger.info("FHI-vibes phonon calculation finished: {}".format(is_done))
            self._phonons_converged = True
            if not is_done:
                self.tasks.remove("phonopy")

    def _get_phonopyfiles(self):
        phonondir = self.maindir.joinpath("phonopy")
        vibes_traj = phonondir.joinpath("trajectory.son")
        assert vibes_traj.exists(), "FHI-vibes trajectory.son not found."

        outputdir = phonondir.joinpath("output")
        assert (
            outputdir.exists()
        ), "FHI-vibes phonopy/output directory does not exist. You have to run `vibes output phonopy phonopy/trajectory.son --full`."
        self.vibes_traj = vibes_traj
        self.outputdir = outputdir
