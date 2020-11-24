from aimstools.misc import *
from aimstools.preparation import FHIAimsSetup

from aimstools.preparation.templates import (
    vibes_phonopy_template,
    vibes_relaxation_template,
    vibes_slurm_template,
)

from aimstools.preparation.utilities import monkhorstpack2kptdensity


from pathlib import Path
import os


class FHIVibesSetup(FHIAimsSetup):
    """Sets up calculation with FHI-vibes."""

    def __init__(self, geometryfile, **kwargs) -> None:
        super().__init__(geometryfile, **kwargs)
        self.basis = kwargs.get("basis", "tight")

    def setup_relaxation(self, overwrite=False):
        logger.info("Setting up relaxation with FHI-vibes.")
        xc = self.aseargs["xc"]
        basis = self.basis
        tier = self.aseargs["tier"]
        k_grid = self.aseargs["k_grid"]
        k_density = monkhorstpack2kptdensity(self.structure, k_grid)

        template = vibes_relaxation_template.format(
            xc=xc, basis=basis, tier=tier, kptdensity=k_density
        )
        relaxationfile = self.dirpath.joinpath("relaxation.in")
        if relaxationfile.exists() and (overwrite == False):
            logger.warning(
                "File relaxation.in is already existing. Set overwrite=True to force overwrite."
            )
        if not relaxationfile.exists() or overwrite:
            logger.info("Writing file {} ...".format(relaxationfile))
            with open(self.dirpath.joinpath("relaxation.in"), "w") as file:
                file.write(template)

    def setup_phonopy(self, overwrite=False):
        logger.info("Setting up phonon calculation with FHI-vibes.")
        xc = self.aseargs["xc"]
        basis = self.basis
        tier = self.aseargs["tier"]
        k_grid = self.aseargs["k_grid"]
        k_density = monkhorstpack2kptdensity(self.structure, k_grid)

        template = vibes_phonopy_template.format(
            xc=xc, basis=basis, tier=tier, kptdensity=k_density
        )
        phonopyfile = self.dirpath.joinpath("phonopy.in")
        if phonopyfile.exists() and (overwrite == False):
            logger.warning(
                "File phonopy.in already exists. Set overwrite=True to force overwrite."
            )
        if not phonopyfile.exists() or overwrite:
            logger.info("Writing file {} ...".format(phonopyfile))
            with open(self.dirpath.joinpath("phonopy.in"), "w") as file:
                file.write(template)

    def write_submission_file(self, task, overwrite=False):
        assert task in ["relaxation", "phonopy"], "Task for FHI-vibes not recognized."
        templatefile = os.getenv("VIBES_SLURM_TEMPLATE")
        if templatefile == None:
            template = vibes_slurm_template
        else:
            with open(templatefile, "r") as file:
                template = file.read()
        if task == "relaxation":
            jobname = self.structure.atoms.get_chemical_formula() + "_relax"
        elif task == "phonopy":
            jobname = self.structure.atoms.get_chemical_formula() + "_phon"
        template = template.format(jobname=jobname, task=task)
        submitfile = self.dirpath.joinpath("submit.sh")
        if submitfile.exists() and (overwrite == False):
            logger.warning(
                "File submit.sh already exists. Set overwrite=True to force overwrite."
            )
        if not submitfile.exists() or overwrite:
            logger.info("Writing {} ...".format(submitfile))
            with open(submitfile, "w") as file:
                file.write(template)
