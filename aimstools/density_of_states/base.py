from aimstools.misc import *
from aimstools.postprocessing import FHIAimsOutputReader
from aimstools.bandstructures.base import get_energy_reference

import re

from collections import namedtuple


class DOSBaseClass(FHIAimsOutputReader):
    def __init__(self, outputfile):
        super().__init__(outputfile)
        assert self.is_converged, "Calculation did not converge."
        tasks = {x for x in self.control["tasks"] if "dos" in x}
        accepted_tasks = set(
            [
                "total dos",
                "total dos tetrahedron",
                "atom-projected dos",
                "atom-projected dos tetrahedron",
                "species-projected dos",
                "species-projected dos tetrahedron",
            ]
        )
        assert any(x in accepted_tasks for x in tasks), "DOS task not accepted."
        self.tasks = tasks
        self.task = None
        self._energy_reference = "not specified"

    def __get_total_dos_files_scalar(self):
        if self.control["include_spin_orbit"]:
            regex = re.compile(r"KS_DOS_total\.dat\.no_soc")
        else:
            regex = re.compile(r"KS_DOS_total\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_total_dos_files_soc(self):
        regex = re.compile(r"KS_DOS_total\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_total_dos_tetrahedron_files_scalar(self):
        if self.control["include_spin_orbit"]:
            regex = re.compile(r"KS_DOS_total_tetrahedron\.dat\.no_soc")
        else:
            regex = re.compile(r"KS_DOS_total_tetrahedron\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_total_dos_tetrahedron_files_soc(self):
        regex = re.compile(r"KS_DOS_total_tetrahedron\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_atom_proj_dos_files_scalar(self, spin="none"):
        spin = "" if spin == "none" else "spin_" + spin
        if self.control["include_spin_orbit"]:
            regex = re.compile(
                r"atom_proj(ected)?_dos_" + spin + r"[A-Z]([a-z])?\d{4}\.dat\.no_soc"
            )
        else:
            regex = re.compile(
                r"atom_proj(ected)?_dos_" + spin + r"[A-Z]([a-z])?\d{4}\.dat"
            )
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_atom_proj_dos_files_soc(self):
        # Spin channels are ill-defined with SOC.
        regex = re.compile(r"atom_proj(ected)?_dos_[A-Z]([a-z])?\d{4}\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_atom_proj_dos_tetrahedron_files_scalar(self, spin="none"):
        # outdated with new version
        # assert (
        #    spin == "none"
        #), "Tetrahedron DOS with open shell is currently not implemented in FHI-aims."
        spin = "" if spin == "none" else "spin_" + spin
        if self.control["include_spin_orbit"]:
            regex = re.compile(
                r"atom_proj(ected)?_dos_tetrahedron_[A-Z]([a-z])?\d{4}\.dat\.no_soc"
            )
        else:
            regex = re.compile(
                r"atom_proj(ected)?_dos_tetrahedron_" + spin + r"[A-Z]([a-z])?\d{4}\.dat"
            )
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_atom_proj_dos_tetrahedron_files_soc(self):
        # Spin channels are ill-defined with SOC.
        regex = re.compile(
            r"atom_proj(ected)?_dos_tetrahedron_[A-Z]([a-z])?\d{4}\.dat$"
        )
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_species_proj_dos_files_scalar(self, spin="none"):
        spin = "" if spin == "none" else "_spin_" + spin
        if self.control["include_spin_orbit"]:
            regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos" + spin + r"\.dat\.no_soc")
        else:
            regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos" + spin + r"\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_species_proj_dos_files_soc(self):
        # Spin channels are ill-defined with SOC.
        regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_species_proj_dos_tetrahedron_files_scalar(self, spin="none"):
        # outdated with new version
        # assert (
        #    spin == "none"
        #), "Tetrahedron DOS with open shell is currently not implemented in FHI-aims."
        spin = "" if spin == "none" else "_spin_" + spin
        if self.control["include_spin_orbit"]:
            regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos_tetrahedron\.dat\.no_soc")
        else:
            regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos_tetrahedron" + spin + r"\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def __get_species_proj_dos_tetrahedron_files_soc(self):
        # Spin channels are ill-defined with SOC.
        regex = re.compile(r"[A-Z]([a-z])?_l_proj_dos_tetrahedron\.dat$")
        dosfiles = list(self.outputdir.glob("*.dat*"))
        dosfiles = [j for j in dosfiles if bool(regex.match(str(j.parts[-1])))]
        return dosfiles

    def get_dos_files_old(self, spin="none", soc=False):
        assert spin in ("none", "up", "dn"), "Spin keyword not recognized."
        total_dos_files = atom_proj_dos_files = species_proj_dos_files = None
        if "total dos" in self.tasks:
            if not soc:
                total_dos_files = self.__get_total_dos_files_scalar()
            if soc:
                total_dos_files = self.__get_total_dos_files_soc()
        if "atom-projected dos" in self.tasks:
            if not soc:
                atom_proj_dos_files = self.__get_atom_proj_dos_files_scalar(spin=spin)
            if soc:
                atom_proj_dos_files = self.__get_atom_proj_dos_files_soc()
        if "species-projected dos" in self.tasks:
            if not soc:
                species_proj_dos_files = self.__get_species_proj_dos_files_scalar(
                    spin=spin
                )
            if soc:
                species_proj_dos_files = self.__get_species_proj_dos_files_soc()
        d = namedtuple("dosfiles", ["total_dos", "atom_proj_dos", "species_proj_dos"])
        return d(total_dos_files, atom_proj_dos_files, species_proj_dos_files)

    def get_dos_files_tetrahedron(self, spin="none", soc=False):
        assert spin in ("none", "up", "dn"), "Spin keyword not recognized."
        total_dos_files = atom_proj_dos_files = species_proj_dos_files = None
        if "total dos tetrahedron" in self.tasks:
            if not soc:
                total_dos_files = self.__get_total_dos_tetrahedron_files_scalar()
            if soc:
                total_dos_files = self.__get_total_dos_tetrahedron_files_soc()
        if "atom-projected dos tetrahedron" in self.tasks:
            if not soc:
                atom_proj_dos_files = self.__get_atom_proj_dos_tetrahedron_files_scalar(
                    spin=spin
                )
            if soc:
                atom_proj_dos_files = self.__get_atom_proj_dos_tetrahedron_files_soc()
        if "species-projected dos tetrahedron" in self.tasks:
            if not soc:
                species_proj_dos_files = self.__get_species_proj_dos_tetrahedron_files_scalar(
                    spin=spin
                )
            if soc:
                species_proj_dos_files = (
                    self.__get_species_proj_dos_tetrahedron_files_soc()
                )
        d = namedtuple("dosfiles", ["total_dos", "atom_proj_dos", "species_proj_dos"])
        return d(total_dos_files, atom_proj_dos_files, species_proj_dos_files)

    def get_dos_files(self, spin="none", soc=False):
        """Looks for dos files according to the tasks specified in control.in.

        By default, the tetrahedron method will be preferred other the old implementation.

        """
        new = any(x for x in self.tasks if "tetrahedron" in x)
        if spin in [None, False, "none", "None", 0]:
            spin = "none"
        elif spin in ["up", "UP", "+", 1]:
            spin = "up"
        elif spin in ["dn", "down", "-", 2]:
            spin = "dn"
        if soc and (spin != "none"):
            logger.warning("Collinear spin and SOC are physically not meaningful.")
        if spin != "none":
            assert self.control["spin"] in [
                "collinear"
            ], "Collinear spin was not specified in control.in ."
        if soc:
            assert self.control[
                "include_spin_orbit"
            ], "SOC was not specified in control.in ."
        if new:
            dosfiles = self.get_dos_files_tetrahedron(spin=spin, soc=soc)
        else:
            dosfiles = self.get_dos_files_old(spin=spin, soc=soc)
        return dosfiles

    @property
    def energy_reference(self):
        return self._energy_reference

    def set_energy_reference(self, reference, soc=False):
        self._energy_reference = get_energy_reference(
            reference,
            self.fermi_level,
            self.band_extrema,
            self.work_function,
            self.control["output_level"],
            soc=soc,
        )

    def spin2index(self, spin):
        if spin in [None, "none", "down", "dn", 0]:
            spin = 0
        elif spin in ["up", "UP", 1, 2]:
            spin = 1
        return spin
