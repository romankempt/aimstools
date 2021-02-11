from aimstools.misc import *
from aimstools.structuretools import Structure

from pathlib import Path

import re, os

from collections import namedtuple


class FHIAimsControlReader(dict):
    """Parses information from control.in file.

    Args:
        str: Path to control.in or directory with control.in file.
    """

    def __init__(self, controlfile) -> None:
        super(FHIAimsControlReader, self).__init__()
        controlfile = Path(controlfile)
        if controlfile.is_dir():
            controlfile = controlfile.joinpath("control.in")
        assert controlfile.exists(), "The path {} does not exist.".format(
            str(controlfile)
        )
        assert (
            str(controlfile.parts[-1]) == "control.in"
        ), "File is not named control.in ."
        self.controlfile = controlfile
        self.read_control()

    def __repr__(self):
        return "{}(controlfile={})".format(
            self.__class__.__name__, repr(self.controlfile)
        )

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(FHIAimsControlReader, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(FHIAimsControlReader, self).__delitem__(key)
        del self.__dict__[key]

    def read_control(self):
        p = {
            "xc": None,
            "dispersion_correction": None,
            "relativistic": "atomic_zora scalar",
            "include_spin_orbit": False,
            "k_grid": None,
            "spin": "none",
            "default_initial_moment": None,
            "fixed_spin_moment": None,
            "tasks": {},
            "band_sections": [],
            "mulliken_band_sections": [],
            "qpe_calc": None,
            "use_dipole_correction": False,
            "compute_dielectric": False,
        }
        tasks = set()
        band_sections = []
        mulliken_band_sections = []
        control = self.controlfile

        with open(control, "r") as file:
            content = [
                line.strip() for line in file.readlines() if not line.startswith("#")
            ]

        for line in content:
            if re.search(r"\bxc\b\s+", line):
                xc = " ".join(line.split()[1:])
                p["xc"] = xc
            if re.search(r"\bspin\b\s+", line):
                spin = line.split()[-1]
                p["spin"] = spin
            if re.search(r"\brelativistic\b\s+", line):
                relativistic = " ".join(line.split()[1:])
                p["relativistic"] = relativistic
            if "include_spin_orbit" in line:
                p["include_spin_orbit"] = True
            if re.search(r"\bk_grid\b\s+", line):
                k_grid = tuple(map(int, line.split()[1:]))
                p["k_grid"] = k_grid
            if "qpe_calc" in line:
                qpe = " ".join(line.split()[1:])
                p["qpe_calc"] = qpe
            # dispersion variants
            if re.search(r"\bmany_body_dispersion\b", line):
                p["dispersion_correction"] = "MBD"
            if re.search(r"\bmany_body_dispersion_nl\b", line):
                p["dispersion_correction"] = "MBD-nl"
            if re.search(r"\bvdw_correction_hirshfeld\b", line):
                p["dispersion_correction"] = "TS"
            # dos variants
            if re.search(r"\boutput\s+dos\b", line):
                tasks.add("total dos")
            if re.search(r"\boutput\s+dos_tetrahedron\b", line):
                tasks.add("total dos tetrahedron")
            if re.search(r"\boutput\s+atom_proj_dos\b", line):
                tasks.add("atom-projected dos")
            if re.search(r"\boutput\s+atom_proj_dos_tetrahedron\b", line):
                tasks.add("atom-projected dos tetrahedron")
            if re.search(r"\boutput\s+species_proj_dos\b", line):
                tasks.add("species-projected dos")
            if re.search(r"\boutput\s+species_proj_dos_tetrahedron\b", line):
                tasks.add("species-projected dos tetrahedron")
            # band structure variants
            if re.search(r"\boutput\s+band\b", line):
                tasks.add("band structure")
                band_sections.append(line)
            if re.search(r"\boutput\s+band_mulliken\b", line):
                tasks.add("mulliken-projected band structure")
                mulliken_band_sections.append(line)
            if "default_initial_moment" in line:
                p["default_initial_moment"] = float(line.strip().split()[-1])
            if "fixed_spin_moment" in line:
                p["fixed_spin_moment"] = float(line.strip().split()[-1])
            if "use_dipole_correction" in line:
                p["use_dipole_correction"] = True
            # charge analysis variants
            if re.search(r"\boutput\s+hirshfeld\b", line):
                tasks.add("hirshfeld charge analysis")
            # TDDFT absorption spectrum
            if "compute_dielectric" in line:
                p["compute_dielectric"] = line.split()[-2:]
                tasks.add("absorption")

        p["tasks"] = tasks
        p["band_sections"] = band_sections
        p["mulliken_band_sections"] = mulliken_band_sections

        for key, item in p.items():
            self[key] = item


class FHIAimsOutputReader(dict):
    """Parses information from output file.

    Args:
        output (pathlib object): Directory of outputfile or outputfile.

    Attributes:
        structure (structure): :class:`~aimstools.structuretools.structure.Structure`.
        is_converged (bool): If calculation finished with 'Have a nice day.'
        control (dict): Dictionary of parameters from control.in.
        aims_version (str): FHI-aims version.
        commit_number (str): Commit number (git tag).
        spin_N (float): Number of electrons with spin up - number of electrons with spin down.
        spin_S (float): Total spin.
        total_energy (float): Total energy uncorrected.
        band_extrema (namedtuple): (vbm_scalar, cbm_scalar, vbm_soc, cbm_soc.
        fermi_level (namedtuple): (scalar, soc, scalar spin up, scalar spin down).
        work_function (namedtuple): (upper_vacuum_level, lower_vacuum_level, upper_work_function, lower_work_function).
        nkpoints (int): Number of k-points.
        nscf_steps (int): Number of SCF steps.
    """

    def __init__(self, output) -> None:

        output = Path(output)
        assert output.exists(), "The path {} does not exist.".format(
            str(output)
        )  # Thanks Aga ;D
        if output.is_file():
            self.outputfile = output
            self.outputdir = output.parent
        elif output.is_dir():
            self.outputdir = output
            self.outputfile = self.__find_outputfile()
        assert self.outputfile != None, "Could not find outputfile!"
        logger.debug("Found outputfile: {}".format(str(self.outputfile)))
        logger.debug("Calculation converged: {}".format(self.is_converged))
        geometry = self.outputdir.joinpath("geometry.in")
        assert geometry.exists(), "File geometry.in not found."
        self.structure = Structure(geometry)

        self.control = FHIAimsControlReader(self.outputdir)
        self.read_outputfile()
        if self.is_converged:
            self.check_consistency()
            self.bandgap = self.get_bandgap()

    def __find_outputfile(self):
        outputdir = self.outputdir
        files = list(outputdir.glob("*.out*"))
        for k in files:
            if str(k.parts[-1]) == "aims.out":
                outputfile = outputdir.joinpath("aims.out")
                break
            else:
                check = os.popen("head {}".format(k)).read()
                if "Invoking FHI-aims ..." in check:
                    outputfile = k
                    break
        else:
            outputfile = None
        return outputfile

    def __repr__(self):
        return "{}(outputfile={}, is_converged={})".format(
            self.__class__.__name__, repr(self.outputfile), self.is_converged
        )

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(FHIAimsOutputReader, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(FHIAimsOutputReader, self).__delitem__(key)
        del self.__dict__[key]

    @property
    def is_converged(self):
        outputfile = self.outputfile
        check = os.popen("tail -n 10 {}".format(outputfile)).read()
        if "Have a nice day." in check:
            return True
        else:
            return False

    def read_outputfile(self):
        outputfile = self.outputfile
        assert outputfile.exists(), "File aims.out not found."

        d = {
            "aims_version": None,
            "commit_number": None,
            "spin_N": 0,
            "spin_S": 0,
            "total_energy": None,
            "electronic_free_energy": None,
            "band_extrema": None,
            "fermi_level": None,
            "work_function": None,
            "nkpoints": None,
            "nscf_steps": None,
            "ntasks": None,
            "total_time": None,
        }

        with open(outputfile, "r") as file:
            lines = file.readlines()

        value = re.compile(r"[-]?(\d+)?\.\d+([E,e][-,+]?\d+)?")
        vbm, cbm, vbm_soc, cbm_soc = None, None, None, None
        scalar_fermi_level = None
        soc_fermi_level = None
        scalar_fermi_level_up, scalar_fermi_level_dn = None, None
        pot_upper, pot_lower, wf_upper, wf_lower = None, None, None, None

        from itertools import cycle

        socread = False
        iterable = cycle(lines)
        for i in range(len(lines)):
            l = next(iterable)
            if "FHI-aims version" in l:
                d["aims_version"] = l.strip().split()[-1]
            if "Commit number" in l:
                d["commit_number"] = l.strip().split()[-1]
            if re.search(r"Using\s+\d+\s+parallel tasks", l):
                d["ntasks"] = int(re.search(r"\d+", l).group())
            if "Number of k-points" in l:
                d["nkpoints"] = int(re.search(r"\d+", l).group())
            if "Number of self-consistency cycles" in l:
                d["nscf_steps"] = int(re.search(r"\d+", l).group())
            if re.match(r"\s+\| Total time\s+:\s+\d+\.\d+\s\w\s+\d+.\d+", l):
                d["total_time"] = float(l.strip().split()[-2])
            if "N = N_up - N_down (sum over all k points):" in l:
                d["spin_N"] = float(re.search(r"\d+\.\d+", l).group())
            if "S (sum over all k points)" in l:
                d["spin_S"] = float(re.search(r"\d+\.\d+", l).group())
            if re.match(r"\s+\|\s+\bTotal energy uncorrected\b\s+:", l):
                d["total_energy"] = float(value.search(l).group())
            if "| Electronic free energy        :" in l:
                d["electronic_free_energy"] = float(value.search(l).group())
            if re.search(r"Highest occupied state \(VBM\)", l) and socread == False:
                vbm = float(value.search(l).group())
            if re.search(r"Lowest unoccupied state \(CBM\)", l) and socread == False:
                cbm = float(value.search(l).group())
            if "Chemical potential (Fermi level)" in l and socread == False:
                scalar_fermi_level = float(value.search(l).group())

            # fixed spin moment:
            if "Chemical potential, spin up:" in l:
                scalar_fermi_level_up = float(value.search(l).group())
            if "Chemical potential, spin dn:" in l:
                scalar_fermi_level_dn = float(value.search(l).group())

            # Reading data specific from the SOC part
            if "STARTING SECOND VARIATIONAL SOC CALCULATION" in l:
                socread = True
            if re.search(r"Highest occupied state \(VBM\)", l) and socread:
                vbm_soc = float(re.search(r"[-]?\d+.\d+", l).group())
            if re.search(r"Lowest unoccupied state \(CBM\)", l) and socread:
                cbm_soc = float(re.search(r"[-]?\d+.\d+", l).group())
            if "Chemical potential (Fermi level)" in l and socread:
                soc_fermi_level = float(value.search(l).group())
            if "Have a nice day." in l:
                socread = False

            # work function stuff
            if re.search(r"Work function \(\"upper\" slab surface\)", l):
                wf_upper = float(value.search(l).group())
            if re.search(r"Work function \(\"lower\" slab surface\)", l):
                wf_lower = float(value.search(l).group())
            if re.search(r"Potential vacuum level, \"upper\" slab surface", l):
                pot_upper = float(value.search(l).group())
            if re.search(r"Potential vacuum level, \"lower\" slab surface", l):
                pot_lower = float(value.search(l).group())

            # VBM and CBM information from bandstructure
            if 'Scalar-relativistic "band gap" of total set of bands:' in l:
                cbm = float(next(iterable).strip().split()[-2])
                vbm = float(next(iterable).strip().split()[-2])
            if 'Spin-orbit-coupled "band gap" of total set of bands:' in l:
                cbm_soc = float(next(iterable).strip().split()[-2])
                vbm_soc = float(next(iterable).strip().split()[-2])

        # VBM, CBM
        be = namedtuple(
            "band_extrema", ["vbm_scalar", "cbm_scalar", "vbm_soc", "cbm_soc"]
        )
        d["band_extrema"] = be(vbm, cbm, vbm_soc, cbm_soc)

        # Fermi level
        fl = namedtuple("fermi_level", ["scalar", "soc", "scalar_up", "scalar_dn"])
        d["fermi_level"] = fl(
            scalar_fermi_level,
            soc_fermi_level,
            scalar_fermi_level_up,
            scalar_fermi_level_dn,
        )

        if self.control["use_dipole_correction"]:
            # work function
            wf = namedtuple(
                "work_function",
                [
                    "upper_vacuum_level",
                    "lower_vacuum_level",
                    "upper_work_function",
                    "lower_work_function",
                ],
            )
            d["work_function"] = wf(pot_upper, pot_lower, wf_upper, wf_lower)
        for key, item in d.items():
            self[key] = item

    def check_consistency(self):
        vbm, cbm, vbm_soc, cbm_soc = self.band_extrema
        if self.control["fixed_spin_moment"] == None:
            assert (
                vbm < cbm
            ), "Scalar valence bands are above conduction bands. Either the calculation was inaccurate or the parsing went wrong."
            scalar_gap = cbm - vbm
            if self.control["include_spin_orbit"]:
                assert (
                    vbm_soc < cbm_soc
                ), "SOC valence bands are above conduction bands. Either the calculation was inaccurate or the parsing went wrong."
                soc_gap = cbm_soc - vbm_soc
                if scalar_gap > 0.1 and soc_gap < 0.1:
                    logger.warning(
                        "System is semiconducting without SOC and becomes metallic with SOC. This is probably due to bad occupations."
                    )
                    logger.warning(
                        "You should check your numerical settings (sc_accuracy_rho, k_grid...)."
                    )

    def get_bandgap(self):
        b = namedtuple("band_gap", ["scalar", "soc"])
        band_extrema = self.band_extrema
        soc_gap = None
        if self.control["include_spin_orbit"]:
            soc_gap = abs(band_extrema.cbm_soc - band_extrema.vbm_soc)
        scalar_gap = abs(band_extrema.vbm_scalar - band_extrema.cbm_scalar)
        return b(scalar_gap, soc_gap)
