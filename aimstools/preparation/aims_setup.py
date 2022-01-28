import ase.io, ase.cell
from ase.calculators.aims import Aims

from aimstools.misc import *
from aimstools.structuretools import Structure
from aimstools.preparation.templates import aims_slurm_template

from ase.calculators.calculator import kptdensity2monkhorstpack

import os
import numpy as np
from pathlib import Path


class FHIAimsSetup:
    """A base class to initialise and prepare AIMS calculations.

    Args:
        geometry (str): Path to geometry file.
    """

    def __init__(self, geometryfile, **kwargs):
        self.geometryfile = Path(geometryfile)
        assert (
            self.geometryfile.exists() and self.geometryfile.is_file()
        ), "No structure input found."
        self.dirpath = Path(self.geometryfile).parent
        self.structure = Structure(self.geometryfile)

        xc = kwargs.get("xc", "pbe")
        spin = kwargs.get("spin", "none")
        tier = kwargs.get("tier", 1)
        basis = kwargs.get("basis", "tight")
        k_grid = kwargs.get("k_grid", [])
        k_density = kwargs.get("k_density", 5)
        if k_grid == []:
            k_grid = kptdensity2monkhorstpack(
                self.structure, kptdensity=k_density, even=False
            )

        species_dir = Path(os.getenv("AIMS_SPECIES_DIR"))
        species_dir = species_dir.joinpath(basis)
        self.aseargs = {
            "xc": xc,
            "relativistic": ("atomic_zora", "scalar"),
            "spin": spin,
            "k_grid": list(k_grid),
            "species_dir": species_dir,
            "tier": tier,
        }
        self.tasks = set()
        tasks = kwargs.get("tasks", set())
        self.set_tasks(tasks)
        if self.structure.is_2d():
            logger.info("Structure is recognized as two-dimensional.")

    def setup_geometry(self, overwrite=False):
        geometry = self.dirpath.joinpath("geometry.in")
        if geometry.exists():
            s1 = self.structure.copy()
            s2 = ase.io.read(self.geometryfile)
            if not (s1 == s2) and (overwrite == False):
                logger.warning(
                    "Input structure from {} and existing geometry.in differ. Set overwrite=True to force overwrite.".format(
                        self.geometryfile
                    )
                )
        if not geometry.exists() or overwrite:
            logger.info("Writing {} ...".format(geometry))
            self.structure.write(geometry)

    def setup_control(self, overwrite=False):
        control = self.dirpath.joinpath("control.in")
        if control.exists() and (overwrite == False):
            logger.warning(
                "File control.in already exists. Set overwrite=True to force overwrite."
            )
        elif not control.exists() or overwrite:
            logger.info("Writing {} ...".format(control))
            calc = Aims(**self.aseargs)
            calc.write_control(self.structure, str(control), debug=False)
            calc.write_species(self.structure, filename=control)
            self.__adjust_control(control)

    def get_bandpath_as_aims_strings(self, pbc=[True, True, True]):
        """This function sets up the band path according to Setyawan-Curtarolo conventions.

        Returns:
            list: List of strings containing the k-path sections.
        """
        from ase.dft.kpoints import parse_path_string, kpoint_convert

        atoms = self.structure.atoms
        atoms.pbc = pbc
        path = parse_path_string(
            atoms.cell.get_bravais_lattice(pbc=atoms.pbc).bandpath().path
        )
        # list Of lists of path segments
        points = atoms.cell.get_bravais_lattice(pbc=atoms.pbc).bandpath().special_points
        segments = []
        for seg in path:
            section = [(i, j) for i, j in zip(seg[:-1], seg[1:])]
            segments.append(section)
        output_bands = []
        index = 1
        for seg in segments:
            output_bands.append("## Brillouin Zone section Nr. {:d}\n".format(index))
            for sec in seg:
                dist = np.array(points[sec[1]]) - np.array(points[sec[0]])
                length = np.linalg.norm(kpoint_convert(atoms.cell, skpts_kc=dist))
                npoints = 21
                vec1 = "{:.6f} {:.6f} {:.6f}".format(*points[sec[0]])
                vec2 = "{:.6f} {:.6f} {:.6f}".format(*points[sec[1]])
                output_bands.append(
                    "{vec1} \t {vec2} \t {npoints} \t {label1} {label2}".format(
                        label1=sec[0],
                        label2=sec[1],
                        npoints=npoints,
                        vec1=vec1,
                        vec2=vec2,
                    )
                )
            index += 1
        return output_bands

    def write_symmetry_block(self):
        """This function sets up parametric symmetry constraints for the FHI-aims lattice relaxation.

        Note:
            This function is deprecated. Use vibes relaxation or the ASE constraint instead.

        Returns:
            str : Symmetry block to be added in geometry.in.
        """
        logger.warning(
            "This function is deprecated. Use vibes relaxation or the ASE constraint instead."
        )
        struc = self.structure.copy()
        try:
            struc.standardize()
            assert len(struc) == len(
                self.structure
            ), "Number of atoms changed due to standardization."
        except:
            logger.warning("Cell could not be standardized.")

        if struc.lattice == "triclinic":
            nlat = 9
            sym_params = "symmetry_params a1 a2 a3 b1 b2 b3 c1 c2 c3"
            latstring = "symmetry_lv a1 , a2 , a3 \nsymmetry_lv b1 , b2 , b3 \nsymmetry_lv c1 , c2 , c3\n"
        elif struc.lattice == "monoclinic":
            # a != b != c, alpha = gamma != beta
            nlat = 4
            sym_params = "symmetry_params a1 b2 c2 c3"
            latstring = "symmetry_lv a1 , 0 , 0 \nsymmetry_lv 0 , b2 , 0 \nsymmetry_lv 0 , c2 , c3\n"
        elif struc.lattice == "orthorhombic":
            # a != b != c, alpha = beta = gamma = 90
            nlat = 3
            sym_params = "symmetry_params a1 b2 c3"
            latstring = "symmetry_lv a1 , 0 , 0 \nsymmetry_lv 0 , b2 , 0 \nsymmetry_lv 0 , 0 , c3\n"
        elif struc.lattice == "tetragonal":
            # a = b != c, alpha = beta = gamma = 90
            nlat = 2
            sym_params = "symmetry_params a1 c3"
            latstring = "symmetry_lv a1 , 0 , 0 \nsymmetry_lv 0 , a1 , 0 \nsymmetry_lv 0 , 0 , c3\n"
        elif struc.lattice in ["trigonal", "hexagonal"]:
            # a = b != c, alpha = beta = 90, gamma = 120
            nlat = 2
            sym_params = "symmetry_params a1 c3"
            latstring = "symmetry_lv a1 , 0 , 0 \nsymmetry_lv -a1/2 , sqrt(3.0)*a1/2 , 0 \nsymmetry_lv 0 , 0 , c3\n"
        elif struc.lattice == "cubic":
            # a = b = c, alpha = beta = gamma = 90
            nlat = 1
            sym_params = "symmetry_params a1"
            latstring = "symmetry_lv a1 , 0 , 0 \nsymmetry_lv 0 , a1 , 0 \nsymmetry_lv 0 , 0 , a1\n"
        else:
            logger.error("Lattice not recognized.")
            return None
        nparams = "symmetry_n_params {} {} {}\n".format(
            nlat + len(struc) * 3, nlat, len(struc) * 3
        )
        sym_frac = ""
        for i in range(len(struc) * 3):
            sym_params += " x{}".format(i)
            if (i % 3) == 0:
                sym_frac += "symmetry_frac x{} , x{} , x{}\n".format(i, i + 1, i + 2)
        return nparams + sym_params + "\n" + latstring + sym_frac

    def __adjust_xc(self, line):
        if "hse06" in line:
            line = "xc \t hse06 0.11\n"
            line += "hse_unit \t bohr-1\n"
            line += "exx_band_structure_version \t 1\n"

        line += "# include_spin_orbit\n"
        line += "# use_dipole_correction\n"
        line += "## Common choices of dispersion methods:\n"
        line += "# \t vdw_correction_hirshfeld\n"
        line += "# \t many_body_dispersion\n"
        line += "# \t many_body_dispersion_nl\n"
        return line

    def __adjust_scf(self, line):
        line += "\n### SCF settings \n"
        line += "adjust_scf \t always \t 3 \n"
        line += "sc_iter_limit \t 100\n"
        line += "# frozen_core_scf \t .true. \n"
        line += "# charge_mix_param \t 0.02\n"
        line += "# occupation_type \t gaussian \t 0.1 \n"
        line += "# sc_accuracy_rho \t 1E-6 \t \t # electron density convergence\n"
        line += "# elsi_restart \t read_and_write \t 1000\n"
        return line

    def set_tasks(self, tasks):
        if tasks == set():
            return set()
        tasks = [k.lower() for k in tasks]
        ftasks = set()

        def is_in(small, big):
            return any(x in big for x in small)

        if is_in(["bs", "band structure", "bandstructure"], tasks):
            ftasks.add("band structure")
        if is_in(["dos", "all dos"], tasks):
            ftasks.add("total dos tetrahedron")
            ftasks.add("atom-projected dos tetrahedron")
        if is_in(["old dos", "dos old"], tasks):
            ftasks.add("total dos")
            ftasks.add("atom-projected dos")
        if is_in(
            [
                "fatbs",
                "mbs",
                "fat band structure",
                "fatbandstructure",
                "mulliken-projected band structure",
            ],
            tasks,
        ):
            ftasks.add("mulliken-projected band structure")
        if is_in(["relaxation", "go", "geometry optimization"], tasks):
            ftasks.add("relaxation")
        if is_in(["phonons", "phonon", "phonopy"], tasks):
            ftasks.add("phonons")
        if is_in(["absorption", "UV-Vis"], tasks):
            ftasks.add("absorption")
        self.tasks = ftasks

    def __write_bandstructure_tasks(self, line):
        if self.structure.is_2d():
            pbc = [True, True, False]
        else:
            pbc = [True, True, True]
        if "band structure" in self.tasks:
            line += "\n### Band structure section \n"
            logger.debug("Setting up band structure calculation.")
            output_bands = self.get_bandpath_as_aims_strings(pbc=pbc)
            for band in output_bands:
                if not band.startswith("#"):
                    line += "output band " + band + "\n"
                else:
                    line += band
        if "mulliken-projected band structure" in self.tasks:
            line += "\n### Mulliken-projected band structure section \n"
            logger.debug("Setting up mulliken-projected band structure calculation.")
            output_bands = self.get_bandpath_as_aims_strings(pbc=pbc)
            for band in output_bands:
                if not band.startswith("#"):
                    line += "output band_mulliken " + band + "\n"
                else:
                    line += band
            line += "# band_mulliken_orbit_num 50 \t # number of orbitals to be written out, default is 50 for scalar and 100 for soc"
        return line

    def __write_dos_tasks(self, line):
        if any(["dos" in x for x in self.tasks]):
            line += "\n### DOS section \n"
            logger.debug("Setting up density of states calculation.")
        if "total dos" in self.tasks:
            line += "output dos \t -20 0 300 0.00\n"
            line += "dos_kgrid_factors \t 4 4 4\n"
        if "atom-projected dos" in self.tasks:
            line += "output atom_proj_dos \t -20 0 300 0.00\n"
        if "total dos tetrahedron" in self.tasks:
            line += "output dos_tetrahedron  -20 0 300\n"
        if "atom-projected dos tetrahedron" in self.tasks:
            line += "output atom_proj_dos_tetrahedron -20 0 300\n"
        return line

    def __write_absorption_tasks(self, line):
        if "absorption" in self.tasks:
            line += "\n### TDDFT Absorption spectrum\n"
            logger.debug("Setting up absorption spectrum calculation.")
            line += "#The first value \omega_max in eV specifies the energy of the incoming photon and the upper limit of possible excitations.\n"
            line += "#The second value n_\omega specifies the number of frequency points to propagate through.\n"
            line += "compute_dielectric 15 100\n"
            line += "dielectric_broadening gaussian 0.1\n"
        return line

    def __adjust_control(self, controlfile):
        """ This function adds some useful lines to the control.in """
        with open(controlfile, "r+") as file:
            control = file.readlines()
        with open(controlfile, "w") as file:
            for line in control:
                write = False if line.startswith("#") else True
                if write:
                    if "xc" in line:  # corrections to the functional
                        line = self.__adjust_xc(line)
                        line += "output_level \t \t normal\n"
                    elif ("spin" in line) and ("collinear" in line):
                        line += "#default_initial_moment   0.1      # only necessary if not specified in geometry.in\n"
                    elif "k_grid" in line:
                        line = self.__adjust_scf(line)
                        line = self.__write_bandstructure_tasks(line)
                        line = self.__write_dos_tasks(line)
                        line = self.__write_absorption_tasks(line)
                file.write(line)

    def write_submission_file(self, overwrite=False):
        templatefile = os.getenv("AIMS_SLURM_TEMPLATE")
        task = ""

        def is_in(small, big):
            return any(x in big for x in small)

        if templatefile == None:
            template = aims_slurm_template
        else:
            with open(templatefile, "r") as file:
                template = file.read()
        if "band structure" in self.tasks:
            task += "_bs"
        if "mulliken-projected band structure" in self.tasks:
            task += "_fatbs"
        if is_in(
            [
                "total dos",
                "total dos tetrahedron",
                "atom-projected dos",
                "atom-projected dos tetrahedron",
            ],
            self.tasks,
        ):
            task += "_dos"

        jobname = self.structure.atoms.get_chemical_formula().format("metal") + task
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
