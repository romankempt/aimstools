from ase import Atoms
import ase.io, ase.cell
from ase.calculators.aims import Aims
import argparse
from pathlib import Path as Path
import glob, sys, os
from AIMS_tools.structuretools import structure
import numpy as np


class prepare:
    """ A base class to initialise and prepare AIMS calculations.
    
    Args:
        geometry (str): Path to geometry file.

    Keyword Arguments:
        xc (str): Exchange-Correlation functional. Defaults to pbe.
        spin (str): None or collinear. Defaults to None.
        basis (str): Basis set quality. Defaults to tight.
        tier (int): Basis set tier. Defaults to 2.
        k_grid (list): List of integers for k-point sampling. Defaults to [6,6,6].
        SOC (bool): Include spin-orbit coupling. Defaults to False.
        pbc (str): "2D" or None. Defaults to None. Enforces 2D boundary conditions.
        vdw (str): TS, MBD or None. Defaults to None. Enables dispersion corrections.
        task (list): List of tasks to perform. Defaults to []. Currently available: BS, DOS, GO.

    Other Parameters:        
        cluster (str): HPC cluster. Defaults to None.
        cost (str): Low, medium or high. Defaults to None. Automatically adjusts nodes and walltimes depending on the cluster.
        memory (int): Memory requirements per node in GB. Defaults to 63.
        nodes (int): Number of nodes. Defaults to 4.
        ppn (int): Processors per node. Defaults to 20.
        walltime (int): Walltime in hours. Defaults to 24.
        
    """

    def __init__(self, geometryfile, *args, **kwargs):
        # Arguments
        self.geometry = geometryfile
        self.cluster = kwargs.get("cluster", None)
        self.cost = kwargs.get("cost", None)
        self.memory = kwargs.get("memory", 63)
        self.nodes = kwargs.get("nodes", 4)
        self.ppn = kwargs.get("ppn", 20)
        self.walltime = kwargs.get("walltime", 24)
        self.xc = kwargs.get("xc", "pbe")
        self.spin = kwargs.get("spin", None)
        self.tier = kwargs.get("tier", 2)
        self.basis = kwargs.get("basis", "tight")
        self.k_grid = kwargs.get("k_grid", [6, 6, 6])
        self.SOC = kwargs.get("SOC", False)
        self.task = kwargs.get("task", [])
        self.pbc = kwargs.get("pbc", None)
        self.vdw = kwargs.get("vdw", None)
        # Initialisation
        self.species_dir = os.getenv("AIMS_SPECIES_DIR")
        cwd = Path.cwd()
        self.species_dir = str(cwd.joinpath(Path(self.species_dir, self.basis)))
        self.structure = structure(geometryfile)
        if self.pbc == "2D":
            self.k_grid = [self.k_grid[0], self.k_grid[1], 1]
            self.structure.enforce_2d()
        if self.SOC != False:
            self.SOC = True

    def setup_calculator(self):
        """ This function sets up the calculator object of the ASE.
        Because certain functions are not implemented yet, it simply
        writes out the input files."""
        calc = Aims(
            xc=self.xc,
            spin=self.spin,
            species_dir=self.species_dir,
            tier=self.tier,
            k_grid=self.k_grid,
            relativistic=("atomic_zora", "scalar"),
            adjust_scf="once 2",
        )
        self.structure.atoms.set_calculator(calc)
        calc.prepare_input_files()

    def setup_bandpath(self):
        """ This function sets up the band path according to AFLOW conventions
        in the AIMS-specific format and stores it in self.output_bands.
        
        Note:
            The ase.cell.lattice.get_bravais_lattice() method apparently does not like Gamma-angles
            that are off 90 or 120 degrees in case of 2D systems. 
            It then aborts with the error message "This transformation
            changes the cell volume."
        """
        atoms = self.structure.atoms
        lattice = atoms.cell.get_bravais_lattice(pbc=atoms.pbc)
        npoints = 31
        points = lattice.get_special_points()
        path = [char for char in lattice.special_path.replace(",", "")]
        AFLOW = []
        for i in range(len(path)):
            if i == 0:
                AFLOW.append(path[i])
            else:
                try:
                    int(path[i])
                except ValueError:
                    AFLOW.append(path[i])
                else:
                    AFLOW[-1] += path[i]
        output_bands = []
        for i in range(len(AFLOW) - 1):
            vec1 = "{:.6f} {:.6f} {:.6f}".format(*points[AFLOW[i]])
            vec2 = "{:.6f} {:.6f} {:.6f}".format(*points[AFLOW[i + 1]])
            output_bands.append(
                "output band {vec1}    {vec2}  {npoints}  {label1} {label2}".format(
                    label1=AFLOW[i],
                    label2=AFLOW[i + 1],
                    npoints=npoints,
                    vec1=vec1,
                    vec2=vec2,
                )
            )
        self.output_bands = output_bands

    def __adjust_xc(self, line):
        if "hse06" in line:
            line = "xc                                 hse06 0.11\n"
            line += "hse_unit        bohr-1\n"
            line += "exx_band_structure_version        1\n"
        if self.vdw in ["MBD", "TS"]:
            if self.vdw == "TS":
                line += "vdw_correction_hirshfeld\n"
            elif (self.vdw == "MBD") and (self.pbc != "2D"):
                line += "many_body_dispersion\n"
            elif (self.vdw == "MBD") and (self.pbc == "2D"):
                line += "many_body dispersion           vacuum=False:False:True\n"
        return line

    def __adjust_scf(self, line):
        line = "### SCF settings \n"
        if "GO" not in self.task:
            line += "adjust_scf     once        2\n"
        else:
            line += "adjust_scf     always      2\n"
        line += "# charge_mix_param  0.05\n"
        line += "# sc_accuracy_eev   1E-3                       # sum of eigenvalues convergence\n"
        line += "# sc_accuracy_etot  1E-6                       # total energy convergence\n"
        line += "# sc_accuracy_rho   1E-3                       # electron density convergence\n"
        return line

    def __adjust_task(self, line):
        if "BS" in self.task:
            line += "### band structure section \n"
            for band in self.output_bands:
                line += band + "\n"
        if "DOS" in self.task:
            line += "### DOS section \n"
            line += "output atom_proj_dos  -10 0 300 0.05       # Estart Eend n_points broadening\n"
            line += "dos_kgrid_factors 4 4 4                    # auxiliary k-grid\n"
        if "GO" in self.task:
            line += "### geometry optimisation section\n"
            line += "relax_geometry    bfgs    1E-2\n"
            line += "relax_unit_cell   fixed_angles             # none, full, fixed_angles \n"
        return line

    def adjust_cost(self):
        """ This function adjusts the cluster-specific cost requirements to some defaults."""
        if self.cluster == "taurus":
            if self.cost == "low":
                self.memory = 60
                self.ppn = 24
                self.nodes = 1
                self.walltime = 24
            if self.cost == "medium":
                self.memory = 126
                self.ppn = 24
                self.nodes = 4
                self.walltime = 72
            if self.cost == "high":
                self.memory = 254
                self.ppn = 24
                self.nodes = 2
                self.walltime = 72
        if self.cluster == "t3000":
            if self.cost == "low":
                self.memory = 60
                self.ppn = 20
                self.nodes = 4
                self.walltime = 150
            if self.cost == "medium":
                self.memory = 128
                self.ppn = 20
                self.nodes = 4
                self.walltime = 150
            if self.cost == "high":
                self.memory = 256
                self.ppn = 20
                self.nodes = 2
                self.walltime = 150

    def adjust_control(self):
        """ This function processes the control.in file to add
        comments that are currently not implemented in the ASE."""
        with open("control.in", "r+") as file:
            control = file.readlines()
        with open("control.in", "w") as file:
            for line in control:
                write = False if line.startswith("#") else True
                if write:
                    if "xc" in line:  # corrections to the functional
                        line = self.__adjust_xc(line)
                    elif ("spin" in line) and ("collinear" in line):
                        line += "#default_initial_moment   0      # only necessary if not specified in geometry.in\n"
                    elif ("atomic_zora scalar" in line) and (self.SOC == True):
                        line += "include_spin_orbit \n"
                    elif "adjust_scf" in line:
                        line = self.__adjust_scf(line)
                        line = self.__adjust_task(line)
                file.write(line)

    def write_submit_t3000(self):
        """ Writes the .sh file to submit on the t3000 via qsub. """
        name = self.geometry.split(".")[0]
        with open(name + ".sh", "w+") as file:
            file.write(
                """#! /bin/sh
#PBS -N {name}
#PBS -l walltime={walltime}:00:00,mem={memory}GB,nodes={nodes}:ppn={ppn}
module ()
    eval `/usr/bin/modulecmd bash $*`
    module load aims
export OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR

mpirun -np {cpus} bash -c "ulimit -s unlimited && aims.171221_1.scalapack.mpi.x" < /dev/null > {name}.out
                """.format(
                    name=name,
                    cpus=self.nodes * self.ppn,
                    walltime=self.walltime,
                    memory=self.memory,
                    nodes=self.nodes,
                    ppn=self.ppn,
                )
            )

    def write_submit_taurus(self):
        """ Writes the .sh file to submit on the taurus via sbatch. """
        name = self.geometry.split(".")[0]
        with open(name + ".sh", "w+") as file:
            file.write(
                """#!/bin/bash
#SBATCH --time={walltime}:00:00 \t\t# walltime in h
#SBATCH --nodes={nodes} \t\t\t# number of nodes
#SBATCH --ntasks={cpus} \t\t\t# number of cpus
#SBATCH --mem-per-cpu={memory}MB \t\t# memory per node per cpu
#SBATCH -J {name} \t\t\t# job name
#SBATCH --error=slurm.err \t\t# error output
#SBATCH --output=slurm.out \t\t# output

module use /projects/m_chemie/privatemodules/
module add aims/aims_2155

COMPUTE_DIR=aims_$SLURM_JOB_ID
ws_allocate -F ssd $COMPUTE_DIR 7
export AIMS_SCRDIR=/ssd/ws/$USER-$COMPUTE_DIR

export OMP_NUM_THREADS=1
srun aims.191127.scalapack.mpi.x > {name}.out
            """.format(
                    name=name,
                    cpus=self.nodes * self.ppn,
                    memory=int(int(self.memory) * 1000 / (self.ppn)),
                    walltime=self.walltime,
                    nodes=self.nodes,
                )
            )
