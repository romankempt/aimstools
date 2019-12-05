from ase import Atoms
import ase.io
from ase.calculators.aims import Aims
import argparse
from pathlib import Path as Path
import glob, sys, os


class prepare:
    """ A base class to initialise and prepare AIMS calculations."""

    def __init__(self, args):
        self.args = args
        self.species_dir = os.getenv("AIMS_SPECIES_DIR")
        cwd = Path.cwd()
        self.species_dir = str(cwd.joinpath(Path(self.species_dir, self.args.basis)))
        if self.args.pbc == "2D":
            self.args.k_grid = [self.args.k_grid[0], self.args.k_grid[1], 1]
        if self.args.SOC != False:
            self.args.SOC = True
        self.setup_calculator()
        if "BS" in self.args.task:
            self.setup_bandpath()
        self.adjust_control()
        self.adjust_cost()
        if self.args.cluster == "t3000":
            self.write_submit_t3000()
        elif self.args.cluster == "taurus":
            self.write_submit_taurus()

    def setup_calculator(self):
        """ This function sets up the calculator object of the ASE.
        Because certain functions are not implemented yet, it simply
        writes out the input files."""
        atoms = ase.io.read(self.args.geometry)
        if atoms.pbc[2] == False:
            atoms.cell[2] = [0.0, 0.0, 100]
        calc = Aims(
            xc=self.args.xc,
            spin=self.args.spin,
            species_dir=self.species_dir,
            tier=self.args.tier,
            k_grid=self.args.k_grid,
            relativistic=("atomic_zora", "scalar"),
            adjust_scf="once 2",
        )
        atoms.set_calculator(calc)
        calc.prepare_input_files()

    def setup_bandpath(self):
        """ This function sets up the band path according to AFLOW conventions
        in the AIMS-specific format. """
        atoms = ase.io.read(self.args.geometry)
        if self.args.pbc == "2D":
            atoms.cell[2] = [0.0, 0.0, 0.0]
        npoints = 31
        points = atoms.cell.get_bravais_lattice().get_special_points()
        path = [
            char
            for char in atoms.cell.get_bravais_lattice().special_path.replace(",", "")
        ]
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

    def adjust_xc(self, line):
        if "hse06" in line:
            line = "xc                                 hse06 0.11\n"
            line += "hse_unit        bohr-1\n"
            line += "exx_band_structure_version        1\n"
        if self.args.vdw in ["MBD", "TS"]:
            if self.args.vdw == "TS":
                line += "vdw_correction_hirshfeld\n"
            elif (self.args.vdw == "MBD") and (self.args.pbc != "2D"):
                line += "many_body_dispersion\n"
            elif (self.args.vdw == "MBD") and (self.args.pbc == "2D"):
                line += "many_body dispersion           vacuum=False:False:True\n"
        return line

    def adjust_scf(self, line):
        line = "### SCF settings \n"
        if "GO" not in self.args.task:
            line += "adjust_scf     once        2\n"
        else:
            line += "adjust_scf     always      2\n"
        line += "# charge_mix_param  0.05\n"
        line += "# sc_accuracy_eev   1E-3                       # sum of eigenvalues convergence\n"
        line += "# sc_accuracy_etot  1E-6                       # total energy convergence\n"
        line += "# sc_accuracy_rho   1E-3                       # electron density convergence\n"
        return line

    def adjust_task(self, line):
        if "BS" in self.args.task:
            line += "### band structure section \n"
            for band in self.output_bands:
                line += band + "\n"
        if "DOS" in self.args.task:
            line += "### DOS section \n"
            line += "output atom_proj_dos  -10 0 300 0.05       # Estart Eend n_points broadening\n"
            line += "dos_kgrid_factors 4 4 4                    # auxiliary k-grid\n"
        if "GO" in self.args.task:
            line += "### geometry optimisation section\n"
            line += "relax_geometry    bfgs    1E-2\n"
            line += "relax_unit_cell   fixed_angles             # none, full, fixed_angles \n"
        return line

    def adjust_cost(self):
        if self.args.cluster == "taurus":
            if self.args.cost == "low":
                self.args.memory = 60
                self.args.ppn = 24
                self.args.nodes = 1
                self.args.walltime = 24
            if self.args.cost == "medium":
                self.args.memory = 126
                self.args.ppn = 24
                self.args.nodes = 4
                self.args.walltime = 72
            if self.args.cost == "high":
                self.args.memory = 254
                self.args.ppn = 24
                self.args.nodes = 2
                self.args.walltime = 72
        if self.args.cluster == "t3000":
            if self.args.cost == "low":
                self.args.memory = 60
                self.args.ppn = 20
                self.args.nodes = 4
                self.args.walltime = 150
            if self.args.cost == "medium":
                self.args.memory = 128
                self.args.ppn = 20
                self.args.nodes = 4
                self.args.walltime = 150
            if self.args.cost == "high":
                self.args.memory = 256
                self.args.ppn = 20
                self.args.nodes = 2
                self.args.walltime = 150

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
                        line = self.adjust_xc(line)
                    elif ("spin" in line) and ("collinear" in line):
                        line += "#default_initial_moment   0      # only necessary if not specified in geometry.in\n"
                    elif ("atomic_zora scalar" in line) and (self.args.SOC == True):
                        line += "include_spin_orbit \n"
                    elif "adjust_scf" in line:
                        line = self.adjust_scf(line)
                        line = self.adjust_task(line)
                file.write(line)

    def write_submit_t3000(self):
        """ Writes the .sh file to submit on the t3000 via qsub. """
        name = self.args.geometry.split(".")[0]
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
                    name=name, cpus=self.args.nodes * self.args.ppn, **vars(self.args)
                )
            )

    def write_submit_taurus(self):
        """ Writes the .sh file to submit on the taurus via sbatch. """
        name = self.args.geometry.split(".")[0]
        with open(name + ".sh", "w+") as file:
            file.write(
                """#!/bin/bash
#SBATCH --time={walltime}:00:00 \t\t# walltime in h
#SBATCH --nodes={nodes} \t\t\t# number of nodes
#SBATCH --ntasks={cpus} \t\t\t# number of cpus
#SBATCH --mem={mem}MB \t\t\t# memory per node
#SBATCH -J {name} \t\t\t# job name
#SBATCH --error={name}.err \t\t# error output
#SBATCH --output={name}.out \t\t# output

module use /projects/m_chemie/privatemodules/
module add aims/aims_2155

COMPUTE_DIR=aims_$SLURM_JOB_ID
ws_allocate -F ssd $COMPUTE_DIR 7
export AIMS_SCRDIR=/ssd/ws/$USER-$COMPUTE_DIR

export OMP_NUM_THREADS=1
srun aims.191127.scalapack.mpi.x > {name}.out
            """.format(
                    name=name,
                    cpus=self.args.nodes * self.args.ppn,
                    mem=int(int(self.args.memory) * 1000 / (self.args.ppn)),
                    **vars(self.args),
                )
            )
