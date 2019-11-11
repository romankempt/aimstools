from ase import Atoms
import ase.io
from ase.calculators.aims import Aims
import argparse
from pathlib import Path as Path


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("geometry", help="Path to geometry", type=str)

    # Optional arguments
    parser.add_argument(
        "-cluster", help="t3000 or taurus (default: t3000)", type=str, default="t3000"
    )
    parser.add_argument(
        "-m", "--memory", help="memory in GB (default: 63)", type=str, default="63"
    )
    parser.add_argument(
        "-n", "--nodes", help="number of nodes (default: 4)", type=int, default=4
    )
    parser.add_argument(
        "-ppn", help="processors per node (default: 20)", type=int, default=20
    )
    parser.add_argument(
        "-wt",
        "--walltime",
        help="walltime in hours (default: 72)",
        type=int,
        default=72,
    )
    parser.add_argument(
        "-xc",
        help="exchange correlation functional (default pbe)",
        type=str,
        default="pbe",
    )
    parser.add_argument(
        "-spin", help="spin keyword (default none)", type=str, default="none"
    )
    parser.add_argument("-tier", help="basis set tier (default 2)", type=int, default=2)
    parser.add_argument(
        "-basis", help="basis set type (default tight)", type=str, default="tight"
    )
    parser.add_argument(
        "-k_grid",
        help="k-points per reciprocal lattice direction for x, y, z (default: 6 6 6)",
        nargs="+",
        type=int,
        default=[6, 6, 6],
    )
    parser.add_argument(
        "-SOC",
        help="include spin-orbit coupling (default False)",
        type=str,
        default=False,
    )
    parser.add_argument(
        "-task",
        nargs="+",
        help="""list of task(s) to perform:
        None (default) = single point;
        BS = band structure;
        DOS = (atom-projected) density of states;
        GO = geometry optimisation""",
        type=str,
        default=[],
    )
    parser.add_argument(
        "-pbc",
        help="""Specify 2D periodic boundary conditions if your 
        coordinate file does not support 2D pbc with --pbc 2D.""",
        type=str,
        default="",
    )
    parser.add_argument(
        "-vdw",
        help="""Add vdW correction, e.g., TS for vdw_correction_hirshfeld
    or MBD for many_body_dispersion (default: None).""",
        type=str,
        default=None,
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def setup_calculator(args):
    """ This function sets up the calculator object of the ASE.
    Because certain functions are not implemented yet, it simply
    writes out the input files."""
    atoms = ase.io.read(args.geometry)
    if atoms.pbc[2] == False:
        atoms.cell[2] = [0.0, 0.0, 100]
    calc = Aims(
        xc=args.xc,
        spin=args.spin,
        species_dir=species_dir,
        tier=args.tier,
        k_grid=args.k_grid,
        relativistic=("atomic_zora", "scalar"),
        adjust_scf="once 2",
    )
    atoms.set_calculator(calc)
    calc.prepare_input_files()


def setup_bandpath():
    """ This function sets up the band path according to AFLOW conventions
    in the AIMS-specific format. """
    atoms = ase.io.read(args.geometry)
    if args.pbc == "2D":
        atoms.cell[2] = [0.0, 0.0, 0.0]
    npoints = 31
    points = atoms.cell.get_bravais_lattice().get_special_points()
    path = [
        char for char in atoms.cell.get_bravais_lattice().special_path.replace(",", "")
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
    return output_bands


def process_control(args, output_bands=None):
    """ This function processes the control.in file to add
    comments that are currently not implemented in the ASE."""
    with open("control.in", "r+") as file:
        control = file.readlines()
    with open("control.in", "w") as file:
        for line in control:
            write = False if line.startswith("#") else True
            if write:
                if "hse06" in line:
                    line = "xc                                 hse06 0.11\n"
                    line += "hse_unit        bohr-1\n"
                    line += "exx_band_structure_version        1\n"
                if "xc" in line:  # corrections to the functional
                    if args.vdw in ["MBD", "TS"]:
                        if args.vdw == "TS":
                            line += "vdw_correction_hirshfeld\n"
                        elif args.vdw == "MBD":
                            line += "many_body_dispersion_rsscs\n"
                elif ("atomic_zora scalar" in line) and (args.SOC == True):
                    line += "include_spin_orbit \n"
                elif "adjust_scf" in line:
                    if "BS" in args.task:
                        line += "### band structure section \n"
                        for band in output_bands:
                            line += band + "\n"
                    if "DOS" in args.task:
                        line += "### DOS section \n"
                        line += (
                            "output atom_proj_dos  -10 0 300 0.05\n"
                        )  # Estart  Eend  n_points broadening
                        line += (
                            "dos_kgrid_factors 4 4 4 \n"
                        )  # auxiliary increased k-grid to make dos smooth# \n
                    if "GO" in args.task:
                        line += "### geometry optimisation section \n"
                        line += "relax_geometry    bfgs    1E-2\n"
                        line += (
                            "relax_unit_cell       fixed_angles\n"
                        )  # none, full, fixed_angles
            file.write(line)


def write_submit_t3000(args):
    """ Writes the .sh file to submit on the t3000 via qsub. """
    name = args.geometry.split(".")[0]
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
                name=name, cpus=args.nodes * args.ppn, **vars(args)
            )
        )


def write_submit_taurus(args):
    """ Writes the .sh file to submit on the taurus via sbatch. """
    name = args.geometry.split(".")[0]
    with open(name + ".sh", "w+") as file:
        file.write(
            """#!/bin/bash
#SBATCH --time={walltime}:00:00   # walltime in h
#SBATCH --nodes={nodes}  # number of nodes
#SBATCH --ntasks={cpus}      # tasks per node
#SBATCH --mem={memory}GB   # memory per node
#SBATCH -J {name}   # job name
#SBATCH --error={name}.err
#SBATCH --output={name}.out

module use /projects/m_chemie/privatemodules/
module add aims/aims_unknown

COMPUTE_DIR=aims_$SLURM_JOB_ID
ws_allocate -F ssd $COMPUTE_DIR 7
export AIMS_SCRDIR=/ssd/ws/$USER-$COMPUTE_DIR

export OMP_NUM_THREADS=1
srun aims.190319.mpi.x > {name}.out
        """.format(
                name=name, cpus=args.nodes * args.ppn, **vars(args)
            )
        )


if __name__ == "__main__":
    import os

    global species_dir
    species_dir = os.getenv("AIMS_SPECIES_DIR")
    args = parseArguments()
    cwd = Path.cwd()
    species_dir = str(cwd.joinpath(Path(species_dir, args.basis)))
    if args.SOC != False:
        args.SOC = True
    atoms = setup_calculator(args)
    if "BS" in args.task:
        output_bands = setup_bandpath()
        process_control(args, output_bands=output_bands)
    else:
        process_control(args)
    if args.cluster == "t3000":
        write_submit_t3000(args)
    elif args.cluster == "taurus":
        write_submit_taurus(args)
