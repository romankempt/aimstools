#!/usr/bin/env python
import argparse
from AIMS_tools.preparation import prepare


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("geometry", help="Path to geometry", type=str)

    # Optional arguments
    parser.add_argument("-cluster", help="t3000 or taurus", type=str, default=None)
    parser.add_argument(
        "-cost",
        help="low, medium or high (automatically adjusts memory and nodes)",
        type=str,
        default=None,
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
        GO = geometry optimisation;
        phonons = Phonon band structure, DOS and free energy""",
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


if __name__ == "__main__":
    args = parseArguments()
    job = prepare(args.geometry, **vars(args))
    job.setup_calculator()
    if ("BS" in args.task) or ("phonons" in args.task):
        job.setup_bandpath()
    job.adjust_control()
    job.adjust_cost()
    if job.cluster == "t3000":
        job.write_submit_t3000()
    elif job.cluster == "taurus":
        job.write_submit_taurus()
