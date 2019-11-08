# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:12:57 2019

@author: roman
"""

import sys, os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from AIMS_tools import bandstructure, dos
from pathlib import Path as Path

# plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.framealpha"] = 0.8
font_name = "Arial"
font_size = 8.5
plt.rcParams.update({"font.sans-serif": font_name, "font.size": font_size})


def combine_bs_dos(BSpath, DOSpath, title, fix_energy_limits=[]):
    fig = plt.figure(constrained_layout=True, figsize=(4, 4))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[3, 1])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])

    plt.sca(ax1)
    ax1 = bandstructure.plot_ZORA_and_SOC(
        BSpath, axes=ax1, fig=fig, fix_energy_limits=fix_energy_limits
    )
    ymin, ymax = ax1.get_ylim()
    plt.sca(ax2)
    DOS = dos.DOS(DOSpath)
    ax2 = DOS.plot_all_atomic_dos(fig=fig, axes=ax2, fix_energy_limits=[ymin, ymax])
    ax2.set_ylabel("")
    ax2.set_yticks([])
    xmin, xmax = ax2.get_xlim()
    ax2.set_xlim(xmin, xmax)

    fig.suptitle(title)
    return fig


if __name__ == "__main__":

    def parseArguments():
        # Create argument parser
        parser = argparse.ArgumentParser()

        # Positional mandatory arguments
        parser.add_argument("BSpath", help="Path to band structure directory", type=str)
        parser.add_argument("DOSpath", help="Path to DOS directory", type=str)
        parser.add_argument("title", help="Title of plot", type=str)
        # Optional arguments
        parser.add_argument(
            "-yl",
            "--y_limits",
            nargs="+",
            help="fixed energy window",
            type=float,
            default=[],
        )

        args = parser.parse_args()
        return args

    args = parseArguments()
    combine_bs_dos(args.BSpath, args.DOSpath, args.title, args.y_limits)
    plt.savefig(args.title + ".png", dpi=300, bbox_inches="tight")

