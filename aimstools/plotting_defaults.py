""" Default plotting settings. """

import matplotlib.pyplot as plt
import os
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent


def set_global_plotsettings(target="presentation"):
    from matplotlib import rcParams
    import matplotlib.font_manager

    if target == "presentation":
        CONFIG_PATH = ROOT_DIR.joinpath("presentation.mplstyle")
    elif target == "paper":
        CONFIG_PATH = ROOT_DIR.joinpath("paper.mplstyle")
    plt.style.use(CONFIG_PATH)


set_global_plotsettings(target="presentation")

# color_definitions
mutedblack = "#1a1a1a"
fermi_color = "#D62728"
fermi_alpha = 1
darkgray = "#636363"

