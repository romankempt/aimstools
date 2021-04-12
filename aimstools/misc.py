import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pretty_errors
from rich.logging import RichHandler

import contextvars

from aimstools.plotting_defaults import *

# declaring the variable
axes_order = contextvars.ContextVar("axes_order", default=0)


pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=2,
    lines_after=1,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)
pretty_errors.blacklist("c:/python")


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt="{message:s}", style="{")
    handler = RichHandler(
        show_time=False, markup=True, rich_tracebacks=True, show_path=False
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = setup_custom_logger("root")


def set_verbosity_level(verbosity):
    logger = logging.getLogger("root")
    if verbosity == 0:
        level = "WARNING"
    elif verbosity == 1:
        level = "INFO"
    else:
        level = "DEBUG"
        formatter = logging.Formatter(
            fmt="{levelname:8s} {module:20s} {funcName:20s} |\n {message:s}", style="{"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


class AxesContext:
    """Base axes context.

    Args:
        main (bool): Helper value to identify execution orders in nested contexts to save/show the figure only in the main context.
    """

    def __init__(
        self,
        ax: "matplotlib.axes.Axes" = None,
        filename: str = None,
        main: bool = True,
        nrows: int = 1,
        ncols: int = 1,
        width_ratios: list = None,
        height_ratios: list = None,
        projections: list = None,
        hspace: float = 0.05,
        wspace: float = 0.05,
        **kwargs
    ) -> None:
        self.ax = ax
        self.filename = filename
        self.figsize = kwargs.get("figsize", plt.rcParams["figure.figsize"])
        self.dpi = kwargs.get("dpi", plt.rcParams["figure.dpi"])
        self.nrows = nrows
        self.ncols = ncols
        self.main = main
        self.width_ratios = width_ratios or [1] * ncols
        self.height_ratios = height_ratios or [1] * nrows
        self.hspace = hspace
        self.wspace = wspace
        self.projections = projections or [
            ["rectilinear" for i in range(ncols)] for j in range(nrows)
        ]
        self.show = kwargs.get("show", True)

    def __enter__(self) -> "matplotlib.axes.Axes":
        if self.ax is None:
            self.figure = plt.figure(constrained_layout=True, figsize=self.figsize)
            self.spec = gridspec.GridSpec(
                ncols=self.ncols,
                nrows=self.nrows,
                figure=self.figure,
                width_ratios=self.width_ratios,
                height_ratios=self.height_ratios,
                hspace=self.hspace,
                wspace=self.wspace,
            )
            for i in range(self.nrows):
                for j in range(self.ncols):
                    self.figure.add_subplot(
                        self.spec[i, j], projection=self.projections[i][j]
                    )
            self.ax = self.figure.axes
            if self.ncols == self.nrows and self.ncols == 1:
                self.ax = self.ax[0]
            self.show = self.show if self.filename is None else False
        else:
            self.figure = plt.gcf()
            plt.sca(self.ax)
            self.show = False

        logger.debug("Is main context: {}".format(self.main))
        return self.ax

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if (exc_type is None) and (self.main):
            # If there was no exception, display/write the plot as appropriate
            if self.figure is None:
                raise Exception("Something went wrong initializing matplotlib figure.")
            if self.show:
                plt.show()
            if self.filename is not None:
                self.figure.savefig(
                    self.filename,
                    dpi=self.dpi,
                    facecolor="white",
                    transparent=False,
                    bbox_inches="tight",
                )
        return None
