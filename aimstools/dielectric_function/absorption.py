from aimstools.misc import *
from aimstools.postprocessing import FHIAimsOutputReader

import numpy as np

import matplotlib.pyplot as plt

import re

from collections import namedtuple

from ase.units import _c

hplanck = 4.135667516 * 1e-15  # eV * s


class Spectrum:
    """ Container class for the spectrum of :class:`~aimstools.dielectric_function.absorption.AbsorptionSpectrum` ."""

    def __init__(
        self, absorption, direction, broadening_type, broadening_width, energy_unit
    ) -> None:
        self.absorption = absorption.copy()
        self.direction = direction
        self.broadening_type = broadening_type
        self.broadening_width = broadening_width
        self._energy_unit = energy_unit

    def __repr__(self):
        return "{}(absorption={}, direction={}, broadening_type={}, broadening_width={}, energy_unit={})".format(
            self.__class__.__name__,
            self.absorption.shape,
            self.direction,
            self.broadening_type,
            self.broadening_width,
            self.energy_unit,
        )

    def _eV_to_nm(self):
        assert self.energy_unit == "eV", "Energy unit is not eV."
        energies = self.absorption[:, 0].copy()
        wavelengths = (hplanck * _c) * 1e9 / energies
        self._energy_unit = "nm"
        self.absorption[:, 0] = wavelengths

    def _nm_to_eV(self):
        assert self.energy_unit == "nm", "Energy unit is not nm."
        wavelengths = self.absorption[:, 0].copy() / 1e9
        energies = (hplanck * _c) / wavelengths
        self._energy_unit = "eV"
        self.absorption[:, 0] = energies

    def set_energy_unit(self, unit="nm"):
        assert unit in ("eV", "nm"), "Unit not supported. Allowed choices are eV, nm."
        if unit == self._energy_unit:
            return None
        if unit == "eV":
            self._nm_to_eV()
        elif unit == "nm":
            self._eV_to_nm()

    @property
    def energy_unit(self):
        return self._energy_unit

    def __add__(self, other):
        assert (
            self.absorption.shape == other.absorption.shape
        ), "The spectra do not have the same length."
        assert (
            self.broadening_type == other.broadening_type
        ), "The spectra do not have the same broadening type."
        assert (
            self.broadening_width == other.broadening_width
        ), "The spectra do not have the same broadening width."
        assert (
            self.energy_unit == other.energy_unit
        ), "The spectra do not have the same energy unit."
        absorption = self.absorption.copy()
        absorption[:, 1] += other.absorption[:, 1]
        direction = "+".join(set([self.direction, other.direction]))
        return Spectrum(
            absorption,
            direction,
            self.broadening_type,
            self.broadening_width,
            self.energy_unit,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


class AbsorptionSpectrum(FHIAimsOutputReader):
    """Analysis and plotting of absorption spectrum via the linear macroscopic dielectric function."""

    def __init__(self, outputfile):
        super().__init__(outputfile)
        assert self.is_converged, "Calculation did not converge."
        tasks = {x for x in self.control["tasks"] if "absorption" in x}
        accepted_tasks = set(["absorption"])
        assert any(x in accepted_tasks for x in tasks), "Absorption task not accepted."
        self.tasks = tasks
        self.task = None
        self.omega_max, self.n_omega = map(float, self.control.compute_dielectric)
        self.absorption_files = self.get_absorption_files()
        self.spectrum = self.read_absorption_files()

    def get_absorption_files(self):
        # spin = "" if spin == "none" else "spin_" + spin
        if self.control["include_spin_orbit"]:
            logger.info("Absorption spectrum was calculated with spin-orbit coupling.")
            regex = re.compile(r"absorption_soc_[A-Z][a-z]+_\d\.\d{4}_\w_\w\.out")
        else:
            logger.info(
                "Absorption spectrum was calculated without spin-orbit coupling."
            )
            regex = re.compile(r"absorption_[A-Z][a-z]+_\d\.\d{4}_\w_\w\.out")
        absfiles = list(self.outputdir.glob("*.out*"))
        absfiles = [j for j in absfiles if bool(regex.match(str(j.parts[-1])))]
        return absfiles

    def read_absorption_files(self):
        spectrum = {}
        for f in self.absorption_files:
            name = str(f.parts[-1])
            direction = re.search(r"_\w_\w", name).group().replace("_", "")
            width = float(re.search(r"\d\.\d{4}", name).group())
            if "Gaussian" in name:
                type = "Gaussian"
            elif "Lorentzian" in name:
                type = "Lorentzian"
            array = np.loadtxt(f)  # omega in eV, alpha as absorption
            sp = Spectrum(
                array,
                direction,
                type,
                width,
                "eV",
            )
            spectrum[direction] = sp
        spectrum["total"] = sum(spectrum.values())
        return spectrum

    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()

        axargs = {}
        axargs["figsize"] = kwargs.pop("figsize", (3, 6))
        axargs["filename"] = kwargs.pop("filename", None)
        axargs["title"] = kwargs.pop("title", None)

        return axargs, kwargs

    def _set_window(self, window, energy_unit):
        if window == "visible":
            if energy_unit == "eV":
                return (1.51, 3.98)
            elif energy_unit == "nm":
                return (820, 315)
            else:
                logger.error("Unit not recognized.")
        else:
            logger.error("Window not recognized.")

    def _check_components(self, component, components):
        allowed = ["xx", "yy", "zz", "total"]
        if type(components) != list:
            try:
                components = [components]
                assert len(components) <= 4, "Too many components?"
            except:
                raise Exception("Could not convert components to list.")
        if component != None:
            components = []
            assert type(component) == str, "Component must be string."
            assert component in allowed, "Component {} is not allowed.".format(
                component
            )
            return [component]

        assert any(
            item in allowed for item in components
        ), "Some of the components are not allowed."
        return components

    def plot(
        self,
        axes=None,
        components=["xx", "yy", "zz", "total"],
        colors=[],
        labels=[],
        energy_unit="nm",
        window="visible",
        **kwargs
    ):
        """Plots absorption spectrum.

        Arguments:
            components (list): Directions of absorption coefficient.
            colors (list): List of colors per component.
            labels (list): List of labels per component.
            energy_unit (str): Currently supported are 'nm', 'eV'. Default is 'nm'.
            window (str): Currently supported is 'visible'. Sets energy limits according to visible range.
            **kwargs (dict): Passed to matplotlib line plot function.

        Returns:
            axes: matplotlib axes object.

        """
        component = kwargs.pop("component", None)
        label = kwargs.pop("label", None)
        color = kwargs.pop("color", None)
        if label != None:
            assert type(label) == str, "Label must be string."
            labels = [label]
        if color != None:
            assert (
                type(color) == str or type(color) == tuple
            ), "Color must be string or tuple."
            colors = [color]
        axargs, kwargs = self._process_kwargs(**kwargs)
        components = self._check_components(component, components)
        if colors == []:
            cmap = plt.cm.get_cmap("tab10")
            colors = dict(
                zip(
                    ["xx", "yy", "zz", "total"], [cmap(c) for c in np.linspace(0, 1, 4)]
                )
            )
            colors = [v for k, v in colors.items() if k in components]
        if labels == []:
            labels = components

        assert len(colors) == len(
            components
        ), "Number of colors and components does not match."
        assert len(labels) == len(
            components
        ), "Number of labels and components does not match."

        with AxesContext(ax=axes, **axargs) as axes:
            for i, l, c in zip(components, labels, colors):
                p = self.spectrum[i]
                p.set_energy_unit(energy_unit)
                axes.plot(
                    p.absorption[:, 0], p.absorption[:, 1], label=l, color=c, **kwargs
                )

            window = self._set_window(window, energy_unit)
            axes.set_xlim(window)
            axes.legend()
            if energy_unit == "nm":
                axes.set_xlabel(r"$\lambda$ [nm]")
            elif energy_unit == "eV":
                axes.set_xlabel(r"$\omega$ [eV]")
            axes.set_ylabel(r"Absorption $\alpha$ [arb. units]")
            axes.set_yticks([])

        return axes
