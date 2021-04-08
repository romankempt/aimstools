from aimstools.misc import *
from aimstools.postprocessing.vibes_parser import FHIVibesParser

import yaml
from pathlib import Path
import numpy as np
import re

from ase.dft.kpoints import parse_path_string, BandPath
from ase.formula import Formula

from collections import namedtuple

from aimstools.phonons.utilities import (
    PhononSpectrum,
    PhononDOS,
    PhononPlot,
    PhononDOSPlot,
)


class FHIVibesPhonons(FHIVibesParser):
    """Handles phonopy output from FHI-vibes.

    Todo:
        Add DOS.

    """

    def __init__(self, outputdir) -> None:
        super().__init__(outputdir)
        self._get_phonopyfiles()
        self._bandpath = None
        self.bands = self.read_bands()
        self._spectrum = self.set_spectrum(bandpath=None, unit=r"cm$^{-1}$")
        self._dos = None

    def __repr__(self):
        return "{}(output directory={})".format(
            self.__class__.__name__, repr(self.outpudir)
        )

    def read_bands(self):
        outputyaml = self.outputdir.joinpath("band.yaml")
        assert outputyaml.exists(), "File band.yaml not found."
        with open(outputyaml, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        labels = data["labels"]
        bpstring = []
        for i, pair in enumerate(labels):
            if i == 0:
                bpstring.append(pair[0])
            elif i == len(labels) - 1:
                bpstring.append(pair[0])
                bpstring.append(pair[1])
            else:
                bpstring.append(pair[0])
        bpstring = "".join(bpstring)
        bpstring = bpstring.replace("|", ",")
        bplist = parse_path_string(bpstring)

        special_points = (
            self.structure.cell.get_bravais_lattice().bandpath().special_points
        )
        self._bandpath = BandPath(
            path=bpstring, cell=self.structure.cell, special_points=special_points
        )

        phon = data["phonon"]
        nqpoints = data["segment_nqpoint"]

        qpoints = np.array([k["q-position"] for k in phon], dtype=np.float32)
        pbands = np.array(
            [[l["frequency"] for l in k["band"]] for k in phon], dtype=np.float32
        )

        b = namedtuple("band", ["qpoints", "frequencies"])
        count = 0
        bands = {}

        i = 0
        for segment in bplist:
            for pair in zip(segment[:-1], segment[1:]):
                j1, j2 = count, nqpoints[i] + count
                count = j2
                qp = qpoints[j1:j2, :]
                pb = pbands[j1:j2, :]
                bands[pair] = b(qp, pb)
                rev = (pair[1], pair[0])
                bands[rev] = b(qp[::-1], pb[::-1])
                i += 1
        return bands

    def set_bandpath(self, bandpathstring):
        new_bandpath = parse_path_string(bandpathstring)
        old_path = self.bandpath
        special_points = old_path.special_points
        pairs = [(s[0], s[1]) for s in self.bands.keys()]
        for segment in new_bandpath:
            assert len(segment) > 1, "A vertex needs at least two points."
            p = zip(segment[:-1], segment[1:])
            for s1, s2 in p:
                assert any(
                    x in pairs for x in ((s1, s2), (s2, s1))
                ), "The k-path {}-{} has not been calculated.".format(s1, s2)
        else:
            new_path = BandPath(
                path=bandpathstring,
                cell=self.structure.cell,
                special_points=special_points,
            )
        self._bandpath = new_path

    @property
    def bandpath(self):
        return self._bandpath

    def set_spectrum(self, bandpath=None, unit=r"cm$^{-1}$"):
        assert unit in [r"cm$^{-1}$", "Thz"], "Unit not recognized."
        bands = self.bands
        if bandpath != None:
            self.set_bandpath(bandpath)
        bp = parse_path_string(self.bandpath.path)
        jumps = []
        qps = []
        qpoint_axis = []
        qpoint_labels = []
        label_coords = []
        spectrum = []
        icell_cv = 2 * np.pi * np.linalg.pinv(self.structure.cell).T
        for segment in bp:
            qpoint_labels.append(segment[0])
            label_coords.append(label_coords[-1] if len(label_coords) > 0 else 0.0)
            for s1, s2 in zip(segment[:-1], segment[1:]):
                freqs = bands[(s1, s2)].frequencies
                qpoints = np.dot(bands[(s1, s2)].qpoints, icell_cv)
                qstep = np.linalg.norm(qpoints[-1, :] - qpoints[0, :])
                qaxis = np.linspace(0, qstep, qpoints.shape[0]) + label_coords[-1]
                qstep += label_coords[-1]
                spectrum.append(freqs)
                qps.append(qpoints)
                qpoint_axis.append(qaxis)
                label_coords.append(qstep)
                qpoint_labels.append(s2)
            jumps.append(label_coords[-1])
        jumps = jumps[:-1]

        spectrum = np.concatenate(spectrum, axis=0)  # unit in Thz
        # 1 Thz = 33.356 cm^-1
        if unit == r"cm$^{-1}$":
            spectrum *= 33.356
        qps = np.concatenate(qps, axis=0)
        qpoint_axis = np.concatenate(qpoint_axis, axis=0)
        atoms = self.structure.copy()
        sp = PhononSpectrum(
            qpoint_axis=qpoint_axis,
            frequencies=spectrum,
            atoms=atoms,
            label_coords=label_coords,
            qpoint_labels=qpoint_labels,
            jumps=jumps,
            unit=unit,
            bandpath=bp,
        )
        self._spectrum = sp

    @property
    def spectrum(self):
        """Phonon spectrum :class:`~aimstools.phonons.PhononSpectrum`"""

        if self._spectrum == None:
            self.set_spectrum(bandpath=None, unit=r"cm$^{-1}$")
        return self._spectrum

    def get_spectrum(self, bandpath=None, unit=r"cm$^{-1}$"):
        self.set_spectrum(bandpath=bandpath, unit=unit)
        return self.spectrum

    def read_dos(self):
        tdos = self.outputdir.joinpath("total_dos.dat")
        assert tdos.exists(), "File total_dos.dat not found."
        tdos = np.loadtxt(tdos)  # frequencies vs. dos
        return tdos

    def set_dos_spectrum(self, unit=r"cm$^{-1}$"):
        assert unit in [r"cm$^{-1}$", "Thz"], "Unit not recognized."
        dos = self.read_dos()
        if unit == r"cm$^{-1}$":
            dos[:, 0] *= 33.356
        dos = PhononDOS(
            atoms=self.structure.copy(),
            frequencies=dos[:, 0],
            contributions=dos[:, 1],
            unit=unit,
        )
        self._dos = dos

    @property
    def dos(self):
        """Phonon density of states :class:`~aimstools.phonons.PhononDOS`"""
        if self._dos == None:
            self.set_dos_spectrum()
        return self._dos

    def get_dos(self, unit=r"cm$^{-1}$"):
        self.set_dos_spectrum(unit=unit)
        return self.dos

    def _process_kwargs(self, kwargs):
        kwargs = kwargs.copy()

        deprecated = ["title", "mark_accoustic"]
        for dep in deprecated:
            if dep in kwargs.keys():
                kwargs.pop(dep)
                logger.warning(
                    f"Keyword {dep} is deprecated. Please do not use this anymore."
                )

        return kwargs

    def plot(self, axes=None, color=mutedblack, main=True, unit=r"cm$^{-1}$", **kwargs):
        """Plots phonon band structure. Similar syntax as :func:`~aimstools.bandstructures.regular_bandstructure.RegularBandStructure.plot`.

        Example:
            >>> from aimstools.phonons import FHIVibesPhonons as FVP
            >>> phon = FVP("path/to/dir")
            >>> phon.plot()

        Args:            
            axes (matplotlib.axes.Axes): Axes to draw on, defaults to None.
            figsize (tuple): Figure size in inches. Defaults to (5,5).
            filename (str): Saves figure to file. Defaults to None.
            spin (int): Spin channel, can be "up", "dn", 0 or 1. Defaults to 0.       
            bandpath (str): Band path for plotting of form "GMK,GA".
            unit (str): Energy unit, can be "cm$^{-1}$" or "Thz". Defaults to "cm$^{-1}$".
            show_grid_lines (bool): Show grid lines for axes ticks. Defaults to True.
            grid_lines_axes (str): Show grid lines for given axes. Defaults to "x".
            grid_linestyle (tuple): Grid lines linestyle. Defaults to (0, (1, 1)).
            grid_linewidth (float): Width of grid lines. Defaults to 1.0.
            grid_linecolor (str): Grid lines color. Defaults to mutedblack.
            show_jumps (bool): Show jumps between Brillouin zone sections by darker vertical lines. Defaults to True.
            jumps_linewidth (float): Width of jump lines. Defaults to mpllinewidth.
            jumps_linestyle (str): Line style of the jump lines. Defaults to "-".
            jumps_linecolor (str): Color of the jump lines. Defaults to mutedblack.
            show_bandstructure (bool): Show band structure lines. Defaults to True.
            bands_color (bool): Color of the band structure lines. Synonymous with color. Defaults to mutedblack.            
            bands_linewidth (float): Line width of band structure lines. Synonymous with linewidth. Defaults to mpllinewidth.         
            bands_linestyle (str): Band structure lines linestyle. Synonymous with linestyle. Defaults to "-".           
            bands_alpha (float): Band structure lines alpha channel. Synonymous with alpha. Defaults to 1.0.
            y_tick_locator (float): Places ticks on energy axis on regular intervals. Defaults to 100 cm^(-1).
            show_acoustic_bands (bool): Highlighs accoustic bands. Defaults to True.
            acoustic_bands_color (str): Color of the accoustic bands.
       
        Returns:
            axes: Axes object.        
        """
        kwargs = self._process_kwargs(kwargs)
        bandpath = kwargs.pop("bandpath", None)

        spectrum = self.get_spectrum(bandpath=bandpath, unit=unit)
        with AxesContext(ax=axes, main=main, **kwargs) as axes:
            pbs = PhononPlot(ax=axes, spectrum=spectrum, main=main, **kwargs)
            pbs.draw()
        return axes

    def plot_dos(
        self, axes=None, color=mutedblack, main=True, unit=r"cm$^{-1}$", **kwargs
    ):
        """Plots phonon density of states. Similar syntax as :func:`~aimstools.density_of_states.total_dos.TotalDOS.plot`.

        Example:
            >>> from aimstools.phonons import FHIVibesPhonons as FVP
            >>> phon = FVP("path/to/dir")
            >>> phon.plot_dos()
     
        Returns:
            axes: Axes object.        
        """
        kwargs = self._process_kwargs(kwargs)
        dos = self.get_dos(unit=unit)
        with AxesContext(ax=axes, main=main, **kwargs) as axes:
            pdos = PhononDOSPlot(ax=axes, dos=dos, main=main, **kwargs)
            pdos.draw()
        return axes

    def read_thermal_properties(self, unit="per mol"):
        """Reads data from thermal_properties.yaml .

        Regarding units, see: https://gitlab.com/vibes-developers/vibes/-/issues/41

        Args:
            unit (str): Either "per atom" or "per mol". "per atom" leaves the unit as given out by phonopy, "per mol" divides by the formula count.

        Note:
            The unit systems of free energy, heat capacity,
            and entropy are kJ/mol, J/K/mol, and J/K/mol, respectively,
            where 1 mol means Na× your input unit cell (not formula unit),
            i.e. you have to divide the value by number of formula unit in your
            unit cell by yourself. For example, in MgO (conventional) unit cell,
            if you want to compare with experimental results in kJ/mol,
            you have to divide the phonopy output by four.

        """

        assert unit in ["per atom", "per mol"], "Unit not recognized."

        outputyaml = self.outputdir.joinpath("thermal_properties.yaml")
        assert outputyaml.exists(), "File thermal_properties.yaml not found."
        with open(outputyaml, "r") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        ZPE = data["zero_point_energy"]  # kJ/ "mol"
        tp = data["thermal_properties"]
        temperature = np.array([k["temperature"] for k in tp], dtype=np.float32)  # K
        free_energy = np.array(
            [k["free_energy"] for k in tp], dtype=np.float32
        )  # kJ / "mol"
        entropy = np.array([k["entropy"] for k in tp], dtype=np.float32)  # J/K/ "mol"
        heat_capacity = np.array(
            [k["heat_capacity"] for k in tp], dtype=np.float32
        )  # J/K/ "mol"

        d = namedtuple(
            "thermal_properties",
            [
                "temperature",
                "ZPE",
                "free_energy",
                "entropy",
                "heat_capacity",
                "reference",
            ],
        )
        if unit == "per mol":
            cf = self.structure.get_chemical_formula()
            form = Formula(cf).stoichiometry()
            ref = form[1]
            count = form[2]
            ZPE /= count
            free_energy /= count
            entropy /= count
            heat_capacity /= count
            return d(temperature, ZPE, free_energy, entropy, heat_capacity, ref)
        elif unit == "per atom":
            return d(
                temperature,
                ZPE,
                free_energy,
                entropy,
                heat_capacity,
                self.structure.get_chemical_formula(),
            )

