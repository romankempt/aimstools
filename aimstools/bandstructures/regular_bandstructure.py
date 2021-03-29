from logging import raiseExceptions
from aimstools.misc import *
from aimstools.bandstructures.base import BandStructureBaseClass
from aimstools.bandstructures.utilities import BandStructurePlot, BandSpectrum

from ase.dft.kpoints import parse_path_string

from collections import namedtuple

import numpy as np


class RegularBandStructure(BandStructureBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        self.soc = soc
        self.spin = "none" if self.control["spin"] != "collinear" else "collinear"
        self.task = "band structure"
        self.band_sections = self.band_sections.regular
        self._set_bandpath_from_sections()
        if self.spin == "none":
            bandfiles = self.get_bandfiles(spin="none", soc=soc)
            bandfiles = bandfiles.regular
            self.bands = self.read_bandfiles(zip(bandfiles, bandfiles))
        else:
            if soc:
                logger.warning(
                    "Past Roman to future Roman: Should I just sum up the spin channels in case of SOC?"
                )
            bandfiles_dn = self.get_bandfiles(spin="dn", soc=soc).regular
            bandfiles_up = self.get_bandfiles(spin="up", soc=soc).regular
            self.bands = self.read_bandfiles(zip(bandfiles_dn, bandfiles_up))
        self._spectrum = self.set_spectrum(None, None)

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def read_bandfiles(self, bandfiles):
        bands = {}
        b = namedtuple("band", ["kpoints", "occupations", "eigenvalues"])
        nspins = 2 if self.spin == "collinear" else 1
        for section, bandfile in zip(self.band_sections, bandfiles):
            ev, occ, rocc, revs = [], [], [], []
            for s in range(nspins):
                bf = bandfile[s]
                # index, k1, k2, k3, occ, ev, occ, ev ...
                data = np.loadtxt(bf)[:, 1:]
                points = data[:, :3]
                occupations = data[:, 3:-2:2]
                eigenvalues = data[:, 4:-1:2]
                rpoints = points[::-1].copy()
                roccs = occupations[::-1].copy()
                reigvs = eigenvalues[::-1].copy()
                ev.append(eigenvalues)
                revs.append(reigvs)
                occ.append(occupations)
                rocc.append(roccs)
            ev = np.stack(ev, axis=1)
            occ = np.stack(occ, axis=1)
            rocc = np.stack(rocc, axis=1)
            revs = np.stack(revs, axis=1)
            band_forward = b(points, occ, ev)
            band_backward = b(rpoints, rocc, revs)
            pathsegment = (section.symbol1, section.symbol2)
            pathsegment_r = (section.symbol2, section.symbol1)
            bands[pathsegment] = band_forward
            bands[pathsegment_r] = band_backward
        return bands

    def set_spectrum(self, bandpath=None, reference=None):
        bands = self.bands
        if bandpath != None:
            self.set_bandpath(bandpath)
        bp = parse_path_string(self.bandpath.path)
        jumps = []
        kps = []
        occs = []
        kpoint_axis = []
        kpoint_labels = []
        label_coords = []
        spectrum = []
        icell_cv = 2 * np.pi * np.linalg.pinv(self.structure.cell).T
        for segment in bp:
            kpoint_labels.append(segment[0])
            label_coords.append(label_coords[-1] if len(label_coords) > 0 else 0.0)
            for s1, s2 in zip(segment[:-1], segment[1:]):
                energies = bands[(s1, s2)].eigenvalues
                kpoints = np.dot(bands[(s1, s2)].kpoints, icell_cv)
                occ = bands[(s1, s2)].occupations
                kstep = np.linalg.norm(kpoints[-1, :] - kpoints[0, :])
                kaxis = np.linspace(0, kstep, kpoints.shape[0]) + label_coords[-1]
                kstep += label_coords[-1]
                spectrum.append(energies)
                kps.append(kpoints)
                occs.append(occ)
                kpoint_axis.append(kaxis)
                label_coords.append(kstep)
                kpoint_labels.append(s2)
            jumps.append(label_coords[-1])
        jumps = jumps[:-1]

        spectrum = np.concatenate(spectrum, axis=0)
        kps = np.concatenate(kps, axis=0)
        kpoint_axis = np.concatenate(kpoint_axis, axis=0)
        occs = np.concatenate(occs, axis=0)

        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        self.set_energy_reference(reference, self.soc)
        reference, shift = self.energy_reference
        sp = BandSpectrum(
            atoms=self.structure.atoms,
            kpoints=kps,
            kpoint_axis=kpoint_axis,
            eigenvalues=spectrum,
            occupations=occs,
            label_coords=label_coords,
            kpoint_labels=kpoint_labels,
            jumps=jumps,
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
            bandpath=bandpath,
        )
        self._spectrum = sp

    @property
    def spectrum(self):
        if self._spectrum == None:
            self.set_spectrum(bandpath=None, reference=None)
        return self._spectrum

    def get_spectrum(self, bandpath=None, reference=None):
        self.set_spectrum(reference=reference, bandpath=bandpath)
        return self.spectrum

    def _process_kwargs(self, kwargs):
        kwargs = kwargs.copy()
        spin = kwargs.pop("spin", None)

        deprecated = ["title", "mark_fermi_level", "mark_band_gap"]
        for dep in deprecated:
            if dep in kwargs.keys():
                kwargs.pop(dep)
                logger.warning(
                    f"Keyword {dep} is deprecated. Please do not use this anymore."
                )

        kwargs["spin"] = self.spin2index(spin)

        return kwargs

    def plot(self, axes=None, **kwargs):
        """Main function to handle plotting of band structures. Supports all keywords of :func:`~aimstools.bandstructures.regular_bandstructure.RegularBandStructure.plot`.

        Example:
            >>> from aimstools.bandstructures import RegularBandStructure as RBS
            >>> bs = RBS("path/to/dir")
            >>> bs.plot()

        Args:            
            axes (matplotlib.axes.Axes): Axes to draw on, defaults to None.
            figsize (tuple): Figure size in inches. Defaults to (5,5).
            filename (str): Saves figure to file. Defaults to None.
            spin (int): Spin channel, can be "up", "dn", 0 or 1. Defaults to 0.       
            bandpath (str): Band path for plotting of form "GMK,GA".
            reference (str): Energy reference for plotting, e.g., "VBM", "middle", "fermi level". Defaults to None.
            show_fermi_level (bool): Show Fermi level. Defaults to True.
            fermi_level_color (str): Color of Fermi level line. Defaults to fermi_color.
            fermi_level_alpha (float): Alpha channel of Fermi level line. Defaults to 1.0.
            fermi_level_linestyle (str): Line style of Fermi level line. Defaults to "--".
            fermi_level_linewidth (float): Line width of Fermi level line. Defaults to mpllinewidth.
            show_grid_lines (bool): Show grid lines for axes ticks. Defaults to True.
            grid_lines_axes (str): Show grid lines for given axes. Defaults to "x".
            grid_linestyle (tuple): Grid lines linestyle. Defaults to (0, (1, 1)).
            grid_linewidth (float): Width of grid lines. Defaults to 1.0.
            show_jumps (bool): Show jumps between Brillouin zone sections by darker vertical lines. Defaults to True.
            jumps_linewidth (float): Width of jump lines. Defaults to mpllinewidth.
            jumps_linestyle (str): Line style of the jump lines. Defaults to "-".
            jumps_linecolor (str): Color of the jump lines. Defaults to mutedblack.
            show_bandstructure (bool): Show band structure lines. Defaults to True.
            bands_color (bool): Color of the band structure lines. Synonymous with color. Defaults to mutedblack.            
            bands_linewidth (float): Line width of band structure lines. Synonymous with linewidth. Defaults to mpllinewidth.         
            bands_linestyle (str): Band structure lines linestyle. Synonymous with linestyle. Defaults to "-".           
            bands_alpha (float): Band structure lines alpha channel. Synonymous with alpha. Defaults to 1.0.
            show_bandgap_vertices (bool): Show direct and indirect band gap transitions. Defaults to True.
            window (tuple): Window on energy axis, can be float or tuple of two floats in eV. Defaults to 3 eV.
            y_tick_locator (float): Places ticks on energy axis on regular intervals. Defaults to 0.5 eV.
       
        Returns:
            axes: Axes object.        
        """
        kwargs = self._process_kwargs(kwargs)
        bandpath = kwargs.pop("bandpath", None)
        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(bandpath=bandpath, reference=reference)

        with AxesContext(ax=axes, **kwargs) as axes:
            bs = BandStructurePlot(ax=axes, spectrum=spectrum, **kwargs)
            bs.draw()

        return axes
