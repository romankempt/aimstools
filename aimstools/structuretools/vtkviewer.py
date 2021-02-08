from aimstools.misc import *

from ase.data.colors import jmol_colors
from ase.data import chemical_symbols, covalent_radii, vdw_radii
import ase.io
import numpy as np
from ase import neighborlist

import vtk

atom_colors = dict(zip(chemical_symbols, jmol_colors))
atom_radii = dict(zip(chemical_symbols, covalent_radii))

diffuse = 0.7
specular = 0.4
specular_power = 20


class VTKViewer:
    def __init__(self, structure) -> None:
        self.structure = structure
        self.bonds = self.get_bonds(structure)

    def get_bonds(self, atoms):
        atoms = atoms.copy()
        nl = neighborlist.NeighborList(
            ase.neighborlist.natural_cutoffs(atoms),
            self_interaction=False,
            bothways=True,
        )
        nl.update(atoms)
        connectivity_matrix = nl.get_connectivity_matrix(sparse=False)

        con_tuples = {}  # connected first neighbors
        for row in range(connectivity_matrix.shape[0]):
            con_tuples[row] = []
            for col in range(connectivity_matrix.shape[1]):
                if connectivity_matrix[row, col] == 1:
                    con_tuples[row].append(col)

        pairs = []  # cleaning up the first neighbors
        for index in con_tuples.keys():
            for value in con_tuples[index]:
                if index > value:
                    pairs.append((index, value))
                elif index <= value:
                    pairs.append((value, index))
        pairs = set(pairs)
        return pairs

    def _get_bond_coordinates(self):
        atoms = self.structure.copy()
        bonds = self.bonds.copy()
        points = []
        for i, j in bonds:
            start = atoms[i].position.copy()
            end = atoms[j].position.copy()
            points.append(tuple((start, end)))
        return points

    def get_sphere_actors(self, center, symbol, radius=None):
        radius = atom_radii[symbol] * 0.6 if not bool(radius) else radius
        # create a Sphere
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(radius)
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)

        # create a mapper
        sphereMapper = vtk.vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

        # create an actor
        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(sphereMapper)
        sphereActor.GetProperty().SetColor(atom_colors[symbol])
        sphereActor.GetProperty().SetDiffuse(diffuse)
        sphereActor.GetProperty().SetSpecular(specular)
        sphereActor.GetProperty().SetSpecularPower(specular_power)
        return sphereActor

    def get_line_actors(self, unitcell):

        uc = unitcell.copy()
        v1, v2, v3 = uc[0, :], uc[1, :], uc[2, :]
        points = []
        points.append([0] * 3)  # origin 0
        points.append(v1)  # 1
        points.append(v2)  # 2
        points.append(v3)  # 3
        points.append(v1 + v2)  # 4
        points.append(v2 + v3)  # 5
        points.append(v1 + v3)  # 6
        points.append(v1 + v2 + v3)  # 7

        pts = vtk.vtkPoints()
        for p in points:
            pts.InsertNextPoint(p)

        vertices = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 6),
            (2, 4),
            (1, 5),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        lines = vtk.vtkCellArray()

        for v1, v2 in vertices:
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(12)
            line.GetPointIds().SetId(v1, v2)
            lines.InsertNextCell(line)

        linesPolyData.SetLines(lines)
        # Setup the visualization pipeline
        namedColors = vtk.vtkNamedColors()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for i in range(12):
            colors.InsertNextTypedTuple(namedColors.GetColor3ub("Black"))

        linesPolyData.GetCellData().SetScalars(colors)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(4)
        return actor

    def get_tube_actors(self, start, end, symbol, radius=0.1):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(start)
        lineSource.SetPoint2(end)

        # Setup actor and mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputConnection(lineSource.GetOutputPort())

        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)

        # Create tube filter
        tubeFilter = vtk.vtkTubeFilter()
        tubeFilter.SetInputConnection(lineSource.GetOutputPort())
        tubeFilter.SetRadius(radius)
        tubeFilter.SetNumberOfSides(50)
        tubeFilter.Update()

        # Setup actor and mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())

        tubeActor = vtk.vtkActor()
        tubeActor.SetMapper(tubeMapper)
        tubeActor.GetProperty().SetColor(atom_colors[symbol])
        tubeActor.GetProperty().SetDiffuse(diffuse)
        tubeActor.GetProperty().SetSpecular(specular)
        tubeActor.GetProperty().SetSpecularPower(specular_power)
        # Make the tube have some transparency.
        # tubeActor.GetProperty().SetOpacity(0.5)

        return tubeActor

    def view(self):
        atoms = self.structure.copy()
        bonds = self.bonds.copy()
        bond_points = self._get_bond_coordinates()
        vtkcolors = vtk.vtkNamedColors()

        sphereActors = []
        for sym, pos in zip(atoms.symbols, atoms.positions):
            sphereActors.append(self.get_sphere_actors(pos, sym))

        bondActors = []
        for indices, p in zip(bonds, bond_points):
            i, j = indices
            sym1, sym2 = atoms[i].symbol, atoms[j].symbol
            p1, p2 = p
            middle = (p2 + p1) / 2
            a = self.get_tube_actors(p1, middle, symbol=sym1)
            bondActors.append(a)
            b = self.get_tube_actors(p2, middle, symbol=sym2)
            bondActors.append(b)

        # a renderer and render window
        renderer = vtk.vtkRenderer()
        renderer.SetUseFXAA(True)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName("Axes")
        renderWindow.AddRenderer(renderer)

        # an interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # add the actors to the scene
        for s in sphereActors:
            renderer.AddActor(s)
        for b in bondActors:
            pass
            # renderer.AddActor(b)

        boxActor = self.get_line_actors(atoms.cell)
        renderer.AddActor(boxActor)

        renderer.SetBackground(vtkcolors.GetColor3d("White"))

        # transform = vtk.vtkTransform()
        # transform.Translate(5.0, 0.0, 0.0)

        # axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
        # axes.SetUserTransform(transform)

        # properties of the axes labels can be set as follows
        # this sets the x axis label to red
        # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d('Red'));

        # the actual text of the axis label can be changed:
        # axes->SetXAxisLabelText('test');

        # renderer.AddActor(axes)

        renderer.GetActiveCamera().Azimuth(50)
        renderer.GetActiveCamera().Elevation(-30)

        renderer.ResetCamera()
        renderWindow.SetWindowName("Name")
        renderWindow.SetSize(500, 500)
        renderWindow.Render()

        # begin mouse interaction
        renderWindowInteractor.Start()

        def OnClose(window, event):
            print("OnClose()")
            # ask the window to close
            window.Finalize()

        renderWindow.AddObserver("ExitEvent", OnClose)
