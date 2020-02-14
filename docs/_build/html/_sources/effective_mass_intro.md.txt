# Effective Masses

Effective masses are implemented via band fitting, where estimates of the fitting parameters are obtained through finite differences. This eliminates the need to specify step sizes and partially remedies the dependency of the results on the density of the k-grid.

In order to evaluate effective masses, you require the eigenvalues on a regular, Gamma-centered grid from FHI-AIMS. This can be obtained by adding the following lines to the control.in file:

```bash
output postscf_eigenvalues
dos_kgrid_factors    8 8 8
```

The k-grid has to be sufficiently dense to obtain reliable results, hence the k-grid factors. Note that the Final_KS_eigenvalues.dat file one obtains from this calculation can become very large and the processing of the file might be very slow.

The effective mass tensor is:

```math
\mathbf{M}_{ij}^{-1} = \sum_{ij} \frac{\partial^2 E(\mathbf{k})}{\partial k_i \partial k_j}
```

In a first step, the elements of this tensor are evaluated with finite differences on a three-point stencil:

```math
\frac{\partial^2 E(\mathbf{k})}{\partial k_i \partial k_j} 
\approx \frac{E(h_i, h_j) - E(h_i, -h_j) - E(-h_i, h_j) + E(-h_i, -h_j)}{4 h_i h_j} + \mathcal{O}(h^2)
```

The step size *h* is automatically chosen from the minimal distance between points on the k-grid along each direction, hence, the quality of the finite difference eigenvalues depend strongly on the grid density. The components of the tensor are used as starting guesses to fit the following ellipsoid around the extreme:

```math
E(\mathbf{k}) \approx g(\mathbf{k}) = m_{xx} x^2 + m_{yy} y^2 + m_{zz} z^2 + m_{xy} xy + m_{xz} xz + m_{yz} yz + m_x x + m_y y + m_z z
```

The residual function in the least-squares fit is defined as:

```math
R = \text{min} \left( \sum_{\mathbf{k}} \left| \left[E(\mathbf{k}) - g(\mathbf{k})\right] \cdot w(\mathbf{k}) \right|^2 \right)\\
~~\text{with } w(\mathbf{k}) = \frac{1}{1 + a \cdot |\mathbf{k}|}
```

The weight factor reduces the impact of points far away from the extreme point.

The derivatives of the fitted function built the effective mass tensor:

```math
\mathbf{M}_{ij}^{-1} = \begin{pmatrix} 
\frac{\partial^2 g(\mathbf{k})}{\partial x^2}          & \frac{\partial^2 g(\mathbf{k})}{\partial x \partial y} & \frac{\partial^2 g(\mathbf{k})}{\partial x \partial z} \\
\frac{\partial^2 g(\mathbf{k})}{\partial x \partial y} & \frac{\partial^2 g(\mathbf{k})}{\partial y^2}          & \frac{\partial^2 g(\mathbf{k})}{\partial y \partial z} \\
\frac{\partial^2 g(\mathbf{k})}{\partial x \partial z} & \frac{\partial^2 g(\mathbf{k})}{\partial y \partial z} & \frac{\partial^2 g(\mathbf{k})}{\partial z^2} 
\end{pmatrix}
=
\begin{pmatrix} 
2 m_{xx} & m_{xy} & m_{xz} \\
m_{xy} & 2 m_{yy} & m_{yz} \\
m_{xz} & m_{yz} & 2 m_{zz} 
\end{pmatrix}
```
  
This tensor is inverted and diagonalized. The eigenvalues correspond to the effective masses and the eigenvectors to their principal directions.

The conductivity effective mass for the dimensionality *d* is then calculated as:

```math
\mu_{cond.} = d \cdot \left(\sum_n^d \frac{1}{e_n}\right)~\text{with } d=2,3
```