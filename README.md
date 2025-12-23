# Fluid–Structure Interaction (FSI) — Monolithic Solver (deal.II)

This project implements a **monolithic solver** for a steady **fluid–structure interaction (FSI)** problem. The computational domain is split into a fluid region and a solid region, coupled along the interface Σ.


This is part of a project-work for Numerical Methods for Partial Differential Equations course [@Polimi](https://www.polimi.it/)

### Authors
Project developed by:
- [Infascelli Riccardo](https://github.com/RiccardoInfascelli)
- [Francesco Rosnati](https://github.com/RosNaviGator)
- [Santoro Dario](https://github.com/DarioSantoroDS)

## Overview

The mathematical model is:
```math
\begin{cases}
    -2 \nu \, \nabla \cdot \boldsymbol{\varepsilon}(\boldsymbol{u}) + \nabla p = 0, & \text{in } \Omega_{\text{fluid}}, \\[6pt]
    
    \nabla \cdot \boldsymbol{u} = 0, 
    & \text{in } \Omega_{\text{fluid}}, \\[6pt]
    
    -\nabla \cdot \sigma(\boldsymbol{d}) = 0, 
    & \text{in } \Omega_{\text{solid}}, \\[6pt]
    
    \boldsymbol{u} = \boldsymbol{0}, 
    & \text{on } \Sigma = \partial\Omega_{\text{fluid}}\cap\partial\Omega_{\text{solid}}, \\[6pt]
    
    \sigma(\boldsymbol{d})\boldsymbol{n} 
    = \bigl( 2 \nu \, \boldsymbol{\varepsilon}(\boldsymbol{u}) - p\,\boldsymbol{I} \bigr)\boldsymbol{n}, 
    & \text{on } \Sigma.

\end{cases}
```

Under the small-displacement assumption, the fluid velocity is forced to vanish on the fluid–solid interface.<!-- TO CHECK -->


## Main Features
- hp-FEM discretization with different finite elements for fluid and solid
- One-way coupled monolithic formulation
- Constraint handling for boundary conditions and fluid–solid continuity 
- Adaptive mesh refinement
- 2D and 3D implementation
- Parallel implementation with MPI and Trilinos


## Project Structure

- `src/fsi.cpp` : entry point of the program, sets up and runs the FSI solver.
- `src/FluidStructureProblem.hpp` : header file defining the `FluidStructureProblem` class and its interface.
- `src/FluidStructureProblem.cpp` : implementation of the FSI solver including mesh generation, DOF setup, system assembly, interface treatment, solution, and output.
- `config.prm` : configuration file for simulation parameters.
- `original46.cpp` : original deal.II step-46 tutorial code for reference and comparison.
- `CMakeLists.txt` : CMake build configuration file.
- `.gdbinit` and `.clang-format` : gdb and formatter configuration files.
- `README.md` : this documentation file.

## Configuration
The simulation parameters are configured in `config.prm`.

### Geometry
Controls the mesh resolution and the polynomial degree of the Finite Elements.
Configurable parameters:
- **Number of elements per edge:** Number of elements along each edge of the square/cube
- **Stokes degree:** Degree of the finite elements used for both fluid domains. The program already add one degree for the velocity space.
- **Elasticity degree:** Degree of the finite elements used for solid domains.
* **Fluid/Solid weights:** Only required if weighted triangulation partitioning is enabled.
### Refinement
Controls the mesh refinement strategy.
Configurable parameters:
- **Refinement cycles:** Number of refinement cycles to perform.

### Physics
Controls the physical parameters of the simulation.
Configurable parameters:
- **Viscosity:** Viscosity of the fluid.
- **Mu:** First Lamé parameter for the solid.
- **Lambda:** Second Lamé parameter for the solid.

## Build and Run

```bash
mkdir build
cd build
cmake ..
make 
./fsi 
```

## Generated Files

- `solution_0*.pvtu` : ParaView-readable file with the velocity, pressure, and displacement fields; also present the material_id and subdomain_id fields. It can be visualized in ParaView. Multiple files are generated, one pvtu for each refinement step, and in case of parallel execution, one vtu for each MPI process.


## References
Each reference can be found in the report.
The original repository is available here: https://github.com/DarioSantoroDS/PDE_Project
