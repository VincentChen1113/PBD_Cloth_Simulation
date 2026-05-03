### Fast Mass-Spring System Simulator
# PBD Cloth Simulation (CSE 328 Final Project)
Name: Vincent Chen
SBU ID: 115598737

This project is developed as a final project for  
**Stony Brook University – CSE 328: Computer Graphics**.

Originally forked from:  
https://github.com/sam007961/FastMassSpring

---

## Project Overview

This project extends a C++ implementation of *Fast Simulation of Mass-Spring Systems* [1], rendered with OpenGL.

The primary goal of this project is to implement a **Position-Based Dynamics (PBD)** solver for cloth simulation, based on Müller et al. (2007). A new solver will be implemented from scratch while reusing the rendering and mesh infrastructure from the original repository.

The current codebase now includes both:

- the original **fast mass-spring solver** for baseline comparison
- a separate **PBD cloth solver** with stretch, shear, bend, fixed-point, sphere collision, plane collision, self-collision, a dedicated floor-drop scene, and optional debug diagnostics

### Objectives

- Implement a **PBD solver** for cloth simulation  
- Support **constraint-based dynamics** (stretch, shear, bend, fixed points)  
- Add **collision handling** (proxy → mesh, if time permits)  
- Compare PBD with the original **mass-spring solver** [1] (optional)

---

## Fast Mass-Spring (Original Implementation)

This section demonstrates the original solver from the forked repository, based on Liu et al. (2013).

### Demo

![curtain_hang](https://user-images.githubusercontent.com/24758349/79005907-97ad1100-7b60-11ea-9e27-90375461beaf.gif)  
![curtain_ball](https://user-images.githubusercontent.com/24758349/79005924-9d0a5b80-7b60-11ea-8ce4-d9fc683441d7.gif)

---

## Position-Based Dynamics (PBD Implementation)

The current PBD solver includes:

- fixed-point constraints for pinned particles
- structural, shear, and bend constraints
- sphere and plane collision as generated inequality constraints
- self-collision detection and resolution with debug counters
- a dedicated `drop-floor` scene for floor interaction and tuning

### Demo

Available runtime scenes:

- `pbd hang`
- `pbd drop`
- `pbd drop-floor`

---

## Dependencies

* **OpenGL, freeGLUT, GLEW, GLM** for rendering  
* **OpenMesh** for computing normals  
* **Eigen** for sparse matrix algebra  

---

## Building

Install required dependencies (OpenGL, GLUT, GLEW), then:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Notes:

- Run the executable from the `build` directory so shader paths resolve correctly.
- The project fetches **OpenMesh**, **Eigen**, and **GLM** through CMake.
- Eigen is used as a header-only dependency in the current CMake setup.

## Running

Run the executable from the `build` directory so it can find the copied shader files:

```bash
cd build
./fast-mass-spring
```

The executable supports both long-form and short-form mode selection:

```bash
./fast-mass-spring [mass-spring|ms] [hang|drop] [--self-thickness value] [--debug]
./fast-mass-spring pbd [hang|drop|drop-floor] [--self-thickness value] [--debug]
```

Or equivalently:

```bash
./fast-mass-spring [ms-hang|ms-drop|pbd-hang|pbd-drop|pbd-drop-floor] [--self-thickness value] [--debug]
```

Examples:

```bash
./fast-mass-spring mass-spring hang
./fast-mass-spring ms drop
./fast-mass-spring pbd hang
./fast-mass-spring pbd drop
./fast-mass-spring pbd drop-floor
./fast-mass-spring pbd drop-floor --self-thickness 0.02
./fast-mass-spring pbd drop-floor --debug
```

Flags:

- `--self-thickness value`: overrides the PBD self-collision thickness with a positive float value
- `--debug`: enables debug diagnostics output. At the moment this is primarily useful for the PBD floor demo, where self-collision counters are printed to the terminal

If no arguments are provided, the program defaults to the mass-spring hanging cloth demo.

On Windows, you will likely need to specify the directories containing GLUT and GLEW in CMAKE_PREFIX_PATH so that cmake can find them.

``` bash
cmake .. -DCMAKE_PERFIX_PATH:PATH=/path/to/libs
```

You will also need to copy the DLLs to the build directory if they are not available globally.

---

## Task Board Checklist

Status snapshot for the current implementation:

## Setup / Integration
- [x] Build and understand existing framework
- [x] Add `PBDSolver` to project / CMake
- [x] Hook PBDSolver into app with solver mode switch
- [x] Preserve original mass-spring solver for comparison

## Core Solver Foundation
- [x] Implement `PBDSolver` skeleton
- [x] Initialize particle state (`x`, `p`, `v`, `invMass`)
- [x] Implement external force / gravity update
- [x] Implement damping
- [x] Implement position prediction / velocity reconstruction

## Constraint System
- [x] Implement fixed-point constraints
- [x] Implement structural distance constraints
- [x] Implement shear constraints
- [x] Implement bend constraints
- [x] Add stiffness / iteration-corrected stiffness handling

## Collision Handling
- [x] Implement plane collision
- [x] Implement sphere collision
- [x] Add explicit collision constraint generation stage
- [x] Tune collision robustness / epsilon

## Rendering / Output
- [x] Write updated positions to render buffer
- [x] Recompute normals
- [x] Verify shading / mesh integrity after simulation

## Testing / Validation
- [ ] Validate single-particle / two-particle cases
- [x] Validate hanging cloth behavior
- [x] Validate cloth drop / draping demo
- [x] Test timestep / iteration / resolution stability
- [ ] Compare against original solver

## Advanced / Stretch Goals
- [ ] Implement mesh collision
- [ ] Add wind / extra forces
- [ ] Add runtime parameter controls
- [ ] Benchmark performance
- [ ] Record comparison demos / metrics

## Final Deliverables
- [x] Stable PBD cloth demos
- [x] Updated README / documentation
- [ ] Demo video / GIF
- [ ] Final report / slides / citations

## License

This project is based on the original FastMassSpring repository by Samer Itani, 
licensed under the MIT License.

Modifications and extensions (including the PBD solver) are developed as part of 
the Stony Brook University CSE 328 final project.

### References

[1] Liu, T., Bargteil, A. W., Obrien, J. F., & Kavan, L. (2013). Fast simulation of mass-spring systems. *ACM Transactions on Graphics,32*(6), 1-7. doi:10.1145/2508363.2508406

[2] Provot, X. (1995). Deformation constraints in a mass-spring modelto describe rigid cloth behavior. *InGraphics Interface* 1995,147-154.

[3] Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007).  
*Position Based Dynamics.*  
In C. Mendoza & I. Navazo (Eds.), Proceedings of the 3rd Workshop in Virtual Reality Interactions and Physical Simulation (VRIPHYS 2006).
