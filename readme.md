# PBD Cloth Simulation

**Name:** Vincent Chen  
**SBU ID:** 115598737  
**Course:** Stony Brook University — CSE 328: Computer Graphics

This project was originally forked from [sam007961/FastMassSpring](https://github.com/sam007961/FastMassSpring).

---

## Project Summary

This project extends a C++ OpenGL implementation of **Fast Simulation of Mass-Spring Systems** [1] by adding a new **Position-Based Dynamics (PBD)** cloth solver based on Müller et al. [3].

The original fast mass-spring solver is preserved for comparison, while the new PBD solver is implemented as a separate solver using the existing rendering, mesh, shader, and interaction infrastructure.

The project currently includes:

- the original fast mass-spring hanging and sphere-drop demos
- a separate PBD cloth solver
- constraint-based stretch, shear, bend, and fixed-point handling
- sphere, plane/floor, and box obstacle collision
- vertex-triangle self-collision with diagnostics
- floor-drop and dual-obstacle stress demos
- a flag-style wind demo using aerodynamic drag, lift, gusts, and procedural noise
- an interactive runtime tuning interface for the PBD hanging demo

---

## Key Features

- Original **fast mass-spring** solver preserved for comparison
- New **Position-Based Dynamics** solver
- Fixed-point constraints for pinned/dragged particles
- Structural and shear distance constraints
- Dihedral-angle bending constraints
- Iteration-corrected stiffness handling
- Sphere collision
- Plane/floor collision
- Analytic box collision with edge-midpoint obstacle sampling
- Vertex-triangle self-collision with debug counters
- Dedicated floor-drop and dual-obstacle scenes
- Flag-style wind demo with drag, lift, gust modulation, and procedural flutter noise
- Runtime tuning interface for the PBD hanging cloth demo

---

## Build Instructions

### Dependencies

- **OpenGL, freeGLUT, GLEW, GLM** for rendering
- **OpenMesh** for computing normals
- **Eigen** for vector/matrix operations

Install the required OpenGL/GLUT/GLEW dependencies, then build with CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Notes:

- Run the executable from the `build` directory so shader paths resolve correctly.
- Shader files are copied from `ClothApp/shaders` into `build/shaders` during each build.
- The project fetches **OpenMesh**, **Eigen**, and **GLM** through CMake.
- Eigen is used as a header-only dependency in the current CMake setup.

On Windows, you may need to specify the directories containing GLUT and GLEW in `CMAKE_PREFIX_PATH`:

```bash
cmake .. -DCMAKE_PREFIX_PATH:PATH=/path/to/libs
```

You may also need to copy the required DLLs into the build directory if they are not available globally.

---

## Running

Run the executable from the `build` directory:

```bash
cd build
./fast-mass-spring
```

If no arguments are provided, the program defaults to the original mass-spring hanging cloth demo.

### Command Syntax

Long-form mode selection:

```bash
./fast-mass-spring [mass-spring|ms] [hang|drop] [--self-thickness value] [--debug]
./fast-mass-spring pbd [hang|hang-wind|drop|drop-floor|drop-floor-dual] [--self-thickness value] [--debug] [--wind-speed value] [--iters value] [--radius value] [--dt value]
```

Short-form mode selection:

```bash
./fast-mass-spring [ms-hang|ms-drop|pbd-hang|pbd-hang-wind|pbd-drop|pbd-drop-floor|pbd-drop-floor-dual] [--self-thickness value] [--debug] [--wind-speed value] [--iters value] [--radius value] [--dt value]
```

### Demo Modes

| Demo | Command | Purpose |
|---|---|---|
| Mass-spring hang | `./fast-mass-spring ms hang` | Original baseline hanging cloth |
| Mass-spring drop | `./fast-mass-spring ms drop` | Original baseline sphere-drop demo |
| PBD hang | `./fast-mass-spring pbd hang --iters 20` | Interactive constraint tuning demo |
| PBD drop | `./fast-mass-spring pbd drop` | PBD cloth dropping onto a sphere |
| PBD drop-floor | `./fast-mass-spring pbd drop-floor --iters 28 --dt 0.003` | Floor contact and self-collision stress test |
| PBD drop-floor-dual | `./fast-mass-spring pbd drop-floor-dual --iters 28` | Cloth interaction with sphere and box obstacles |
| PBD hang-wind | `./fast-mass-spring pbd hang-wind --wind-speed 5` | Flag-style wind demo |

### Example Commands

```bash
./fast-mass-spring mass-spring hang
./fast-mass-spring ms drop
./fast-mass-spring pbd hang --iters 20
./fast-mass-spring pbd drop
./fast-mass-spring pbd drop-floor
./fast-mass-spring pbd drop-floor --iters 28 --dt 0.003
./fast-mass-spring pbd drop-floor --debug
./fast-mass-spring pbd drop-floor --self-thickness 0.02
./fast-mass-spring pbd drop-floor-dual
./fast-mass-spring pbd drop-floor-dual --iters 28
./fast-mass-spring pbd hang-wind --wind-speed 5
./fast-mass-spring pbd-hang-wind --wind-speed 8
```

---

## Controls and Runtime Interface

### PBD Hang Controls

The `pbd hang` demo includes an interactive tuning interface for studying how different constraints affect cloth behavior.

Startup option:

- `--iters value`: sets the number of PBD projection passes per timestep for the hang demo; accepts an integer in `[1, 80]`

Keyboard controls:

| Key | Action |
|---|---|
| `1 / 2` | Decrease / increase stretch stiffness |
| `3 / 4` | Decrease / increase shear stiffness |
| `5 / 6` | Decrease / increase bend stiffness |
| `7 / 8` | Decrease / increase damping factor |
| `R` | Reset cloth to the initial hanging pose while keeping current tuning values |
| `T` | Reset hang-demo tuning values to defaults |
| `P` | Pause / resume simulation |

Parameter meaning:

- **Stretch stiffness** controls structural distance constraints.
- **Shear stiffness** controls diagonal/shear distance constraints.
- **Bend stiffness** controls dihedral bending constraints.
- **Damping** controls velocity energy decay.
- **Iterations** control the number of PBD projection passes per timestep.

The current hang-demo values are displayed in the on-screen overlay while the demo is running.

### PBD Drop Controls

The `pbd drop` demo includes an interactive tuning interface for sphere collision, material tuning, damping, and mesh-resolution comparison.

Startup options:

- `--radius value`: sets the sphere collider radius for the drop demo; accepts a float in `[0.1, 1.5]`
- `--iters value`: sets the number of PBD projection passes per timestep for the drop demo; accepts an integer in `[1, 80]`

Keyboard controls:

| Key | Action |
|---|---|
| `1 / 2` | Decrease / increase stretch stiffness |
| `3 / 4` | Decrease / increase shear stiffness |
| `5 / 6` | Decrease / increase bend stiffness |
| `7 / 8` | Decrease / increase damping factor |
| `[` / `]` | Decrease / increase the pending mesh resolution by `2` |
| `R` | Reset cloth and apply the pending mesh resolution |
| `T` | Reset drop-demo tuning values to defaults and reset pending mesh resolution to `33` |
| `P` | Pause / resume simulation |

Parameter meaning:

- **Sphere radius** changes the collision equation $C(p) = \lVert p - c \rVert - r \ge 0$.
- **Stretch stiffness** controls structural distance constraints.
- **Shear stiffness** controls diagonal/shear distance constraints.
- **Bend stiffness** controls dihedral bending constraints.
- **Damping** controls velocity energy decay.
- **Iterations** control the number of PBD projection passes per timestep.
- **Mesh resolution** changes particle and constraint count and is applied only when the cloth system is rebuilt on reset.

The current and pending mesh resolutions are shown in the on-screen overlay. If the pending mesh differs from the current mesh, the overlay prints `Pending mesh: NxN, press R to apply`.

### PBD Drop-Floor Controls

The `pbd drop-floor` demo includes an interactive tuning interface for self-collision robustness, floor response, timestep speed, and mesh-resolution comparison.

Startup options:

- `--iters value`: sets the number of PBD projection passes per timestep for the drop-floor demo; accepts an integer in `[1, 80]`
- `--dt value`: sets the simulation timestep for the drop-floor demo; accepts a float in `[0.001, 0.01]`

Keyboard controls:

| Key | Action |
|---|---|
| `1 / 2` | Decrease / increase self-collision stiffness |
| `3 / 4` | Decrease / increase max self-collision contacts per vertex |
| `5 / 6` | Decrease / increase self-collision thickness |
| `7 / 8` | Decrease / increase floor friction |
| `Q / W` | Decrease / increase bend stiffness |
| `A / S` | Decrease / increase damping factor |
| `[` / `]` | Decrease / increase the pending mesh resolution |
| `9 / 0` | Step to a slower / faster timestep preset |
| `R` | Reset cloth and apply the pending mesh resolution |
| `T` | Reset drop-floor tuning values to stable defaults |
| `P` | Pause / resume simulation |
| `D` | Toggle debug diagnostics |

Parameter meaning:

- **Self-collision stiffness** controls how strongly vertex self-collision constraints are projected apart.
- **Max self-collision contacts** limits how many self-collision contacts are processed per vertex each solver step.
- **Self-collision thickness** is the minimum separation band enforced between cloth layers.
- **Floor friction** damps tangential sliding after plane contact.
- **Bend stiffness** controls dihedral bending constraints.
- **Damping** controls velocity energy decay.
- **Iterations** control the number of PBD projection passes per timestep.
- **Timestep** changes simulation speed and contact robustness; faster presets are less stable and are labeled in the overlay.
- **Mesh resolution** changes particle and constraint count and is applied only when the cloth system is rebuilt on reset.

The overlay shows the current self-collision, floor, timestep, and mesh settings. If the pending mesh differs from the current mesh, the overlay prints `Pending mesh: NxN, press R to apply`.

### PBD Drop-Floor-Dual Controls

The `pbd drop-floor-dual` demo includes a simple interactive interface for comparing how the cloth drapes over a smooth sphere versus a sharp cube.

Startup option:

- `--iters value`: sets the number of PBD projection passes per timestep for the dual-obstacle demo; accepts an integer in `[1, 80]`

Keyboard controls:

| Key | Action |
|---|---|
| `1 / 2` | Decrease / increase stretch stiffness |
| `3 / 4` | Decrease / increase shear stiffness |
| `5 / 6` | Decrease / increase bend stiffness |
| `[` / `]` | Decrease / increase the pending mesh resolution |
| `Q / W` | Decrease / increase the pending sphere radius |
| `A / S` | Decrease / increase the pending cube size |
| `9 / 0` | Step to a slower / faster timestep preset |
| `R` | Reset cloth and apply the pending mesh and obstacle sizes |
| `T` | Reset material values and pending mesh/object sizes to defaults |
| `P` | Pause / resume simulation |

Parameter meaning:

- **Stretch stiffness** controls structural distance constraints.
- **Shear stiffness** controls diagonal/shear distance constraints.
- **Bend stiffness** controls dihedral bending constraints.
- **Mesh resolution** changes particle, constraint, and contact count and is applied only when the cloth system is rebuilt on reset.
- **Sphere radius** changes the smooth obstacle collision size.
- **Cube size** changes the sharp analytic box collider and visual cube together.
- **Iterations** control the number of PBD projection passes per timestep.
- **Timestep** changes simulation speed and contact robustness; faster presets are less stable and are labeled in the overlay.

The overlay shows the current material values, timestep, and currently applied obstacle sizes. If the pending mesh, sphere radius, or cube size differs from the current state, the overlay prints `Pending changes: press R to apply`.

### Wind Demo Controls

The `pbd hang-wind` demo configures the cloth like a flag:

- one side edge is pinned like cloth attached to a pole
- wind blows horizontally, perpendicular to gravity
- aerodynamic forcing uses triangle-based drag and lift
- gust and noise terms modulate the base wind speed over time

Required startup option:

- `--wind-speed value`: required for `pbd hang-wind`; accepts a float in `[0, 15]`

The wind speed is the base horizontal wind speed used by the drag/lift/gust model.

### Debug Flags

- `--debug`: enables PBD diagnostic output and prints self-collision counters to the terminal
- `--self-thickness value`: overrides the PBD self-collision thickness with a positive float value

---

## Implementation Details

### PBD Solver Pipeline

The PBD solver follows the standard position-based simulation pipeline:

1. Apply external forces
2. Apply damping
3. Predict positions
4. Generate collision constraints
5. Iteratively project persistent and generated constraints
6. Update velocities from projected positions
7. Apply post-collision velocity damping/friction
8. Commit positions to the render buffer

### Constraint Formulations

The solver currently supports:

- **Fixed-point constraints** for pinned and interactively dragged particles
- **Structural distance constraints** for edge-length preservation
- **Shear distance constraints** for diagonal deformation control
- **Dihedral bending constraints** for fold-angle preservation
- **Iteration-corrected stiffness** so the effective stiffness is more consistent when changing the number of solver iterations

Stretch and shear constraints are distance-based. Bending is handled with a dihedral-angle constraint over adjacent triangle pairs rather than only using longer distance springs.

### Collision Handling

The solver includes several collision types:

- **Sphere collision** as an inequality constraint
- **Plane/floor collision** as an inequality constraint
- **Analytic box collision** using face contacts and closest-point style handling
- **Edge-midpoint sampling** for improving box obstacle collision near sharp edges
- **Vertex-triangle self-collision** for cloth self-intersection handling

Collision constraints are generated from predicted positions each timestep, then solved together with the persistent cloth constraints during projection.

### Self-Collision Diagnostics

When `--debug` is enabled, the solver prints self-collision statistics such as:

- generated self-collision contacts
- initially violated contacts
- remaining violated contacts after projection
- maximum penetration before projection
- maximum penetration after projection

These diagnostics are useful for determining whether a self-collision problem is caused by missed contact generation or insufficient projection convergence.

### Wind Model

The wind demo implements wind as an external aerodynamic force rather than a full wind-field solver.

For each cloth triangle, the solver computes:

- triangle face normal
- triangle face area
- average triangle velocity
- wind velocity
- relative velocity between the cloth face and the air

The aerodynamic model includes:

- **drag**, which acts opposite the relative velocity
- **lift**, which acts perpendicular to relative velocity in the plane formed by relative velocity and the face normal
- **gust modulation**, which varies the base wind speed over time
- **procedural noise**, which adds small local flutter variation

This is a lightweight approximation inspired by the aerodynamic force model in Keckeisen et al. [4]. It does not implement a full Navier-Stokes or particle-tracing wind field.

---

## Demo Notes

### Fast Mass-Spring Baseline

The original solver from the forked repository is based on Liu et al. [1]. It is preserved for comparison with the PBD solver.

![curtain_hang](https://user-images.githubusercontent.com/24758349/79005907-97ad1100-7b60-11ea-9e27-90375461beaf.gif)  
![curtain_ball](https://user-images.githubusercontent.com/24758349/79005924-9d0a5b80-7b60-11ea-8ce4-d9fc683441d7.gif)

### PBD Hang

Demonstrates pinned cloth behavior, constraint stiffness tuning, damping, and mouse interaction.

### PBD Drop

Demonstrates cloth collision against a smooth sphere obstacle.

### PBD Drop-Floor

Demonstrates plane collision, friction, floor contact, and self-collision behavior under dense folded contact.

### PBD Drop-Floor-Dual

Demonstrates interaction with multiple static obstacles, including a sphere and an analytic box/cube.

### PBD Hang-Wind

Demonstrates flag-like cloth motion under aerodynamic drag, lift, gust, and procedural noise.

---

## Project Status

### Setup / Integration

- [x] Build and understand existing framework
- [x] Add `PBDSolver` to project / CMake
- [x] Hook PBDSolver into app with solver mode switch
- [x] Preserve original mass-spring solver for comparison

### Core Solver Foundation

- [x] Implement `PBDSolver` skeleton
- [x] Initialize particle state (`x`, `p`, `v`, `invMass`)
- [x] Implement external force / gravity update
- [x] Implement damping
- [x] Implement position prediction / velocity reconstruction

### Constraint System

- [x] Implement fixed-point constraints
- [x] Implement structural distance constraints
- [x] Implement shear constraints
- [x] Implement dihedral bend constraints
- [x] Add stiffness / iteration-corrected stiffness handling
- [x] Add runtime parameter controls for PBD hang demo

### Collision Handling

- [x] Implement plane collision
- [x] Implement sphere collision
- [x] Implement analytic box collision
- [x] Add explicit collision constraint generation stage
- [x] Add self-collision detection and response
- [x] Add self-collision debug diagnostics
- [x] Tune collision robustness / epsilon

### Rendering / Output

- [x] Write updated positions to render buffer
- [x] Recompute normals
- [x] Verify shading / mesh integrity after simulation
- [x] Add visual support for floor, sphere, cube, and simple shadows

### Testing / Validation

- [x] Validate single-particle / two-particle cases
- [x] Validate hanging cloth behavior
- [x] Validate cloth drop / draping demo
- [x] Test timestep / iteration / resolution stability
- [x] Test floor-contact and self-collision diagnostics
- [ ] Compare quantitatively against original solver

### Advanced / Stretch Goals

- [x] Add wind / extra aerodynamic forces
- [x] Add runtime parameter controls
- [x] Add dual-obstacle collision demo
- [ ] Implement full mesh collision
- [ ] Implement full edge-edge cloth self-collision
- [ ] Benchmark performance
- [ ] Record comparison demos / metrics

### Final Deliverables

- [x] Stable PBD cloth demos
- [x] Updated README / documentation
- [ ] Demo video / GIF
- [ ] Final report / slides / citations

---

## Known Limitations and Future Work

- Self-collision is discrete, not full continuous collision detection.
- Self-collision currently uses vertex-triangle constraints; full edge-edge self-collision is not implemented.
- Box collision is analytic/proxy-based, not full mesh collision.
- Edge-midpoint sampling improves obstacle collision near sharp edges but is still an approximation.
- Wind is modeled as an external aerodynamic force, not a full Navier-Stokes or particle-tracing wind field.
- Quantitative performance benchmarking is not yet complete.
- Demo videos/GIFs are future deliverables.

---

## Credits and License

This project is based on the original **FastMassSpring** repository by Samer Itani, licensed under the MIT License.

Modifications and extensions, including the PBD solver, are developed as part of the Stony Brook University CSE 328 final project.

---

## References

[1] Liu, T., Bargteil, A. W., O'Brien, J. F., & Kavan, L. (2013). Fast simulation of mass-spring systems. *ACM Transactions on Graphics, 32*(6), 1–7. https://doi.org/10.1145/2508363.2508406

[2] Provot, X. (1995). Deformation constraints in a mass-spring model to describe rigid cloth behavior. In *Graphics Interface* 1995, 147–154.

[3] Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007). *Position Based Dynamics.* In C. Mendoza & I. Navazo (Eds.), Proceedings of the 3rd Workshop in Virtual Reality Interactions and Physical Simulation (VRIPHYS 2006).

[4] Keckeisen, M., Kimmerle, S., Thomaszewski, B., & Wacker, M. (2004). *Modelling Effects of Wind Fields in Cloth Animations.* Journal of WSCG, 12(1–3).