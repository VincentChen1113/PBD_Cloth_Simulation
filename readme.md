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

This section will demonstrate the new PBD-based cloth simulation system implemented for this project.

### Demo

*(To be added)*

Examples you can include later:
- Cloth dropping onto a sphere  
- Cloth draping over different objects  
- Comparison between low/high resolution cloth  
- Side-by-side comparison with mass-spring solver  

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

On Windows, you will likely need to specify the directories containing GLUT and GLEW in CMAKE_PREFIX_PATH so that cmake can find them.

``` bash
cmake .. -DCMAKE_PERFIX_PATH:PATH=/path/to/libs
```

You will also need to copy the DLLs to the build directory if they are not available globally.

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
