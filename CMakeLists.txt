cmake_minimum_required(VERSION 3.10.0)

if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

project(fast-mass-spring)

set(OpenGL_GL_PREFERENCE GLVND)

set(Sources
    ClothApp/app.cpp
    ClothApp/MassSpringSolver.cpp
  ClothApp/PBDSolver.cpp
    ClothApp/Mesh.cpp
    ClothApp/Renderer.cpp
    ClothApp/Shader.cpp
    ClothApp/UserInteraction.cpp
)

# find OpenGL, GLUT, GLEW
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

# add Eigen, OpenMesh, glm
include(FetchContent)
FetchContent_Declare(
  openmesh
  GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git 
)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
  GIT_TAG         3.4.0
)

FetchContent_Declare(
  glm
  GIT_REPOSITORY  https://github.com/g-truc/glm
)

FetchContent_GetProperties(openmesh)
if(NOT openmesh_POPULATED)
  FetchContent_Populate(openmesh)
  add_subdirectory(${openmesh_SOURCE_DIR} ${openmesh_BINARY_DIR})
endif()

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})


FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
  add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
endif()

# needed for OpenMesh on Windows
if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

# create executable
add_executable(fast-mass-spring ${Sources})
target_link_libraries(fast-mass-spring ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} OpenMeshCore eigen glm)

# copy shaders to binary directory on every build.
add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/ClothApp/shaders
          ${CMAKE_CURRENT_BINARY_DIR}/shaders
)

add_dependencies(fast-mass-spring copy_shaders)
