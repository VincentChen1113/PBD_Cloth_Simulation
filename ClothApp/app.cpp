#include <GL/glew.h>
#include <GL/glut.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

#include "Shader.h"
#include "Mesh.h"
#include "Renderer.h"
#include "MassSpringSolver.h"
#include "PBDSolver.h"
#include "UserInteraction.h"

// G L O B A L S ///////////////////////////////////////////////////////////////////

// Window
static int g_windowWidth = 1280, g_windowHeight = 1280;
static bool g_mouseClickDown = false;
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX;
static int g_mouseClickY;

// User Interaction
static UserInteraction* UI;
static Renderer* g_pickRenderer;
static ProgramInput* g_floor_target;
static ProgramInput* g_sphere_target;
static unsigned int g_sphere_index_count = 0u;
static ProgramInput* g_cube_target;
static unsigned int g_cube_index_count = 0u;

// Constants
static const float PI = glm::pi<float>();
static const glm::vec3 g_floor_albedo(0.55f, 0.55f, 0.58f);
static const glm::vec3 g_floor_ambient(0.04f, 0.04f, 0.04f);
static const glm::vec3 g_sphere_albedo(0.36f, 0.38f, 0.40f);
static const glm::vec3 g_sphere_ambient(0.05f, 0.05f, 0.05f);
static const glm::vec3 g_cube_albedo(0.36f, 0.38f, 0.40f);
static const glm::vec3 g_cube_ambient(0.05f, 0.05f, 0.05f);
static const float g_specular_strength = 0.28f;
static const float g_shininess = 32.0f;
static const glm::vec4 g_shadow_color(0.0f, 0.0f, 0.0f, 0.32f);
static const float g_floor_collision_height = -1.75f;
static const float g_floor_render_offset = -0.002f;
static const float g_floor_extent = 3.5f;

// Shader Handles
static PhongShader* g_phongShader; // linked phong shader
static ShadowShader* g_shadowShader; // linked shadow shader
static PickShader* g_pickShader; // linked pick shader

// Shader parameters
static const glm::vec3 g_albedo(0.0f, 0.3f, 0.7f);
static const glm::vec3 g_ambient(0.01f, 0.01f, 0.01f);
static const glm::vec3 g_light(1.0f, 1.0f, -1.0f);

// Mesh
static Mesh* g_clothMesh; // halfedge data structure

// Render Target
static ProgramInput* g_render_target; // vertex, normal, texutre, index

// Animation
static const int g_fps = 60; // frames per second  | 60
static const int g_iter = 5; // iterations per time step | 10
static const int g_frame_time = 15; // approximate time for frame calculations | 15
static const int g_animation_timer = (int) ((1.0f / g_fps) * 1000 - g_frame_time);

// Mass Spring System
static mass_spring_system* g_system;
static MassSpringSolver* g_solver;

// PBD System
static pbd_system* g_pbdSystem;
static PBDSolver* g_pbdSolver;
static float g_selfCollisionThicknessOverride = -1.0f;
static unsigned int g_pbdFrameCounter = 0u;
static bool g_enableDebugDiagnostics = false;

enum class WindCliMode {
	None,
	Speed
};

static WindCliMode g_windCliMode = WindCliMode::None;
static float g_windSpeed = 0.0f;
static bool g_windDirectionCliProvided = false;
static Eigen::Vector3f g_windDirection(1.0f, 0.0f, 0.0f);
static unsigned int g_pbdHangIterationsOverride = 0u;
static unsigned int g_pbdDropIterationsOverride = 0u;
static unsigned int g_pbdFloorIterationsOverride = 0u;
static unsigned int g_pbdDualIterationsOverride = 0u;
static float g_pbdFloorTimestepOverride = -1.0f;

// Constraint Graph
static CgRootNode* g_cgRootNode;

// Scene parameters
static const float g_camera_distance = 4.2f;
static const float g_flag_camera_distance = 7.2f;

// Scene matrices
static glm::mat4 g_ModelViewMatrix;
static glm::mat4 g_ProjectionMatrix;

// System parameters for fast-mass-spring
namespace SystemParam {
	static const int n = 33; // must be odd, n * n = n_vertices | 61
	static const float w = 2.0f; // width | 2.0f
	static const float h = 0.008f; // time step, smaller for better results | 0.008f = 0.016f/2
	static const float r = w / (n - 1) * 1.05f; // spring rest legnth
	static const float k = 1.0f; // spring stiffness | 1.0f;
	static const float m = 0.25f / (n * n); // point mass | 0.25f
	static const float a = 0.993f; // damping, close to 1.0 | 0.993f
	static const float g = 9.8f * m; // gravitational force | 9.8f
}

// System parameters for PBD
namespace PBDSystemParam {
	static const int n = 33; // must be odd, n * n = n_vertices
	static const float w = 2.0f; // cloth width
	static const float h = 0.008f; // time step
	static const float r = w / (n - 1); // rest length
	static const float m = 0.25f / (n * n); // point mass
	static const float g = 9.8f; // gravitational acceleration

	static const int n_iter = 20; // solver iterations | 15
	static const float a = 0.02f; // damping factor
	static const float eps = 1e-4f; // collision epsilon
	static const float k_stretch = 0.9f; // stretch stiffness | 0.9f
	static const float k_shear = 0.8f; // shear stiffness | 0.8f
	static const float k_bend = 0.01f; // bend stiffness | 0.01f
	static const float sphere_radius = 0.64f;  // radius of sphere collider in drop demo | 0.64f
}

namespace PBDFloorDemoParam {
	static const float h = 0.003f; // Smaller time step for floor-contact stability.
	static const int n_iter = 28; // More solver iterations to resolve floor/self contacts robustly.
	static const float selfCollisionStiffness = 0.45f; // Extra stiffness for self-collision corrections in floor demos.
	static const unsigned int maxSelfCollisionContactsPerVertex = 4u; // Cap on generated self-collision contacts per vertex.
}

namespace PBDDebugParam {
	static const unsigned int debugPrintPeriod = 20u;
}

namespace PBDHangCliParam {
	static const unsigned int minIterations = 1u;
	static const unsigned int maxIterations = 80u;
}

namespace PBDHangControlParam {
	static const float stretchMin = 0.0f;
	static const float stretchMax = 1.0f;
	static const float shearMin = 0.0f;
	static const float shearMax = 1.0f;
	static const float bendMin = 0.0f;
	static const float bendMax = 0.10f;
	static const float dampingMin = 0.0f;
	static const float dampingMax = 0.20f;
	static const float stretchStep = 0.05f;
	static const float shearStep = 0.05f;
	static const float bendStep = 0.005f;
	static const float dampingStep = 0.01f;
	static const float defaultDamping = 0.07f;
}

namespace PBDDropCliParam {
	static const float minRadius = 0.1f;
	static const float maxRadius = 1.5f;
	static const unsigned int minIterations = 1u;
	static const unsigned int maxIterations = 80u;
}

namespace PBDDropControlParam {
	static const float stretchMin = 0.0f;
	static const float stretchMax = 1.0f;
	static const float shearMin = 0.0f;
	static const float shearMax = 1.0f;
	static const float bendMin = 0.0f;
	static const float bendMax = 0.10f;
	static const float dampingMin = 0.0f;
	static const float dampingMax = 0.20f;
	static const float stretchStep = 0.05f;
	static const float shearStep = 0.05f;
	static const float bendStep = 0.005f;
	static const float dampingStep = 0.01f;
	static const float defaultDamping = 0.07f;
	// Keep the drop demo on odd grid resolutions because the cloth builder and
	// spring layout are authored for odd n values. Adjusting by 2 preserves that.
	static const unsigned int minMeshResolution = 17u;
	static const unsigned int maxMeshResolution = 65u;
	static const unsigned int meshResolutionStep = 2u;
	static const unsigned int defaultMeshResolution = 33u;
}

namespace PBDFloorCliParam {
	static const unsigned int minIterations = 1u;
	static const unsigned int maxIterations = 80u;
	static const float minTimestep = 0.001f;
	static const float maxTimestep = 0.01f;
}

namespace PBDFloorControlParam {
	static const float selfCollisionStiffnessMin = 0.0f;
	static const float selfCollisionStiffnessMax = 1.0f;
	static const float selfCollisionStiffnessStep = 0.05f;
	static const unsigned int minSelfCollisionContacts = 1u;
	static const unsigned int maxSelfCollisionContacts = 12u;
	static const float selfCollisionThicknessMin = 0.001f;
	static const float selfCollisionThicknessMax = 0.03f;
	static const float selfCollisionThicknessStep = 0.001f;
	static const float floorFrictionMin = 0.0f;
	static const float floorFrictionMax = 0.8f;
	static const float floorFrictionStep = 0.05f;
	static const float bendMin = 0.0f;
	static const float bendMax = 0.05f;
	static const float bendStep = 0.0025f;
	static const float dampingMin = 0.0f;
	static const float dampingMax = 0.20f;
	static const float dampingStep = 0.01f;
	static const std::array<unsigned int, 6> meshChoices = { 17u, 25u, 33u, 41u, 55u, 65u };
	static const unsigned int defaultMeshResolution = 33u;
	static const std::array<float, 3> speedPresets = { 0.003f, 0.0045f, 0.006f };
	static const std::array<const char*, 3> speedPresetLabels = { "Stable", "Balanced", "Fast" };
};

namespace PBDDualCliParam {
	static const unsigned int minIterations = 1u;
	static const unsigned int maxIterations = 80u;
}

namespace PBDDualControlParam {
	static const float stretchMin = 0.0f;
	static const float stretchMax = 1.0f;
	static const float shearMin = 0.0f;
	static const float shearMax = 1.0f;
	static const float bendMin = 0.0f;
	static const float bendMax = 0.10f;
	static const float stretchStep = 0.05f;
	static const float shearStep = 0.05f;
	static const float bendStep = 0.005f;
	static const std::array<unsigned int, 6> meshChoices = { 17u, 25u, 33u, 41u, 55u, 65u };
	static const unsigned int defaultMeshResolution = 33u;
	static const float defaultSphereRadius = 0.30f;
	static const float defaultCubeSize = 0.44f;
	static const float sphereRadiusMin = 0.15f;
	static const float sphereRadiusMax = 0.80f;
	static const float cubeSizeMin = 0.15f;
	static const float cubeSizeMax = 0.80f;
	static const float objectSizeStep = 0.05f;
}

namespace PBDWindCliParam {
	static const float minValue = 0.0f; // Safe lower bound for user-provided --wind-speed.
	static const float maxValue = 15.0f; // Safe upper bound for user-provided --wind-speed.
	static const glm::vec3 windDirection(1.0f, 0.0f, 0.0f); // wind direction
	static const float minDirectionNorm = 1e-4f;
	static const float gustFraction = 0.30f; // Gust amplitude scale, used in A_gust = gustFraction * userWindValue.
	static const float gustFrequency = 1.35f; // Gust frequency in u(t) = u_base + A_gust * sin(2 * pi * gustFrequency * t) + noise(t, x).
	static const float noiseFraction = 0.08f; // Small procedural flutter scale added to u(t) after the sinusoidal gust term.
	static const float dragCoefficient = 1.15f; // Drag coefficient C_D in F_drag = 0.5 * rho * C_D * A * |v_rel|^2 * exposure.
	static const float liftCoefficient = 0.35f; // Lift coefficient C_L in F_lift = 0.5 * rho * C_L * A * |v_rel|^2 * orientationTerm.
	static const float airDensity = 1.225f; // Air density rho used by the drag-only aerodynamic force.
	static const float dampingFactor = 0.06f; // Velocity damping for the wind demo to keep flutter stable.
	static const float bendStiffness = 0.008f; // Softer bending stiffness so the flag can ripple under gusts.
	static const float flagHeightOffset = 1.15f; // Vertical lift applied when rotating the cloth into the flag pose.
}

namespace PBDWindControlParam {
	static const float speedMin = 0.0f;
	static const float speedMax = 15.0f;
	static const float speedStep = 0.5f;
	static const float directionMin = -1.0f;
	static const float directionMax = 1.0f;
	static const float directionStep = 0.1f;
	static const float verticalWarningThreshold = 0.35f;
}

struct PBDHangRuntimeState {
	bool initialized = false;
	bool paused = false;
	float defaultStretch = PBDSystemParam::k_stretch;
	float defaultShear = PBDSystemParam::k_shear;
	float defaultBend = PBDSystemParam::k_bend;
	float defaultDamping = PBDHangControlParam::defaultDamping;
	std::vector<float> initialPositions;
};

static PBDHangRuntimeState g_pbdHangRuntime;

struct PBDWindRuntimeState {
	bool initialized = false;
	bool paused = false;
	float currentWindSpeed = 0.0f;
	float startupWindSpeed = 0.0f;
	Eigen::Vector3f currentDirectionInput = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
	Eigen::Vector3f appliedWindDirection = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
};

static PBDWindRuntimeState g_pbdWindRuntime;

struct PBDDropRuntimeState {
	bool initialized = false;
	bool paused = false;
	float defaultStretch = PBDSystemParam::k_stretch;
	float defaultShear = PBDSystemParam::k_shear;
	float defaultBend = PBDSystemParam::k_bend;
	float defaultDamping = PBDDropControlParam::defaultDamping;
	unsigned int currentMeshResolution = PBDDropControlParam::defaultMeshResolution;
	unsigned int pendingMeshResolution = PBDDropControlParam::defaultMeshResolution;
	float sphereRadius = PBDSystemParam::sphere_radius;
	unsigned int solverIterations = static_cast<unsigned int>(PBDSystemParam::n_iter);
};

static PBDDropRuntimeState g_pbdDropRuntime;
static float g_pbdDropStartupSphereRadius = PBDSystemParam::sphere_radius;

struct PBDFloorRuntimeState {
	bool initialized = false;
	bool paused = false;
	float defaultSelfCollisionStiffness = PBDFloorDemoParam::selfCollisionStiffness;
	unsigned int defaultMaxSelfCollisionContacts = PBDFloorDemoParam::maxSelfCollisionContactsPerVertex;
	float defaultSelfCollisionThickness = 0.0f;
	float defaultFloorFriction = 0.15f;
	float defaultBend = PBDSystemParam::k_bend;
	float defaultDamping = PBDSystemParam::a;
	unsigned int solverIterations = static_cast<unsigned int>(PBDFloorDemoParam::n_iter);
	float currentTimestep = PBDFloorDemoParam::h;
	int speedPresetIndex = 0;
	bool customTimestep = false;
	unsigned int currentMeshResolution = PBDFloorControlParam::defaultMeshResolution;
	unsigned int pendingMeshResolution = PBDFloorControlParam::defaultMeshResolution;
	bool debugEnabled = false;
};

static PBDFloorRuntimeState g_pbdFloorRuntime;

struct PBDDualRuntimeState {
	bool initialized = false;
	bool paused = false;
	float defaultStretch = PBDSystemParam::k_stretch;
	float defaultShear = PBDSystemParam::k_shear;
	float defaultBend = PBDSystemParam::k_bend;
	float currentTimestep = PBDFloorDemoParam::h;
	int speedPresetIndex = 0;
	bool customTimestep = false;
	unsigned int currentMeshResolution = PBDDualControlParam::defaultMeshResolution;
	unsigned int pendingMeshResolution = PBDDualControlParam::defaultMeshResolution;
	float currentSphereRadius = PBDDualControlParam::defaultSphereRadius;
	float pendingSphereRadius = PBDDualControlParam::defaultSphereRadius;
	float currentCubeSize = PBDDualControlParam::defaultCubeSize;
	float pendingCubeSize = PBDDualControlParam::defaultCubeSize;
	unsigned int solverIterations = static_cast<unsigned int>(PBDFloorDemoParam::n_iter);
};

static PBDDualRuntimeState g_pbdDualRuntime;

// F U N C T I O N S //////////////////////////////////////////////////////////////
// state initialization
static void initGlutState(int, char**);
static void initGLState();
static void parseSimMode(int, char**);
static void parseOptionalArgs(int argc, char** argv, int startIndex);
static void validateParsedOptions();

static void initShaders(); // Read, compile and link shaders
static void initCloth(); // Generate cloth mesh
static void initFloor(); // Generate floor mesh
static void initSphereColliderVisual(float radius, const glm::vec3& center); // Generate sphere collider mesh
static void initCubeColliderVisual(const glm::vec3& center, const glm::vec3& halfExtents); // Generate cube collider mesh
static void initScene(); // Generate scene matrices
static void initMouseInteraction(FixedPointController*, unsigned int);
static void rebuildClothMesh(unsigned int resolution);
static Eigen::Vector3f normalizedWindDirectionOrFallback(const Eigen::Vector3f& direction, const Eigen::Vector3f& fallback, bool* usedFallback = nullptr);
static PBDWindConfig makeWindConfig(float windSpeed, const Eigen::Vector3f& direction);
static void configurePBDWindSolver(float windSpeed, const Eigen::Vector3f& directionInput, bool captureDefaults);
static void applyPBDWindSettings(float windSpeed, const Eigen::Vector3f& directionInput);
static void resetPBDWindDemo(bool resetParameters);
static void logPBDWindControlState(const std::string& reason);
static void logPBDHangControlState(const std::string& reason);
static unsigned int previousPBDDropResolution(unsigned int resolution);
static unsigned int nextPBDDropResolution(unsigned int resolution);
static void configurePBDDropSolver(float stretch, float shear, float bend, float damping, unsigned int iterations, float sphereRadius, unsigned int meshResolution, bool captureDefaults);
static void resetPBDDropDemo(bool resetParameters);
static void logPBDDropControlState(const std::string& reason);
static unsigned int previousPBDFloorResolution(unsigned int resolution);
static unsigned int nextPBDFloorResolution(unsigned int resolution);
static int closestPBDFloorSpeedPreset(float dt);
static void setPBDFloorRuntimeTimestep(float dt, int presetIndex, bool custom);
static void configurePBDFloorSolver(
	float selfCollisionStiffness,
	unsigned int maxContactsPerVertex,
	float selfCollisionThickness,
	float floorFriction,
	float bend,
	float damping,
	unsigned int iterations,
	float timestep,
	unsigned int meshResolution,
	bool captureDefaults
);
static void resetPBDFloorDemo(bool resetParameters);
static void logPBDFloorControlState(const std::string& reason);
static unsigned int previousPBDDualResolution(unsigned int resolution);
static unsigned int nextPBDDualResolution(unsigned int resolution);
static void setPBDDualRuntimeTimestep(float dt, int presetIndex, bool custom);
static void configurePBDDualSolver(
	float stretch,
	float shear,
	float bend,
	unsigned int iterations,
	float timestep,
	unsigned int meshResolution,
	float sphereRadius,
	float cubeSize,
	bool captureDefaults
);
static void resetPBDDualDemo(bool resetParameters);
static void logPBDDualControlState(const std::string& reason);
static void drawBitmapText(float x, float y, const std::string& text);
static void drawWindDirectionIndicator(const Eigen::Vector3f& normalizedDirection);
static void drawPBDHangOverlay();
static void drawPBDWindOverlay();
static void drawPBDDropOverlay();
static void drawPBDFloorOverlay();
static void drawPBDDualOverlay();
static glm::mat4 floorShadowMatrix(float planeHeight, const glm::vec3& lightDirection);
static void orientClothForFloorDrop();
static void orientClothFlatForDualFloorDrop(float sphereRadius, float cubeSize);
static void orientClothForWindFlag();
static void logPBDSelfCollisionDiagnostics();
static bool isPBDMode();
static bool hasSphereColliderVisual();
static bool hasCubeColliderVisual();
static unsigned int activeGridSize();
static float activeClothWidth();
static pbd_system* buildPBDSystem(const mass_spring_system& system);

struct AnalyticBoxDefinition {
	glm::vec3 center;
	glm::vec3 halfExtents;
};

// demos
namespace PBDDualObstacleDemoParam {
	static const glm::vec3 sphereCenter(0.42f, 0.0f, g_floor_collision_height + 0.30f);
	static const float sphereRadius = PBDDualControlParam::defaultSphereRadius;
	static const AnalyticBoxDefinition cube = {
		glm::vec3(-0.42f, 0.0f, g_floor_collision_height + 0.44f),
		glm::vec3(PBDDualControlParam::defaultCubeSize, PBDDualControlParam::defaultCubeSize, PBDDualControlParam::defaultCubeSize)
	};
}

enum class SimMode {
	MassSpringHang,
	MassSpringDrop,
	PBDHang,
	PBDHangWind,
	PBDDrop,
	PBDDropFloor,
	PBDDropFloorDual
};

static SimMode g_mode = SimMode::MassSpringHang; // default to mass-spring hanging demo, switch to other demos later
// demos
static void demo_hang();
static void demo_drop();
static void demo_pbd_hang();
static void demo_pbd_hang_wind();
static void demo_pbd_drop();
static void demo_pbd_drop_floor();
static void demo_pbd_drop_floor_dual();
static void(*g_demo)() = demo_hang;

static bool isFloorDemo() {
	return g_mode == SimMode::PBDDropFloor || g_mode == SimMode::PBDDropFloorDual;
}

static bool hasSceneFloor() {
	return isFloorDemo() || g_mode == SimMode::PBDHangWind;
}

static bool hasSphereColliderVisual() {
	return g_mode == SimMode::MassSpringDrop || g_mode == SimMode::PBDDrop || g_mode == SimMode::PBDDropFloorDual;
}

static bool hasCubeColliderVisual() {
	return g_mode == SimMode::PBDDropFloorDual;
}

static void selectDemo() {
	switch (g_mode) {
	case SimMode::PBDDropFloorDual:
		g_demo = demo_pbd_drop_floor_dual;
		break;
	case SimMode::MassSpringHang:
		g_demo = demo_hang;
		break;
	case SimMode::MassSpringDrop:
		g_demo = demo_drop;
		break;
	case SimMode::PBDHang:
		g_demo = demo_pbd_hang;
		break;
	case SimMode::PBDHangWind:
		g_demo = demo_pbd_hang_wind;
		break;
	case SimMode::PBDDrop:
		g_demo = demo_pbd_drop;
		break;
	case SimMode::PBDDropFloor:
		g_demo = demo_pbd_drop_floor;
		break;
	}
}

// glut callbacks
static void display();
static void reshape(int, int);
static void keyboard(unsigned char, int, int);
static void mouse(int, int, int, int);
static void motion(int, int);

// draw cloth function
static void drawFloor();
static void drawFloorShadows();
static void drawCloth();
static void animateCloth(int value);

// scene update
static void updateProjection();
static void updateRenderTarget();

// cleaning
static void cleanUp();

// error checks
void checkGlErrors();



// M A I N //////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	try {
		parseSimMode(argc, argv);
		validateParsedOptions();
		initGlutState(argc, argv);
		glewInit();
		initGLState();

		selectDemo();
		initShaders();
		initCloth();
		initFloor();
		initScene();

		glutTimerFunc(g_animation_timer, animateCloth, 0);
		glutMainLoop();

		cleanUp();
		return 0;
	}
	catch (const std::runtime_error& e) {
		std::cout << "Exception caught: " << e.what() << std::endl;
		return -1;
	}
}


// S T A T E  I N I T I A L I Z A T O N /////////////////////////////////////////////
static void parseSimMode(int argc, char** argv) {
	if (argc <= 1) return;

	const std::string arg1(argv[1]);
	if (arg1 == "mass-spring-hang" || arg1 == "ms-hang") {
		g_mode = SimMode::MassSpringHang;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "mass-spring-drop" || arg1 == "ms-drop") {
		g_mode = SimMode::MassSpringDrop;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "pbd-hang") {
		g_mode = SimMode::PBDHang;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "pbd-hang_wind" || arg1 == "pbd-hang-wind") {
		g_mode = SimMode::PBDHangWind;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "pbd-drop") {
		g_mode = SimMode::PBDDrop;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "pbd-drop-floor") {
		g_mode = SimMode::PBDDropFloor;
		parseOptionalArgs(argc, argv, 2);
		return;
	}
	if (arg1 == "pbd-drop-floor-dual") {
		g_mode = SimMode::PBDDropFloorDual;
		parseOptionalArgs(argc, argv, 2);
		return;
	}

	if (argc >= 3) {
		const std::string solver(argv[1]);
		const std::string scene(argv[2]);
		if ((solver == "mass-spring" || solver == "ms") && scene == "hang") {
			g_mode = SimMode::MassSpringHang;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if ((solver == "mass-spring" || solver == "ms") && scene == "drop") {
			g_mode = SimMode::MassSpringDrop;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if (solver == "pbd" && scene == "hang") {
			g_mode = SimMode::PBDHang;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if (solver == "pbd" && (scene == "hang_wind" || scene == "hang-wind")) {
			g_mode = SimMode::PBDHangWind;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if (solver == "pbd" && scene == "drop") {
			g_mode = SimMode::PBDDrop;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if (solver == "pbd" && scene == "drop-floor") {
			g_mode = SimMode::PBDDropFloor;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
		if (solver == "pbd" && scene == "drop-floor-dual") {
			g_mode = SimMode::PBDDropFloorDual;
			parseOptionalArgs(argc, argv, 3);
			return;
		}
	}

	throw std::runtime_error(
		"Usage: ./fast-mass-spring [mass-spring|ms] [hang|drop] [--self-thickness value] [--debug], ./fast-mass-spring pbd hang [--iters value] [--self-thickness value] [--debug], ./fast-mass-spring pbd hang-wind [--wind-speed value] [--wind-dir x y z] [--self-thickness value] [--debug], ./fast-mass-spring pbd drop [--radius value] [--iters value] [--self-thickness value] [--debug], ./fast-mass-spring pbd drop-floor [--iters value] [--dt value] [--self-thickness value] [--debug], ./fast-mass-spring pbd drop-floor-dual [--iters value] [--self-thickness value] [--debug], or ./fast-mass-spring [ms-hang|ms-drop|pbd-hang|pbd-hang-wind|pbd-drop|pbd-drop-floor|pbd-drop-floor-dual] [--self-thickness value] [--debug] [--wind-speed value] [--wind-dir x y z] [--iters value]"
	);
}

static void parseOptionalArgs(int argc, char** argv, int startIndex) {
	for (int i = startIndex; i < argc; ++i) {
		const std::string arg(argv[i]);
		if (arg == "--debug") {
			g_enableDebugDiagnostics = true;
			continue;
		}

		if (arg == "--self-thickness") {
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --self-thickness");
			}

			std::stringstream valueStream(argv[++i]);
			float thickness = -1.0f;
			valueStream >> thickness;
			if (!valueStream || !valueStream.eof() || thickness <= 0.0f) {
				throw std::runtime_error("--self-thickness expects a positive float value");
			}

			g_selfCollisionThicknessOverride = thickness;
			continue;
		}

		if (arg == "--wind-speed") {
			if (g_mode != SimMode::PBDHangWind) {
				throw std::runtime_error("--wind-speed is only valid for the pbd hang-wind demo");
			}
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --wind-speed");
			}
			if (g_windCliMode != WindCliMode::None) {
				throw std::runtime_error("Specify --wind-speed only once");
			}

			std::stringstream valueStream(argv[++i]);
			float windSpeed = 0.0f;
			valueStream >> windSpeed;
			if (!valueStream || !valueStream.eof()
				|| windSpeed < PBDWindCliParam::minValue
				|| windSpeed > PBDWindCliParam::maxValue) {
				throw std::runtime_error("--wind-speed expects a float value in [0, 15]");
			}

			g_windCliMode = WindCliMode::Speed;
			g_windSpeed = windSpeed;
			continue;
		}

		if (arg == "--wind-dir") {
			if (g_mode != SimMode::PBDHangWind) {
				throw std::runtime_error("--wind-dir is only valid for the pbd hang-wind demo");
			}
			if (i + 3 >= argc) {
				throw std::runtime_error("--wind-dir expects three float values: x y z");
			}
			if (g_windDirectionCliProvided) {
				throw std::runtime_error("Specify --wind-dir only once");
			}

			std::stringstream xStream(argv[++i]);
			std::stringstream yStream(argv[++i]);
			std::stringstream zStream(argv[++i]);
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;
			xStream >> x;
			yStream >> y;
			zStream >> z;
			if (!xStream || !xStream.eof() || !yStream || !yStream.eof() || !zStream || !zStream.eof()) {
				throw std::runtime_error("--wind-dir expects three float values: x y z");
			}

			const Eigen::Vector3f direction(x, y, z);
			if (direction.norm() < PBDWindCliParam::minDirectionNorm) {
				throw std::runtime_error("--wind-dir expects a non-zero vector");
			}

			g_windDirection = direction;
			g_windDirectionCliProvided = true;
			continue;
		}

		if (arg == "--iters") {
			if (g_mode != SimMode::PBDHang && g_mode != SimMode::PBDDrop && g_mode != SimMode::PBDDropFloor && g_mode != SimMode::PBDDropFloorDual) {
				throw std::runtime_error("--iters is only valid for the pbd hang, pbd drop, pbd drop-floor, or pbd drop-floor-dual demo");
			}
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --iters");
			}
			if ((g_mode == SimMode::PBDHang && g_pbdHangIterationsOverride != 0u)
				|| (g_mode == SimMode::PBDDrop && g_pbdDropIterationsOverride != 0u)
				|| (g_mode == SimMode::PBDDropFloor && g_pbdFloorIterationsOverride != 0u)
				|| (g_mode == SimMode::PBDDropFloorDual && g_pbdDualIterationsOverride != 0u)) {
				throw std::runtime_error("Specify --iters only once");
			}

			std::stringstream valueStream(argv[++i]);
			int iterations = 0;
			valueStream >> iterations;
			// --iters controls the number of PBD projection passes per timestep,
			// so keep it within a safe startup range for the interactive demos.
			if (!valueStream || !valueStream.eof()
				|| iterations < static_cast<int>(PBDHangCliParam::minIterations)
				|| iterations > static_cast<int>(PBDHangCliParam::maxIterations)) {
				throw std::runtime_error("--iters expects an integer value in [1, 80]");
			}

			if (g_mode == SimMode::PBDHang) {
				g_pbdHangIterationsOverride = static_cast<unsigned int>(iterations);
			}
			else if (g_mode == SimMode::PBDDrop) {
				g_pbdDropIterationsOverride = static_cast<unsigned int>(iterations);
			}
			else if (g_mode == SimMode::PBDDropFloor) {
				g_pbdFloorIterationsOverride = static_cast<unsigned int>(iterations);
			}
			else {
				g_pbdDualIterationsOverride = static_cast<unsigned int>(iterations);
			}
			continue;
		}

		if (arg == "--dt") {
			if (g_mode != SimMode::PBDDropFloor) {
				throw std::runtime_error("--dt is only valid for the pbd drop-floor demo");
			}
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --dt");
			}

			std::stringstream valueStream(argv[++i]);
			float dt = 0.0f;
			valueStream >> dt;
			if (!valueStream || !valueStream.eof()
				|| dt < PBDFloorCliParam::minTimestep
				|| dt > PBDFloorCliParam::maxTimestep) {
				throw std::runtime_error("--dt expects a float value in [0.001, 0.01]");
			}

			g_pbdFloorTimestepOverride = dt;
			continue;
		}

		if (arg == "--radius") {
			if (g_mode != SimMode::PBDDrop) {
				throw std::runtime_error("--radius is only valid for the pbd drop demo");
			}
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --radius");
			}

			std::stringstream valueStream(argv[++i]);
			float radius = 0.0f;
			valueStream >> radius;
			if (!valueStream || !valueStream.eof()
				|| radius < PBDDropCliParam::minRadius
				|| radius > PBDDropCliParam::maxRadius) {
				throw std::runtime_error("--radius expects a float value in [0.1, 1.5]");
			}

			g_pbdDropStartupSphereRadius = radius;
			continue;
		}

		throw std::runtime_error("Unknown argument: " + arg);
	}
}

static void validateParsedOptions() {
	if (g_mode == SimMode::PBDHangWind && g_windCliMode == WindCliMode::None) {
		throw std::runtime_error("The pbd hang-wind demo requires --wind-speed with a value in [0, 15]");
	}
}

static void initGlutState(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(g_windowWidth, g_windowHeight);
	glutCreateWindow("Cloth App");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
}

static void initGLState() {
	glClearColor(0.25f, 0.25f, 0.25f, 0);
	glClearDepth(1.);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glReadBuffer(GL_BACK);
	glEnable(GL_FRAMEBUFFER_SRGB);

	checkGlErrors();
}

static bool isPBDMode() {
	return g_mode == SimMode::PBDHang
		|| g_mode == SimMode::PBDHangWind
		|| g_mode == SimMode::PBDDrop
		|| g_mode == SimMode::PBDDropFloor
		|| g_mode == SimMode::PBDDropFloorDual;
}

static unsigned int activeGridSize() {
	return isPBDMode() ? PBDSystemParam::n : SystemParam::n;
}

static float activeClothWidth() {
	return isPBDMode() ? PBDSystemParam::w : SystemParam::w;
}

static void rebuildClothMesh(unsigned int resolution) {
	delete g_clothMesh;
	g_clothMesh = nullptr;
	delete g_render_target;
	g_render_target = nullptr;

	MeshBuilder meshBuilder;
	meshBuilder.uniformGrid(PBDSystemParam::w, static_cast<int>(resolution));
	g_clothMesh = meshBuilder.getResult();

	g_render_target = new ProgramInput;
	g_render_target->setPositionData(g_clothMesh->vbuff(), g_clothMesh->vbuffLen());
	g_render_target->setNormalData(g_clothMesh->nbuff(), g_clothMesh->nbuffLen());
	g_render_target->setTextureData(g_clothMesh->tbuff(), g_clothMesh->tbuffLen());
	g_render_target->setIndexData(g_clothMesh->ibuff(), g_clothMesh->ibuffLen());
}

static void initShaders() {
	GLShader basic_vert(GL_VERTEX_SHADER);
	GLShader phong_frag(GL_FRAGMENT_SHADER);
	GLShader shadow_frag(GL_FRAGMENT_SHADER);
	GLShader pick_frag(GL_FRAGMENT_SHADER);

	auto ibasic = std::ifstream("./shaders/basic.vshader");
	auto iphong = std::ifstream("./shaders/phong.fshader");
	auto ishadow = std::ifstream("./shaders/shadow.fshader");
	auto ifrag = std::ifstream("./shaders/pick.fshader");

	basic_vert.compile(ibasic);
	phong_frag.compile(iphong);
	shadow_frag.compile(ishadow);
	pick_frag.compile(ifrag);

	g_phongShader = new PhongShader;
	g_shadowShader = new ShadowShader;
	g_pickShader = new PickShader;
	g_phongShader->link(basic_vert, phong_frag);
	g_shadowShader->link(basic_vert, shadow_frag);
	g_pickShader->link(basic_vert, pick_frag);

	checkGlErrors();
}

static void initCloth() {
	const unsigned int n = activeGridSize();
	const float w = activeClothWidth();

	// generate mesh
	MeshBuilder meshBuilder;
	meshBuilder.uniformGrid(w, n);			// generate uniform grid mesh with width w and n vertices per side
	g_clothMesh = meshBuilder.getResult();	// halfedge data structure

	// fill program input
	g_render_target = new ProgramInput;		// vertex, normal, texutre, index check Shader.h for details
	g_render_target->setPositionData(g_clothMesh->vbuff(), g_clothMesh->vbuffLen());
	g_render_target->setNormalData(g_clothMesh->nbuff(), g_clothMesh->nbuffLen());
	g_render_target->setTextureData(g_clothMesh->tbuff(), g_clothMesh->tbuffLen());
	g_render_target->setIndexData(g_clothMesh->ibuff(), g_clothMesh->ibuffLen());

	// check errors
	checkGlErrors();

	// build demo system
	g_demo();
}

static void initFloor() {
	const float floorRenderHeight = g_floor_collision_height + g_floor_render_offset;
	const float floorPositions[] = {
		-g_floor_extent, -g_floor_extent, floorRenderHeight,
		 g_floor_extent, -g_floor_extent, floorRenderHeight,
		 g_floor_extent,  g_floor_extent, floorRenderHeight,
		-g_floor_extent,  g_floor_extent, floorRenderHeight
	};
	const float floorNormals[] = {
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f
	};
	const float floorTexcoords[] = {
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};
	unsigned int floorIndices[] = {
		0, 1, 2,
		0, 2, 3
	};

	g_floor_target = new ProgramInput;
	g_floor_target->setPositionData(const_cast<float*>(floorPositions), 12);
	g_floor_target->setNormalData(const_cast<float*>(floorNormals), 12);
	g_floor_target->setTextureData(const_cast<float*>(floorTexcoords), 8);
	g_floor_target->setIndexData(floorIndices, 6);
}

static void initSphereColliderVisual(float radius, const glm::vec3& center) {
	const unsigned int stacks = 24u;
	const unsigned int slices = 48u;
	std::vector<float> positions;
	std::vector<float> normals;
	std::vector<float> texcoords;
	std::vector<unsigned int> indices;

	positions.reserve((stacks + 1u) * (slices + 1u) * 3u);
	normals.reserve((stacks + 1u) * (slices + 1u) * 3u);
	texcoords.reserve((stacks + 1u) * (slices + 1u) * 2u);
	indices.reserve(stacks * slices * 6u);

	for (unsigned int stack = 0; stack <= stacks; ++stack) {
		const float v = static_cast<float>(stack) / static_cast<float>(stacks);
		const float phi = PI * v;
		const float sinPhi = std::sin(phi);
		const float cosPhi = std::cos(phi);

		for (unsigned int slice = 0; slice <= slices; ++slice) {
			const float u = static_cast<float>(slice) / static_cast<float>(slices);
			const float theta = 2.0f * PI * u;
			const float sinTheta = std::sin(theta);
			const float cosTheta = std::cos(theta);
			const glm::vec3 normal(sinPhi * cosTheta, sinPhi * sinTheta, cosPhi);
			const glm::vec3 position = center + radius * normal;

			positions.push_back(position.x);
			positions.push_back(position.y);
			positions.push_back(position.z);
			normals.push_back(normal.x);
			normals.push_back(normal.y);
			normals.push_back(normal.z);
			texcoords.push_back(u);
			texcoords.push_back(v);
		}
	}

	for (unsigned int stack = 0; stack < stacks; ++stack) {
		for (unsigned int slice = 0; slice < slices; ++slice) {
			const unsigned int rowStart = stack * (slices + 1u);
			const unsigned int nextRowStart = (stack + 1u) * (slices + 1u);
			const unsigned int topLeft = rowStart + slice;
			const unsigned int topRight = topLeft + 1u;
			const unsigned int bottomLeft = nextRowStart + slice;
			const unsigned int bottomRight = bottomLeft + 1u;

			indices.push_back(topLeft);
			indices.push_back(bottomLeft);
			indices.push_back(topRight);
			indices.push_back(topRight);
			indices.push_back(bottomLeft);
			indices.push_back(bottomRight);
		}
	}

	delete g_sphere_target;
	g_sphere_target = new ProgramInput;
	g_sphere_target->setPositionData(positions.data(), static_cast<unsigned int>(positions.size()));
	g_sphere_target->setNormalData(normals.data(), static_cast<unsigned int>(normals.size()));
	g_sphere_target->setTextureData(texcoords.data(), static_cast<unsigned int>(texcoords.size()));
	g_sphere_target->setIndexData(indices.data(), static_cast<unsigned int>(indices.size()));
	g_sphere_index_count = static_cast<unsigned int>(indices.size());
}

static void initCubeColliderVisual(const glm::vec3& center, const glm::vec3& halfExtents) {
	const glm::vec3 corners[8] = {
		center + glm::vec3(-halfExtents.x, -halfExtents.y, -halfExtents.z),
		center + glm::vec3( halfExtents.x, -halfExtents.y, -halfExtents.z),
		center + glm::vec3( halfExtents.x,  halfExtents.y, -halfExtents.z),
		center + glm::vec3(-halfExtents.x,  halfExtents.y, -halfExtents.z),
		center + glm::vec3(-halfExtents.x, -halfExtents.y,  halfExtents.z),
		center + glm::vec3( halfExtents.x, -halfExtents.y,  halfExtents.z),
		center + glm::vec3( halfExtents.x,  halfExtents.y,  halfExtents.z),
		center + glm::vec3(-halfExtents.x,  halfExtents.y,  halfExtents.z)
	};
	const unsigned int faceCorners[6][4] = {
		{ 0, 3, 2, 1 },
		{ 4, 5, 6, 7 },
		{ 0, 4, 7, 3 },
		{ 1, 2, 6, 5 },
		{ 0, 1, 5, 4 },
		{ 3, 7, 6, 2 }
	};
	const glm::vec3 faceNormals[6] = {
		glm::vec3(0.0f, 0.0f, -1.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f)
	};
	const glm::vec2 faceTexcoords[4] = {
		glm::vec2(0.0f, 0.0f),
		glm::vec2(1.0f, 0.0f),
		glm::vec2(1.0f, 1.0f),
		glm::vec2(0.0f, 1.0f)
	};

	std::vector<float> positions;
	std::vector<float> normals;
	std::vector<float> texcoords;
	std::vector<unsigned int> indices;
	positions.reserve(24u * 3u);
	normals.reserve(24u * 3u);
	texcoords.reserve(24u * 2u);
	indices.reserve(36u);

	for (unsigned int face = 0; face < 6u; ++face) {
		const unsigned int baseIndex = static_cast<unsigned int>(positions.size() / 3u);
		for (unsigned int corner = 0; corner < 4u; ++corner) {
			const glm::vec3& position = corners[faceCorners[face][corner]];
			positions.push_back(position.x);
			positions.push_back(position.y);
			positions.push_back(position.z);
			normals.push_back(faceNormals[face].x);
			normals.push_back(faceNormals[face].y);
			normals.push_back(faceNormals[face].z);
			texcoords.push_back(faceTexcoords[corner].x);
			texcoords.push_back(faceTexcoords[corner].y);
		}

		indices.push_back(baseIndex + 0u);
		indices.push_back(baseIndex + 1u);
		indices.push_back(baseIndex + 2u);
		indices.push_back(baseIndex + 0u);
		indices.push_back(baseIndex + 2u);
		indices.push_back(baseIndex + 3u);
	}

	delete g_cube_target;
	g_cube_target = new ProgramInput;
	g_cube_target->setPositionData(positions.data(), static_cast<unsigned int>(positions.size()));
	g_cube_target->setNormalData(normals.data(), static_cast<unsigned int>(normals.size()));
	g_cube_target->setTextureData(texcoords.data(), static_cast<unsigned int>(texcoords.size()));
	g_cube_target->setIndexData(indices.data(), static_cast<unsigned int>(indices.size()));
	g_cube_index_count = static_cast<unsigned int>(indices.size());
}

static glm::mat4 floorShadowMatrix(float planeHeight, const glm::vec3& lightDirection) {
	const glm::vec4 plane(0.0f, 0.0f, 1.0f, -planeHeight);
	const glm::vec4 light(lightDirection.x, lightDirection.y, lightDirection.z, 0.0f);
	const float dot = glm::dot(plane, light);
	if (std::abs(dot) <= 1e-6f) {
		return glm::mat4(1.0f);
	}

	glm::mat4 shadow(0.0f);
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			shadow[col][row] = ((row == col) ? dot : 0.0f) - light[row] * plane[col];
		}
	}
	return shadow;
}

static void initScene() {
	const float cameraDistance = (g_mode == SimMode::PBDHangWind)
		? g_flag_camera_distance
		: g_camera_distance;
	const glm::vec3 focusPoint = (g_mode == SimMode::PBDHangWind)
		? glm::vec3(0.0f, 0.0f, PBDWindCliParam::flagHeightOffset + activeClothWidth() / 4.0f)
		: glm::vec3(0.0f, 0.0f, -1.0f);
	g_ModelViewMatrix = glm::lookAt(
		glm::vec3(0.618, -0.786, 0.3f) * cameraDistance,
		focusPoint,
		glm::vec3(0.0f, 0.0f, 1.0f)
	) * glm::translate(glm::mat4(1), glm::vec3(0.0f, 0.0f, activeClothWidth() / 4));
	updateProjection();
}

static void orientClothForFloorDrop() {
	if (g_clothMesh == nullptr || g_render_target == nullptr) return;

	const float width = activeClothWidth();
	const float halfWidth = 0.5f * width;
	const float lift = g_floor_collision_height + 0.5f * width + 0.35f;
	float* const positions = g_clothMesh->vbuff();
	const unsigned int vertexCount = g_clothMesh->n_vertices();

	for (unsigned int i = 0; i < vertexCount; ++i) {
		const float x = positions[3 * i + 0];
		const float y = positions[3 * i + 1];
		const float xNormalized = (halfWidth > 1e-6f) ? (x / halfWidth) : 0.0f;
		const float heightNormalized = (halfWidth > 1e-6f) ? (-y / halfWidth) : 0.0f;
		const float lateralOffset = 0.035f * xNormalized + 0.0125f * heightNormalized * heightNormalized;

		positions[3 * i + 0] = x;
		positions[3 * i + 1] = lateralOffset;
		positions[3 * i + 2] = -y + lift;
	}

	g_clothMesh->request_face_normals();
	g_clothMesh->update_normals();
	g_clothMesh->release_face_normals();
	updateRenderTarget();
}

static void orientClothFlatForDualFloorDrop(float sphereRadius, float cubeSize) {
	if (g_clothMesh == nullptr || g_render_target == nullptr) return;

	const float sphereTop = PBDDualObstacleDemoParam::sphereCenter.z + sphereRadius;
	const float cubeTop = PBDDualObstacleDemoParam::cube.center.z + cubeSize;
	const float targetHeight = std::max(sphereTop, cubeTop) + 0.8f;
	float* const positions = g_clothMesh->vbuff();
	const unsigned int vertexCount = g_clothMesh->n_vertices();

	for (unsigned int i = 0; i < vertexCount; ++i) {
		positions[3 * i + 2] = targetHeight;
	}

	g_clothMesh->request_face_normals();
	g_clothMesh->update_normals();
	g_clothMesh->release_face_normals();
	updateRenderTarget();
}

static void orientClothForWindFlag() {
	if (g_clothMesh == nullptr || g_render_target == nullptr) return;

	float* const positions = g_clothMesh->vbuff();
	const unsigned int vertexCount = g_clothMesh->n_vertices();
	for (unsigned int i = 0; i < vertexCount; ++i) {
		const float x = positions[3 * i + 0];
		const float y = positions[3 * i + 1];

		positions[3 * i + 0] = 0.0f;
		positions[3 * i + 1] = x;
		positions[3 * i + 2] = y + PBDWindCliParam::flagHeightOffset;
	}

	g_clothMesh->request_face_normals();
	g_clothMesh->update_normals();
	g_clothMesh->release_face_normals();
	updateRenderTarget();
}

static void initMouseInteraction(FixedPointController* mouseFixer, unsigned int n) {
	if (UI != nullptr) {
		UI->releasePoint();
		delete UI;
		UI = nullptr;
	}
	delete g_pickRenderer;
	g_pickRenderer = nullptr;

	g_pickRenderer = new Renderer();
	g_pickRenderer->setProgram(g_pickShader);
	g_pickRenderer->setProgramInput(g_render_target);
	g_pickRenderer->setElementCount(g_clothMesh->ibuffLen());
	g_pickShader->setTessFact(n);
	UI = new GridMeshUI(g_pickRenderer, mouseFixer, g_clothMesh->vbuff(), n);

}

static void configurePBDHangSolver(
	float stretch,
	float shear,
	float bend,
	float damping,
	unsigned int iterations,
	bool captureDefaults
) {
	const unsigned int n = PBDSystemParam::n;

	delete g_pbdSolver;
	g_pbdSolver = nullptr;
	delete g_pbdSystem;
	g_pbdSystem = nullptr;

	MassSpringBuilder builder;
	builder.uniformGrid(
		PBDSystemParam::n,
		PBDSystemParam::h,
		PBDSystemParam::r,
		1.0f,
		PBDSystemParam::m,
		PBDSystemParam::a,
		PBDSystemParam::g
	);

	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}

	// Stretch and shear are distance-constraint stiffness controls.
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), shear);
	// Bend controls dihedral bending stiffness.
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), bend);
	// Damping controls velocity energy decay without changing gravity.
	g_pbdSolver->setDampingFactor(damping);
	g_pbdSolver->setSolverIterations(iterations);
	g_pbdSolver->pinPoint(0);
	g_pbdSolver->pinPoint(n - 1);
	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, n);

	if (captureDefaults || !g_pbdHangRuntime.initialized) {
		g_pbdHangRuntime.defaultStretch = g_pbdSolver->getStructuralStiffness();
		g_pbdHangRuntime.defaultShear = g_pbdSolver->getShearStiffness();
		g_pbdHangRuntime.defaultBend = g_pbdSolver->getBendStiffness();
		g_pbdHangRuntime.defaultDamping = g_pbdSolver->getDampingFactor();
		g_pbdHangRuntime.initialPositions.assign(
			g_clothMesh->vbuff(),
			g_clothMesh->vbuff() + g_clothMesh->vbuffLen()
		);
		g_pbdHangRuntime.initialized = true;
	}
}

static void resetPBDHangDemo(bool resetParameters) {
	if (g_mode != SimMode::PBDHang || g_clothMesh == nullptr || !g_pbdHangRuntime.initialized) return;

	const float stretch = resetParameters
		? g_pbdHangRuntime.defaultStretch
		: g_pbdSolver->getStructuralStiffness();
	const float shear = resetParameters
		? g_pbdHangRuntime.defaultShear
		: g_pbdSolver->getShearStiffness();
	const float bend = resetParameters
		? g_pbdHangRuntime.defaultBend
		: g_pbdSolver->getBendStiffness();
	const float damping = resetParameters
		? g_pbdHangRuntime.defaultDamping
		: g_pbdSolver->getDampingFactor();
	const unsigned int iterations = (g_pbdSolver != nullptr)
		? g_pbdSolver->getSolverIterations()
		: ((g_pbdHangIterationsOverride > 0u) ? g_pbdHangIterationsOverride : static_cast<unsigned int>(PBDSystemParam::n_iter));

	if (UI != nullptr) {
		UI->releasePoint();
	}
	g_mouseClickDown = false;
	g_mouseLClickButton = false;
	g_mouseRClickButton = false;
	g_mouseMClickButton = false;

	std::copy(
		g_pbdHangRuntime.initialPositions.begin(),
		g_pbdHangRuntime.initialPositions.end(),
		g_clothMesh->vbuff()
	);
	g_clothMesh->request_face_normals();
	g_clothMesh->update_normals();
	g_clothMesh->release_face_normals();
	updateRenderTarget();

	configurePBDHangSolver(stretch, shear, bend, damping, iterations, false);
	logPBDHangControlState(resetParameters ? "reset-parameters" : "reset-cloth");
	glutPostRedisplay();
}

static void logPBDHangControlState(const std::string& reason) {
	if (g_mode != SimMode::PBDHang || g_pbdSolver == nullptr) return;
	std::cout
		<< "[pbd hang controls] " << reason
		<< " stretch=" << g_pbdSolver->getStructuralStiffness()
		<< ", shear=" << g_pbdSolver->getShearStiffness()
		<< ", bend=" << g_pbdSolver->getBendStiffness()
		<< ", damping=" << g_pbdSolver->getDampingFactor()
		<< ", iters=" << g_pbdSolver->getSolverIterations()
		<< (g_pbdHangRuntime.paused ? ", paused" : ", running")
		<< std::endl;
}

static Eigen::Vector3f normalizedWindDirectionOrFallback(const Eigen::Vector3f& direction, const Eigen::Vector3f& fallback, bool* usedFallback) {
	const float norm = direction.norm();
	if (norm < PBDWindCliParam::minDirectionNorm) {
		if (usedFallback != nullptr) {
			*usedFallback = true;
		}
		const float fallbackNorm = fallback.norm();
		if (fallbackNorm < PBDWindCliParam::minDirectionNorm) {
			return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		}
		return fallback / fallbackNorm;
	}
	if (usedFallback != nullptr) {
		*usedFallback = false;
	}
	return direction / norm;
}

static PBDWindConfig makeWindConfig(float windSpeed, const Eigen::Vector3f& direction) {
	PBDWindConfig windConfig;
	windConfig.inputMode = PBDWindInputMode::Speed;
	// Wind direction controls the direction of the applied aerodynamic force and
	// is normalized before it is given to the solver.
	windConfig.windDirection = normalizedWindDirectionOrFallback(direction, Eigen::Vector3f(1.0f, 0.0f, 0.0f), nullptr);
	// Wind speed controls the magnitude of the external wind velocity and is
	// clamped to [0, 15] for stability.
	windConfig.baseSpeed = std::max(PBDWindControlParam::speedMin, std::min(PBDWindControlParam::speedMax, windSpeed));
	windConfig.baseAcceleration = 0.0f;
	const float gustReference = windConfig.baseSpeed;
	// Drag, lift, gust, and noise remain internal defaults to keep the demo simple.
	windConfig.gustAmplitude = PBDWindCliParam::gustFraction * gustReference;
	windConfig.gustFrequency = PBDWindCliParam::gustFrequency;
	windConfig.noiseStrength = PBDWindCliParam::noiseFraction * gustReference;
	windConfig.dragCoefficient = PBDWindCliParam::dragCoefficient;
	windConfig.liftCoefficient = PBDWindCliParam::liftCoefficient;
	windConfig.airDensity = PBDWindCliParam::airDensity;
	windConfig.maxWindSpeed = PBDWindCliParam::maxValue;
	return windConfig;
}

static void applyPBDWindSettings(float windSpeed, const Eigen::Vector3f& directionInput) {
	if (g_pbdSolver == nullptr) return;

	const float clampedSpeed = std::max(PBDWindControlParam::speedMin, std::min(PBDWindControlParam::speedMax, windSpeed));
	const Eigen::Vector3f normalizedDirection = normalizedWindDirectionOrFallback(
		directionInput,
		g_pbdWindRuntime.appliedWindDirection,
		nullptr
	);
	g_pbdSolver->setWindConfig(makeWindConfig(clampedSpeed, directionInput));
	g_pbdWindRuntime.currentWindSpeed = clampedSpeed;
	g_pbdWindRuntime.currentDirectionInput = directionInput;
	g_pbdWindRuntime.appliedWindDirection = normalizedDirection;
}

static void configurePBDWindSolver(float windSpeed, const Eigen::Vector3f& directionInput, bool captureDefaults) {
	const unsigned int n = PBDSystemParam::n;
	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);

	if (UI != nullptr) {
		UI->releasePoint();
	}

	delete g_pbdSolver;
	g_pbdSolver = nullptr;
	delete g_pbdSystem;
	g_pbdSystem = nullptr;

	orientClothForWindFlag();

	MassSpringBuilder builder;
	builder.uniformGrid(
		PBDSystemParam::n,
		PBDSystemParam::h,
		PBDSystemParam::r,
		1.0f,
		PBDSystemParam::m,
		PBDSystemParam::a,
		PBDSystemParam::g
	);

	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}

	applyPBDWindSettings(windSpeed, directionInput);
	g_pbdSolver->setDampingFactor(PBDWindCliParam::dampingFactor);
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), PBDWindCliParam::bendStiffness);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);
	for (unsigned int row = 0; row < n; ++row) {
		g_pbdSolver->pinPoint(row * n);
	}

	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, n);

	if (captureDefaults || !g_pbdWindRuntime.initialized) {
		g_pbdWindRuntime.startupWindSpeed = g_pbdWindRuntime.currentWindSpeed;
		g_pbdWindRuntime.initialized = true;
	}
}

static void resetPBDWindDemo(bool resetParameters) {
	if (g_mode != SimMode::PBDHangWind || g_pbdSolver == nullptr || !g_pbdWindRuntime.initialized) return;

	if (resetParameters) {
		applyPBDWindSettings(
			g_pbdWindRuntime.startupWindSpeed,
			Eigen::Vector3f(PBDWindCliParam::windDirection.x, PBDWindCliParam::windDirection.y, PBDWindCliParam::windDirection.z)
		);
		logPBDWindControlState("reset-parameters");
		glutPostRedisplay();
		return;
	}

	g_mouseClickDown = false;
	g_mouseLClickButton = false;
	g_mouseRClickButton = false;
	g_mouseMClickButton = false;

	configurePBDWindSolver(g_pbdWindRuntime.currentWindSpeed, g_pbdWindRuntime.currentDirectionInput, false);
	logPBDWindControlState("reset-cloth");
	glutPostRedisplay();
}

static void logPBDWindControlState(const std::string& reason) {
	if (g_mode != SimMode::PBDHangWind || g_pbdSolver == nullptr) return;
	const Eigen::Vector3f& rawDirection = g_pbdWindRuntime.currentDirectionInput;
	const Eigen::Vector3f& normalizedDirection = g_pbdWindRuntime.appliedWindDirection;
	std::cout
		<< "[pbd wind controls] " << reason
		<< " speed=" << g_pbdWindRuntime.currentWindSpeed
		<< " range=[0, 15]"
		<< ", direction=(" << rawDirection.x() << ", " << rawDirection.y() << ", " << rawDirection.z() << ")"
		<< ", normalized=(" << normalizedDirection.x() << ", " << normalizedDirection.y() << ", " << normalizedDirection.z() << ")"
		<< (g_pbdWindRuntime.paused ? ", paused" : ", running")
		<< std::endl;
}

static unsigned int previousPBDDropResolution(unsigned int resolution) {
	if (resolution <= PBDDropControlParam::minMeshResolution) {
		return PBDDropControlParam::minMeshResolution;
	}
	return std::max(
		PBDDropControlParam::minMeshResolution,
		resolution - PBDDropControlParam::meshResolutionStep
	);
}

static unsigned int nextPBDDropResolution(unsigned int resolution) {
	if (resolution >= PBDDropControlParam::maxMeshResolution) {
		return PBDDropControlParam::maxMeshResolution;
	}
	return std::min(
		PBDDropControlParam::maxMeshResolution,
		resolution + PBDDropControlParam::meshResolutionStep
	);
}

static void configurePBDDropSolver(
	float stretch,
	float shear,
	float bend,
	float damping,
	unsigned int iterations,
	float sphereRadius,
	unsigned int meshResolution,
	bool captureDefaults
) {
	if (UI != nullptr) {
		UI->releasePoint();
	}

	delete g_pbdSolver;
	g_pbdSolver = nullptr;
	delete g_pbdSystem;
	g_pbdSystem = nullptr;

	// Mesh resolution changes particle and constraint count, so applying a new
	// resolution requires rebuilding the cloth mesh, render buffers, and PBD system.
	rebuildClothMesh(meshResolution);
	const float totalClothMass = PBDSystemParam::m * static_cast<float>(PBDSystemParam::n * PBDSystemParam::n);
	const float pointMass = totalClothMass / static_cast<float>(meshResolution * meshResolution);

	MassSpringBuilder builder;
	builder.uniformGrid(
		meshResolution,
		PBDSystemParam::h,
		PBDSystemParam::w / static_cast<float>(meshResolution - 1u),
		1.0f,
		pointMass,
		PBDSystemParam::a,
		PBDSystemParam::g
	);

	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}

	// Stretch and shear are distance-constraint stiffness controls.
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), shear);
	// Bend controls dihedral bending stiffness.
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), bend);
	// Damping controls velocity energy decay.
	g_pbdSolver->setDampingFactor(damping);
	// Iterations control the number of PBD projection passes per timestep.
	g_pbdSolver->setSolverIterations(iterations);

	const glm::vec3 sphereCenter(0.0f, 0.0f, -1.0f);
	// Sphere radius changes the collision equation C(p) = |p - c| - r >= 0.
	g_pbdSolver->addSphereCollider(Eigen::Vector3f(sphereCenter.x, sphereCenter.y, sphereCenter.z), sphereRadius);
	initSphereColliderVisual(sphereRadius, sphereCenter);

	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, meshResolution);

	g_pbdDropRuntime.currentMeshResolution = meshResolution;
	g_pbdDropRuntime.pendingMeshResolution = meshResolution;
	g_pbdDropRuntime.sphereRadius = sphereRadius;
	g_pbdDropRuntime.solverIterations = iterations;

	if (captureDefaults || !g_pbdDropRuntime.initialized) {
		g_pbdDropRuntime.defaultStretch = g_pbdSolver->getStructuralStiffness();
		g_pbdDropRuntime.defaultShear = g_pbdSolver->getShearStiffness();
		g_pbdDropRuntime.defaultBend = g_pbdSolver->getBendStiffness();
		g_pbdDropRuntime.defaultDamping = g_pbdSolver->getDampingFactor();
		g_pbdDropRuntime.initialized = true;
	}
}

static void resetPBDDropDemo(bool resetParameters) {
	if (g_mode != SimMode::PBDDrop || g_pbdSolver == nullptr || !g_pbdDropRuntime.initialized) return;

	if (resetParameters) {
		g_pbdSolver->setStructuralStiffness(g_pbdDropRuntime.defaultStretch);
		g_pbdSolver->setShearStiffness(g_pbdDropRuntime.defaultShear);
		g_pbdSolver->setBendStiffness(g_pbdDropRuntime.defaultBend);
		g_pbdSolver->setDampingFactor(g_pbdDropRuntime.defaultDamping);
		g_pbdDropRuntime.pendingMeshResolution = PBDDropControlParam::defaultMeshResolution;
		logPBDDropControlState("reset-parameters");
		glutPostRedisplay();
		return;
	}

	const float stretch = g_pbdSolver->getStructuralStiffness();
	const float shear = g_pbdSolver->getShearStiffness();
	const float bend = g_pbdSolver->getBendStiffness();
	const float damping = g_pbdSolver->getDampingFactor();

	g_mouseClickDown = false;
	g_mouseLClickButton = false;
	g_mouseRClickButton = false;
	g_mouseMClickButton = false;

	configurePBDDropSolver(
		stretch,
		shear,
		bend,
		damping,
		g_pbdDropRuntime.solverIterations,
		g_pbdDropRuntime.sphereRadius,
		g_pbdDropRuntime.pendingMeshResolution,
		false
	);
	logPBDDropControlState("reset-cloth");
	glutPostRedisplay();
}

static void logPBDDropControlState(const std::string& reason) {
	if (g_mode != SimMode::PBDDrop || g_pbdSolver == nullptr) return;
	std::cout
		<< "[pbd drop controls] " << reason
		<< " radius=" << g_pbdDropRuntime.sphereRadius
		<< ", iters=" << g_pbdSolver->getSolverIterations()
		<< ", stretch=" << g_pbdSolver->getStructuralStiffness()
		<< ", shear=" << g_pbdSolver->getShearStiffness()
		<< ", bend=" << g_pbdSolver->getBendStiffness()
		<< ", damping=" << g_pbdSolver->getDampingFactor()
		<< ", mesh=" << g_pbdDropRuntime.currentMeshResolution << "x" << g_pbdDropRuntime.currentMeshResolution;
	if (g_pbdDropRuntime.pendingMeshResolution != g_pbdDropRuntime.currentMeshResolution) {
		std::cout << ", pending-mesh=" << g_pbdDropRuntime.pendingMeshResolution << "x" << g_pbdDropRuntime.pendingMeshResolution;
	}
	std::cout << (g_pbdDropRuntime.paused ? ", paused" : ", running") << std::endl;
}

static unsigned int previousPBDFloorResolution(unsigned int resolution) {
	for (std::size_t i = 0; i < PBDFloorControlParam::meshChoices.size(); ++i) {
		if (PBDFloorControlParam::meshChoices[i] == resolution) {
			return (i == 0u)
				? PBDFloorControlParam::meshChoices.front()
				: PBDFloorControlParam::meshChoices[i - 1u];
		}
	}
	return PBDFloorControlParam::meshChoices.front();
}

static unsigned int nextPBDFloorResolution(unsigned int resolution) {
	for (std::size_t i = 0; i < PBDFloorControlParam::meshChoices.size(); ++i) {
		if (PBDFloorControlParam::meshChoices[i] == resolution) {
			return (i + 1u >= PBDFloorControlParam::meshChoices.size())
				? PBDFloorControlParam::meshChoices.back()
				: PBDFloorControlParam::meshChoices[i + 1u];
		}
	}
	return PBDFloorControlParam::meshChoices.back();
}

static int closestPBDFloorSpeedPreset(float dt) {
	int bestIndex = 0;
	float bestDistance = std::abs(dt - PBDFloorControlParam::speedPresets[0]);
	for (std::size_t i = 1; i < PBDFloorControlParam::speedPresets.size(); ++i) {
		const float distance = std::abs(dt - PBDFloorControlParam::speedPresets[i]);
		if (distance < bestDistance) {
			bestDistance = distance;
			bestIndex = static_cast<int>(i);
		}
	}
	return bestIndex;
}

static void setPBDFloorRuntimeTimestep(float dt, int presetIndex, bool custom) {
	g_pbdFloorRuntime.currentTimestep = std::max(PBDFloorCliParam::minTimestep, std::min(PBDFloorCliParam::maxTimestep, dt));
	g_pbdFloorRuntime.speedPresetIndex = presetIndex;
	g_pbdFloorRuntime.customTimestep = custom;
	if (g_pbdSystem != nullptr) {
		g_pbdSystem->time_step = g_pbdFloorRuntime.currentTimestep;
	}
}

static void configurePBDFloorSolver(
	float selfCollisionStiffness,
	unsigned int maxContactsPerVertex,
	float selfCollisionThickness,
	float floorFriction,
	float bend,
	float damping,
	unsigned int iterations,
	float timestep,
	unsigned int meshResolution,
	bool captureDefaults
) {
	if (UI != nullptr) {
		UI->releasePoint();
	}

	delete g_pbdSolver;
	g_pbdSolver = nullptr;
	delete g_pbdSystem;
	g_pbdSystem = nullptr;

	// Mesh resolution changes particle, constraint, and contact counts, so it
	// must be applied by rebuilding the cloth system instead of changing live.
	rebuildClothMesh(meshResolution);
	orientClothForFloorDrop();

	const float totalClothMass = PBDSystemParam::m * static_cast<float>(PBDSystemParam::n * PBDSystemParam::n);
	const float pointMass = totalClothMass / static_cast<float>(meshResolution * meshResolution);

	MassSpringBuilder builder;
	builder.uniformGrid(
		meshResolution,
		PBDSystemParam::h,
		PBDSystemParam::w / static_cast<float>(meshResolution - 1u),
		1.0f,
		pointMass,
		PBDSystemParam::a,
		PBDSystemParam::g
	);

	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSystem->time_step = timestep;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	g_pbdSolver->setSolverIterations(iterations);

	// Self-collision stiffness controls how strongly self-collision inequality
	// constraints are projected each iteration.
	g_pbdSolver->setSelfCollisionStiffness(std::max(
		PBDFloorControlParam::selfCollisionStiffnessMin,
		std::min(PBDFloorControlParam::selfCollisionStiffnessMax, selfCollisionStiffness)
	));
	// Max contacts per vertex prevents over-constraining/conflicting contacts.
	g_pbdSolver->setMaxSelfCollisionContactsPerVertex(std::max(
		PBDFloorControlParam::minSelfCollisionContacts,
		std::min(PBDFloorControlParam::maxSelfCollisionContacts, maxContactsPerVertex)
	));

	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}
	else {
		// Self-collision thickness is the h term in
		// C(q,p1,p2,p3) = (q - p1) . n - h >= 0.
		g_pbdSolver->setSelfCollisionThickness(std::max(
			PBDFloorControlParam::selfCollisionThicknessMin,
			std::min(PBDFloorControlParam::selfCollisionThicknessMax, selfCollisionThickness)
		));
	}

	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), bend);

	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);
	// Floor friction is velocity-level tangential damping after contact.
	g_pbdSolver->setPlaneFriction(std::max(
		PBDFloorControlParam::floorFrictionMin,
		std::min(PBDFloorControlParam::floorFrictionMax, floorFriction)
	));
	// Bend stiffness controls resistance to folding/unrolling.
	g_pbdSolver->setBendStiffness(std::max(
		PBDFloorControlParam::bendMin,
		std::min(PBDFloorControlParam::bendMax, bend)
	));
	// Damping controls velocity energy decay.
	g_pbdSolver->setDampingFactor(std::max(
		PBDFloorControlParam::dampingMin,
		std::min(PBDFloorControlParam::dampingMax, damping)
	));

	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, meshResolution);

	g_pbdFloorRuntime.currentMeshResolution = meshResolution;
	g_pbdFloorRuntime.pendingMeshResolution = meshResolution;
	g_pbdFloorRuntime.solverIterations = iterations;
	g_pbdFloorRuntime.debugEnabled = g_enableDebugDiagnostics;

	const int closestPreset = closestPBDFloorSpeedPreset(timestep);
	const bool customTimestep = std::abs(timestep - PBDFloorControlParam::speedPresets[closestPreset]) > 1e-5f;
	setPBDFloorRuntimeTimestep(timestep, closestPreset, customTimestep);

	if (captureDefaults || !g_pbdFloorRuntime.initialized) {
		g_pbdFloorRuntime.defaultSelfCollisionStiffness = g_pbdSolver->getSelfCollisionStiffness();
		g_pbdFloorRuntime.defaultMaxSelfCollisionContacts = g_pbdSolver->getMaxSelfCollisionContactsPerVertex();
		g_pbdFloorRuntime.defaultSelfCollisionThickness = g_pbdSolver->getSelfCollisionThickness();
		g_pbdFloorRuntime.defaultFloorFriction = g_pbdSolver->getPlaneFriction();
		g_pbdFloorRuntime.defaultBend = g_pbdSolver->getBendStiffness();
		g_pbdFloorRuntime.defaultDamping = g_pbdSolver->getDampingFactor();
		g_pbdFloorRuntime.initialized = true;
	}
}

static void resetPBDFloorDemo(bool resetParameters) {
	if (g_mode != SimMode::PBDDropFloor || g_pbdSolver == nullptr || !g_pbdFloorRuntime.initialized) return;

	if (resetParameters) {
		g_pbdSolver->setSelfCollisionStiffness(g_pbdFloorRuntime.defaultSelfCollisionStiffness);
		g_pbdSolver->setMaxSelfCollisionContactsPerVertex(g_pbdFloorRuntime.defaultMaxSelfCollisionContacts);
		if (g_selfCollisionThicknessOverride > 0.0f) {
			g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
		}
		else {
			g_pbdSolver->setSelfCollisionThickness(g_pbdFloorRuntime.defaultSelfCollisionThickness);
		}
		g_pbdSolver->setPlaneFriction(g_pbdFloorRuntime.defaultFloorFriction);
		g_pbdSolver->setBendStiffness(g_pbdFloorRuntime.defaultBend);
		g_pbdSolver->setDampingFactor(g_pbdFloorRuntime.defaultDamping);
		g_pbdFloorRuntime.pendingMeshResolution = PBDFloorControlParam::defaultMeshResolution;
		setPBDFloorRuntimeTimestep(PBDFloorControlParam::speedPresets[0], 0, false);
		g_enableDebugDiagnostics = g_pbdFloorRuntime.debugEnabled;
		logPBDFloorControlState("reset-parameters");
		glutPostRedisplay();
		return;
	}

	g_mouseClickDown = false;
	g_mouseLClickButton = false;
	g_mouseRClickButton = false;
	g_mouseMClickButton = false;

	configurePBDFloorSolver(
		g_pbdSolver->getSelfCollisionStiffness(),
		g_pbdSolver->getMaxSelfCollisionContactsPerVertex(),
		g_pbdSolver->getSelfCollisionThickness(),
		g_pbdSolver->getPlaneFriction(),
		g_pbdSolver->getBendStiffness(),
		g_pbdSolver->getDampingFactor(),
		g_pbdFloorRuntime.solverIterations,
		g_pbdFloorRuntime.currentTimestep,
		g_pbdFloorRuntime.pendingMeshResolution,
		false
	);
	logPBDFloorControlState("reset-cloth");
	glutPostRedisplay();
}

static void logPBDFloorControlState(const std::string& reason) {
	if (g_mode != SimMode::PBDDropFloor || g_pbdSolver == nullptr) return;
	std::cout
		<< "[pbd drop-floor controls] " << reason
		<< " dt=" << g_pbdFloorRuntime.currentTimestep
		<< ", iters=" << g_pbdFloorRuntime.solverIterations
		<< ", self-k=" << g_pbdSolver->getSelfCollisionStiffness()
		<< ", max-contacts=" << g_pbdSolver->getMaxSelfCollisionContactsPerVertex()
		<< ", self-thickness=" << g_pbdSolver->getSelfCollisionThickness()
		<< ", floor-friction=" << g_pbdSolver->getPlaneFriction()
		<< ", bend=" << g_pbdSolver->getBendStiffness()
		<< ", damping=" << g_pbdSolver->getDampingFactor()
		<< ", mesh=" << g_pbdFloorRuntime.currentMeshResolution << "x" << g_pbdFloorRuntime.currentMeshResolution;
	if (g_pbdFloorRuntime.pendingMeshResolution != g_pbdFloorRuntime.currentMeshResolution) {
		std::cout << ", pending-mesh=" << g_pbdFloorRuntime.pendingMeshResolution << "x" << g_pbdFloorRuntime.pendingMeshResolution;
	}
	std::cout
		<< ", debug=" << (g_enableDebugDiagnostics ? "on" : "off")
		<< (g_pbdFloorRuntime.paused ? ", paused" : ", running")
		<< std::endl;
	if (g_pbdFloorRuntime.speedPresetIndex > 0) {
		std::cout << "[pbd drop-floor controls] faster timestep may increase penetration, jitter, or missed self-collision" << std::endl;
	}
}

static unsigned int previousPBDDualResolution(unsigned int resolution) {
	for (std::size_t i = 0; i < PBDDualControlParam::meshChoices.size(); ++i) {
		if (PBDDualControlParam::meshChoices[i] == resolution) {
			return (i == 0u)
				? PBDDualControlParam::meshChoices.front()
				: PBDDualControlParam::meshChoices[i - 1u];
		}
	}
	return PBDDualControlParam::meshChoices.front();
}

static unsigned int nextPBDDualResolution(unsigned int resolution) {
	for (std::size_t i = 0; i < PBDDualControlParam::meshChoices.size(); ++i) {
		if (PBDDualControlParam::meshChoices[i] == resolution) {
			return (i + 1u >= PBDDualControlParam::meshChoices.size())
				? PBDDualControlParam::meshChoices.back()
				: PBDDualControlParam::meshChoices[i + 1u];
		}
	}
	return PBDDualControlParam::meshChoices.back();
}

static void setPBDDualRuntimeTimestep(float dt, int presetIndex, bool custom) {
	g_pbdDualRuntime.currentTimestep = std::max(PBDFloorCliParam::minTimestep, std::min(PBDFloorCliParam::maxTimestep, dt));
	g_pbdDualRuntime.speedPresetIndex = presetIndex;
	g_pbdDualRuntime.customTimestep = custom;
	if (g_pbdSystem != nullptr) {
		g_pbdSystem->time_step = g_pbdDualRuntime.currentTimestep;
	}
}

static void configurePBDDualSolver(
	float stretch,
	float shear,
	float bend,
	unsigned int iterations,
	float timestep,
	unsigned int meshResolution,
	float sphereRadius,
	float cubeSize,
	bool captureDefaults
) {
	if (UI != nullptr) {
		UI->releasePoint();
	}

	delete g_pbdSolver;
	g_pbdSolver = nullptr;
	delete g_pbdSystem;
	g_pbdSystem = nullptr;

	// Mesh resolution changes particle, constraint, and contact count, so it
	// must be applied through reset by rebuilding the cloth and solver state.
	rebuildClothMesh(meshResolution);

	const float clampedSphereRadius = std::max(
		PBDDualControlParam::sphereRadiusMin,
		std::min(PBDDualControlParam::sphereRadiusMax, sphereRadius)
	);
	const float clampedCubeSize = std::max(
		PBDDualControlParam::cubeSizeMin,
		std::min(PBDDualControlParam::cubeSizeMax, cubeSize)
	);
	orientClothFlatForDualFloorDrop(clampedSphereRadius, clampedCubeSize);

	const float totalClothMass = PBDSystemParam::m * static_cast<float>(PBDSystemParam::n * PBDSystemParam::n);
	const float pointMass = totalClothMass / static_cast<float>(meshResolution * meshResolution);

	MassSpringBuilder builder;
	builder.uniformGrid(
		meshResolution,
		PBDSystemParam::h,
		PBDSystemParam::w / static_cast<float>(meshResolution - 1u),
		1.0f,
		pointMass,
		PBDSystemParam::a,
		PBDSystemParam::g
	);

	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSystem->time_step = std::max(PBDFloorCliParam::minTimestep, std::min(PBDFloorCliParam::maxTimestep, timestep));
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	g_pbdSolver->setSolverIterations(iterations);
	g_pbdSolver->setSelfCollisionStiffness(PBDFloorDemoParam::selfCollisionStiffness);
	g_pbdSolver->setMaxSelfCollisionContactsPerVertex(PBDFloorDemoParam::maxSelfCollisionContactsPerVertex);
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}

	// Stretch and shear are distance-constraint stiffness controls.
	g_pbdSolver->addStructuralConstraints(
		builder.getStructIndex(),
		std::max(PBDDualControlParam::stretchMin, std::min(PBDDualControlParam::stretchMax, stretch))
	);
	g_pbdSolver->addShearConstraints(
		builder.getShearIndex(),
		std::max(PBDDualControlParam::shearMin, std::min(PBDDualControlParam::shearMax, shear))
	);
	// Bend controls dihedral bending stiffness.
	g_pbdSolver->addBendConstraints(
		builder.getBendIndex(),
		std::max(PBDDualControlParam::bendMin, std::min(PBDDualControlParam::bendMax, bend))
	);

	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);

	const glm::vec3 cubeHalfExtents(clampedCubeSize, clampedCubeSize, clampedCubeSize);
	// Sphere radius changes the smooth obstacle collision size.
	g_pbdSolver->addSphereCollider(
		Eigen::Vector3f(
			PBDDualObstacleDemoParam::sphereCenter.x,
			PBDDualObstacleDemoParam::sphereCenter.y,
			PBDDualObstacleDemoParam::sphereCenter.z
		),
		clampedSphereRadius
	);
	// Cube size changes the sharp analytic box collider and visual cube together.
	g_pbdSolver->addBoxCollider(
		Eigen::Vector3f(
			PBDDualObstacleDemoParam::cube.center.x,
			PBDDualObstacleDemoParam::cube.center.y,
			PBDDualObstacleDemoParam::cube.center.z
		),
		Eigen::Vector3f(cubeHalfExtents.x, cubeHalfExtents.y, cubeHalfExtents.z)
	);
	// Pending object sizes are applied only on reset so visuals and colliders
	// stay synchronized after the rebuild.
	initSphereColliderVisual(clampedSphereRadius, PBDDualObstacleDemoParam::sphereCenter);
	initCubeColliderVisual(PBDDualObstacleDemoParam::cube.center, cubeHalfExtents);

	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, meshResolution);

	g_pbdDualRuntime.currentMeshResolution = meshResolution;
	g_pbdDualRuntime.pendingMeshResolution = meshResolution;
	g_pbdDualRuntime.currentSphereRadius = clampedSphereRadius;
	g_pbdDualRuntime.pendingSphereRadius = clampedSphereRadius;
	g_pbdDualRuntime.currentCubeSize = clampedCubeSize;
	g_pbdDualRuntime.pendingCubeSize = clampedCubeSize;
	g_pbdDualRuntime.solverIterations = iterations;

	const int closestPreset = closestPBDFloorSpeedPreset(g_pbdSystem->time_step);
	const bool customTimestep = std::abs(g_pbdSystem->time_step - PBDFloorControlParam::speedPresets[closestPreset]) > 1e-5f;
	setPBDDualRuntimeTimestep(g_pbdSystem->time_step, closestPreset, customTimestep);

	if (captureDefaults || !g_pbdDualRuntime.initialized) {
		g_pbdDualRuntime.defaultStretch = g_pbdSolver->getStructuralStiffness();
		g_pbdDualRuntime.defaultShear = g_pbdSolver->getShearStiffness();
		g_pbdDualRuntime.defaultBend = g_pbdSolver->getBendStiffness();
		g_pbdDualRuntime.initialized = true;
	}
}

static void resetPBDDualDemo(bool resetParameters) {
	if (g_mode != SimMode::PBDDropFloorDual || g_pbdSolver == nullptr || !g_pbdDualRuntime.initialized) return;

	if (resetParameters) {
		g_pbdSolver->setStructuralStiffness(g_pbdDualRuntime.defaultStretch);
		g_pbdSolver->setShearStiffness(g_pbdDualRuntime.defaultShear);
		g_pbdSolver->setBendStiffness(g_pbdDualRuntime.defaultBend);
		setPBDDualRuntimeTimestep(PBDFloorControlParam::speedPresets[0], 0, false);
		g_pbdDualRuntime.pendingMeshResolution = PBDDualControlParam::defaultMeshResolution;
		g_pbdDualRuntime.pendingSphereRadius = PBDDualControlParam::defaultSphereRadius;
		g_pbdDualRuntime.pendingCubeSize = PBDDualControlParam::defaultCubeSize;
		logPBDDualControlState("reset-parameters");
		glutPostRedisplay();
		return;
	}

	g_mouseClickDown = false;
	g_mouseLClickButton = false;
	g_mouseRClickButton = false;
	g_mouseMClickButton = false;

	configurePBDDualSolver(
		g_pbdSolver->getStructuralStiffness(),
		g_pbdSolver->getShearStiffness(),
		g_pbdSolver->getBendStiffness(),
		g_pbdDualRuntime.solverIterations,
		g_pbdDualRuntime.currentTimestep,
		g_pbdDualRuntime.pendingMeshResolution,
		g_pbdDualRuntime.pendingSphereRadius,
		g_pbdDualRuntime.pendingCubeSize,
		false
	);
	logPBDDualControlState("reset-cloth");
	glutPostRedisplay();
}

static void logPBDDualControlState(const std::string& reason) {
	if (g_mode != SimMode::PBDDropFloorDual || g_pbdSolver == nullptr) return;
	std::cout
		<< "[pbd dual-obstacle controls] " << reason
		<< " iters=" << g_pbdDualRuntime.solverIterations
		<< ", dt=" << g_pbdDualRuntime.currentTimestep
		<< ", stretch=" << g_pbdSolver->getStructuralStiffness()
		<< ", shear=" << g_pbdSolver->getShearStiffness()
		<< ", bend=" << g_pbdSolver->getBendStiffness()
		<< ", mesh=" << g_pbdDualRuntime.currentMeshResolution << "x" << g_pbdDualRuntime.currentMeshResolution
		<< ", sphere=" << g_pbdDualRuntime.currentSphereRadius
		<< ", cube=" << g_pbdDualRuntime.currentCubeSize;
	if (g_pbdDualRuntime.pendingMeshResolution != g_pbdDualRuntime.currentMeshResolution) {
		std::cout << ", pending-mesh=" << g_pbdDualRuntime.pendingMeshResolution << "x" << g_pbdDualRuntime.pendingMeshResolution;
	}
	if (std::abs(g_pbdDualRuntime.pendingSphereRadius - g_pbdDualRuntime.currentSphereRadius) > 1e-5f) {
		std::cout << ", pending-sphere=" << g_pbdDualRuntime.pendingSphereRadius;
	}
	if (std::abs(g_pbdDualRuntime.pendingCubeSize - g_pbdDualRuntime.currentCubeSize) > 1e-5f) {
		std::cout << ", pending-cube=" << g_pbdDualRuntime.pendingCubeSize;
	}
	std::cout << (g_pbdDualRuntime.paused ? ", paused" : ", running") << std::endl;
}

static pbd_system* buildPBDSystem(const mass_spring_system& system) {
	pbd_system* pbdSystem = new pbd_system;
	pbdSystem->n_points = system.n_points;
	pbdSystem->n_constraints = system.n_springs;
	pbdSystem->time_step = system.time_step;
	pbdSystem->spring_list = system.spring_list;
	pbdSystem->rest_lengths = system.rest_lengths;
	pbdSystem->masses = system.masses;
	if (g_clothMesh != nullptr) {
		pbdSystem->triangle_indices.assign(
			g_clothMesh->ibuff(),
			g_clothMesh->ibuff() + g_clothMesh->ibuffLen()
		);
	}
	return pbdSystem;
}

static void demo_hang() {
	// short hand
	const int n = SystemParam::n;

	// initialize mass spring system
	MassSpringBuilder massSpringBuilder;
	massSpringBuilder.uniformGrid(
		SystemParam::n,
		SystemParam::h,
		SystemParam::r,
		SystemParam::k,
		SystemParam::m,
		SystemParam::a,
		SystemParam::g
	);
	g_system = massSpringBuilder.getResult();

	// initialize mass spring solver
	g_solver = new MassSpringSolver(g_system, g_clothMesh->vbuff());

	// deformation constraint parameters
	const float tauc = 0.4f; // critical spring deformation | 0.4f
	const unsigned int deformIter = 15; // number of iterations | 15

	// initialize constraints
	// spring deformation constraint
	CgSpringDeformationNode* deformationNode =
		new CgSpringDeformationNode(g_system, g_clothMesh->vbuff(), tauc, deformIter);
	deformationNode->addSprings(massSpringBuilder.getShearIndex());
	deformationNode->addSprings(massSpringBuilder.getStructIndex());

	// fix top corners
	CgPointFixNode* cornerFixer = new CgPointFixNode(g_system, g_clothMesh->vbuff());
	cornerFixer->fixPoint(0);
	cornerFixer->fixPoint(n - 1);

	// initialize user interaction
	CgPointFixNode* mouseFixer = new CgPointFixNode(g_system, g_clothMesh->vbuff());
	initMouseInteraction(mouseFixer, n);

	// build constraint graph
	g_cgRootNode = new CgRootNode(g_system, g_clothMesh->vbuff());

	// first layer
	g_cgRootNode->addChild(deformationNode);

	// second layer
	deformationNode->addChild(cornerFixer);
	deformationNode->addChild(mouseFixer);
}

static void demo_drop() {
	// short hand
	const int n = SystemParam::n;

	// initialize mass spring system
	MassSpringBuilder massSpringBuilder;
	massSpringBuilder.uniformGrid(
		SystemParam::n,
		SystemParam::h,
		SystemParam::r,
		SystemParam::k,
		SystemParam::m,
		SystemParam::a,
		SystemParam::g
	);
	g_system = massSpringBuilder.getResult();

	// initialize mass spring solver
	g_solver = new MassSpringSolver(g_system, g_clothMesh->vbuff());

	// sphere collision constraint parameters
	const float radius = 0.64f; // sphere radius | 0.64f
	const Eigen::Vector3f center(0, 0, -1);// sphere center | (0, 0, -1)

	// deformation constraint parameters
	const float tauc = 0.12f; // critical spring deformation | 0.12f
	const unsigned int deformIter = 15; // number of iterations | 15

	// initialize constraints
	// sphere collision constraint
	CgSphereCollisionNode* sphereCollisionNode =
		new CgSphereCollisionNode(g_system, g_clothMesh->vbuff(), radius, center);
	initSphereColliderVisual(radius, glm::vec3(center[0], center[1], center[2]));

	// spring deformation constraint
	CgSpringDeformationNode* deformationNode =
		new CgSpringDeformationNode(g_system, g_clothMesh->vbuff(), tauc, deformIter);
	deformationNode->addSprings(massSpringBuilder.getShearIndex());
	deformationNode->addSprings(massSpringBuilder.getStructIndex());

	// initialize user interaction
	CgPointFixNode* mouseFixer = new CgPointFixNode(g_system, g_clothMesh->vbuff());
	initMouseInteraction(mouseFixer, n);

	// build constraint graph
	g_cgRootNode = new CgRootNode(g_system, g_clothMesh->vbuff());

	// first layer
	g_cgRootNode->addChild(deformationNode);
	g_cgRootNode->addChild(sphereCollisionNode);

	// second layer
	deformationNode->addChild(mouseFixer);
}

static void demo_pbd_hang() {
	const unsigned int iterations = (g_pbdHangIterationsOverride > 0u)
		? g_pbdHangIterationsOverride
		: static_cast<unsigned int>(PBDSystemParam::n_iter);
	configurePBDHangSolver(
		PBDSystemParam::k_stretch,
		PBDSystemParam::k_shear,
		PBDSystemParam::k_bend,
		PBDHangControlParam::defaultDamping,
		iterations,
		true
	);
	logPBDHangControlState("startup");
}

static void demo_pbd_hang_wind() {
	configurePBDWindSolver(g_windSpeed, g_windDirection, true);
	logPBDWindControlState("startup");
}

static void demo_pbd_drop() {
	const unsigned int iterations = (g_pbdDropIterationsOverride > 0u)
		? g_pbdDropIterationsOverride
		: static_cast<unsigned int>(PBDSystemParam::n_iter);
	configurePBDDropSolver(
		PBDSystemParam::k_stretch,
		PBDSystemParam::k_shear,
		PBDSystemParam::k_bend,
		PBDDropControlParam::defaultDamping,
		iterations,
		g_pbdDropStartupSphereRadius,
		PBDDropControlParam::defaultMeshResolution,
		true
	);
	logPBDDropControlState("startup");
}

static void demo_pbd_drop_floor() {
	const unsigned int iterations = (g_pbdFloorIterationsOverride > 0u)
		? g_pbdFloorIterationsOverride
		: static_cast<unsigned int>(PBDFloorDemoParam::n_iter);
	const float timestep = (g_pbdFloorTimestepOverride > 0.0f)
		? g_pbdFloorTimestepOverride
		: PBDFloorDemoParam::h;
	configurePBDFloorSolver(
		PBDFloorDemoParam::selfCollisionStiffness,
		PBDFloorDemoParam::maxSelfCollisionContactsPerVertex,
		(g_selfCollisionThicknessOverride > 0.0f) ? g_selfCollisionThicknessOverride : PBDFloorControlParam::selfCollisionThicknessMin,
		0.15f,
		PBDSystemParam::k_bend,
		0.07f,
		iterations,
		timestep,
		PBDFloorControlParam::defaultMeshResolution,
		true
	);
	logPBDFloorControlState("startup");
}

static void demo_pbd_drop_floor_dual() {
	const unsigned int iterations = (g_pbdDualIterationsOverride > 0u)
		? g_pbdDualIterationsOverride
		: static_cast<unsigned int>(PBDFloorDemoParam::n_iter);
	configurePBDDualSolver(
		PBDSystemParam::k_stretch,
		PBDSystemParam::k_shear,
		PBDSystemParam::k_bend,
		iterations,
		PBDFloorDemoParam::h,
		PBDDualControlParam::defaultMeshResolution,
		PBDDualControlParam::defaultSphereRadius,
		PBDDualControlParam::defaultCubeSize,
		true
	);
	logPBDDualControlState("startup");
}
// G L U T  C A L L B A C K S //////////////////////////////////////////////////////
static void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (hasSceneFloor()) {
		drawFloor();
		drawFloorShadows();
	}
	if (hasSphereColliderVisual() && g_sphere_target != nullptr) {
		Renderer renderer;
		renderer.setProgram(g_phongShader);
		renderer.setModelview(g_ModelViewMatrix);
		renderer.setProjection(g_ProjectionMatrix);
		g_phongShader->setUseFlagPattern(false);
		g_phongShader->setAlbedo(g_sphere_albedo);
		g_phongShader->setAmbient(g_sphere_ambient);
		g_phongShader->setLight(g_light);
		g_phongShader->setSpecularStrength(g_specular_strength);
		g_phongShader->setShininess(g_shininess);
		renderer.setProgramInput(g_sphere_target);
		renderer.setElementCount(g_sphere_index_count);
		renderer.draw();
	}
	if (hasCubeColliderVisual() && g_cube_target != nullptr) {
		Renderer renderer;
		renderer.setProgram(g_phongShader);
		renderer.setModelview(g_ModelViewMatrix);
		renderer.setProjection(g_ProjectionMatrix);
		g_phongShader->setUseFlagPattern(false);
		g_phongShader->setAlbedo(g_sphere_albedo);
		g_phongShader->setAmbient(g_sphere_ambient);
		g_phongShader->setLight(g_light);
		g_phongShader->setSpecularStrength(g_specular_strength);
		g_phongShader->setShininess(g_shininess);
		renderer.setProgramInput(g_cube_target);
		renderer.setElementCount(g_cube_index_count);
		renderer.draw();
	}
	drawCloth();
	drawPBDHangOverlay();
	drawPBDWindOverlay();
	drawPBDDropOverlay();
	drawPBDFloorOverlay();
	drawPBDDualOverlay();
	glutSwapBuffers();

	checkGlErrors();
}

static void reshape(int w, int h) {
	g_windowWidth = w;
	g_windowHeight = h;
	glViewport(0, 0, w, h);
	updateProjection();
	glutPostRedisplay();
}

static void keyboard(unsigned char key, int, int) {
	if (g_pbdSolver == nullptr) return;

	if (g_mode == SimMode::PBDHangWind) {
		switch (key) {
		case '1':
			applyPBDWindSettings(
				g_pbdWindRuntime.currentWindSpeed - PBDWindControlParam::speedStep,
				g_pbdWindRuntime.currentDirectionInput
			);
			logPBDWindControlState("speed-");
			break;
		case '2':
			applyPBDWindSettings(
				g_pbdWindRuntime.currentWindSpeed + PBDWindControlParam::speedStep,
				g_pbdWindRuntime.currentDirectionInput
			);
			logPBDWindControlState("speed+");
			break;
		case '3': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.x() = std::max(PBDWindControlParam::directionMin, direction.x() - PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-x-");
			break;
		}
		case '4': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.x() = std::min(PBDWindControlParam::directionMax, direction.x() + PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-x+");
			break;
		}
		case '5': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.y() = std::max(PBDWindControlParam::directionMin, direction.y() - PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-y-");
			break;
		}
		case '6': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.y() = std::min(PBDWindControlParam::directionMax, direction.y() + PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-y+");
			break;
		}
		case '7': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.z() = std::max(PBDWindControlParam::directionMin, direction.z() - PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-z-");
			break;
		}
		case '8': {
			Eigen::Vector3f direction = g_pbdWindRuntime.currentDirectionInput;
			direction.z() = std::min(PBDWindControlParam::directionMax, direction.z() + PBDWindControlParam::directionStep);
			if (direction.norm() >= PBDWindCliParam::minDirectionNorm) {
				applyPBDWindSettings(g_pbdWindRuntime.currentWindSpeed, direction);
			}
			logPBDWindControlState("dir-z+");
			break;
		}
		case 'r':
		case 'R':
			resetPBDWindDemo(false);
			break;
		case 't':
		case 'T':
			resetPBDWindDemo(true);
			break;
		case 'p':
		case 'P':
			g_pbdWindRuntime.paused = !g_pbdWindRuntime.paused;
			logPBDWindControlState(g_pbdWindRuntime.paused ? "pause" : "resume");
			break;
		default:
			return;
		}

		glutPostRedisplay();
		return;
	}

	if (g_mode == SimMode::PBDHang) {
		switch (key) {
		case '1':
			// Stretch is a distance-constraint stiffness control.
			g_pbdSolver->setStructuralStiffness(std::max(
				PBDHangControlParam::stretchMin,
				g_pbdSolver->getStructuralStiffness() - PBDHangControlParam::stretchStep
			));
			logPBDHangControlState("stretch-");
			break;
		case '2':
			g_pbdSolver->setStructuralStiffness(std::min(
				PBDHangControlParam::stretchMax,
				g_pbdSolver->getStructuralStiffness() + PBDHangControlParam::stretchStep
			));
			logPBDHangControlState("stretch+");
			break;
		case '3':
			// Shear is also a distance-constraint stiffness control.
			g_pbdSolver->setShearStiffness(std::max(
				PBDHangControlParam::shearMin,
				g_pbdSolver->getShearStiffness() - PBDHangControlParam::shearStep
			));
			logPBDHangControlState("shear-");
			break;
		case '4':
			g_pbdSolver->setShearStiffness(std::min(
				PBDHangControlParam::shearMax,
				g_pbdSolver->getShearStiffness() + PBDHangControlParam::shearStep
			));
			logPBDHangControlState("shear+");
			break;
		case '5':
			// Bend controls dihedral bending stiffness.
			g_pbdSolver->setBendStiffness(std::max(
				PBDHangControlParam::bendMin,
				g_pbdSolver->getBendStiffness() - PBDHangControlParam::bendStep
			));
			logPBDHangControlState("bend-");
			break;
		case '6':
			g_pbdSolver->setBendStiffness(std::min(
				PBDHangControlParam::bendMax,
				g_pbdSolver->getBendStiffness() + PBDHangControlParam::bendStep
			));
			logPBDHangControlState("bend+");
			break;
		case '7':
			// Damping controls velocity energy decay and keeps gravity unchanged.
			g_pbdSolver->setDampingFactor(std::max(
				PBDHangControlParam::dampingMin,
				g_pbdSolver->getDampingFactor() - PBDHangControlParam::dampingStep
			));
			logPBDHangControlState("damping-");
			break;
		case '8':
			g_pbdSolver->setDampingFactor(std::min(
				PBDHangControlParam::dampingMax,
				g_pbdSolver->getDampingFactor() + PBDHangControlParam::dampingStep
			));
			logPBDHangControlState("damping+");
			break;
		case 'r':
		case 'R':
			resetPBDHangDemo(false);
			break;
		case 't':
		case 'T':
			resetPBDHangDemo(true);
			break;
		case 'p':
		case 'P':
			g_pbdHangRuntime.paused = !g_pbdHangRuntime.paused;
			logPBDHangControlState(g_pbdHangRuntime.paused ? "pause" : "resume");
			break;
		default:
			return;
		}

		glutPostRedisplay();
		return;
	}

	if (g_mode != SimMode::PBDDrop) {
		if (g_mode == SimMode::PBDDropFloor) {
			switch (key) {
			case '1':
				// Self-collision stiffness controls how strongly self-collision
				// inequality constraints are projected.
				g_pbdSolver->setSelfCollisionStiffness(std::max(
					PBDFloorControlParam::selfCollisionStiffnessMin,
					g_pbdSolver->getSelfCollisionStiffness() - PBDFloorControlParam::selfCollisionStiffnessStep
				));
				logPBDFloorControlState("self-k-");
				break;
			case '2':
				g_pbdSolver->setSelfCollisionStiffness(std::min(
					PBDFloorControlParam::selfCollisionStiffnessMax,
					g_pbdSolver->getSelfCollisionStiffness() + PBDFloorControlParam::selfCollisionStiffnessStep
				));
				logPBDFloorControlState("self-k+");
				break;
			case '3':
				g_pbdSolver->setMaxSelfCollisionContactsPerVertex(std::max(
					PBDFloorControlParam::minSelfCollisionContacts,
					g_pbdSolver->getMaxSelfCollisionContactsPerVertex() - 1u
				));
				logPBDFloorControlState("max-contacts-");
				break;
			case '4':
				g_pbdSolver->setMaxSelfCollisionContactsPerVertex(std::min(
					PBDFloorControlParam::maxSelfCollisionContacts,
					g_pbdSolver->getMaxSelfCollisionContactsPerVertex() + 1u
				));
				logPBDFloorControlState("max-contacts+");
				break;
			case '5':
				g_pbdSolver->setSelfCollisionThickness(std::max(
					PBDFloorControlParam::selfCollisionThicknessMin,
					g_pbdSolver->getSelfCollisionThickness() - PBDFloorControlParam::selfCollisionThicknessStep
				));
				logPBDFloorControlState("self-thickness-");
				break;
			case '6':
				g_pbdSolver->setSelfCollisionThickness(std::min(
					PBDFloorControlParam::selfCollisionThicknessMax,
					g_pbdSolver->getSelfCollisionThickness() + PBDFloorControlParam::selfCollisionThicknessStep
				));
				logPBDFloorControlState("self-thickness+");
				break;
			case '7':
				// Floor friction is velocity-level tangential damping after contact.
				g_pbdSolver->setPlaneFriction(std::max(
					PBDFloorControlParam::floorFrictionMin,
					g_pbdSolver->getPlaneFriction() - PBDFloorControlParam::floorFrictionStep
				));
				logPBDFloorControlState("floor-friction-");
				break;
			case '8':
				g_pbdSolver->setPlaneFriction(std::min(
					PBDFloorControlParam::floorFrictionMax,
					g_pbdSolver->getPlaneFriction() + PBDFloorControlParam::floorFrictionStep
				));
				logPBDFloorControlState("floor-friction+");
				break;
			case 'q':
			case 'Q':
				g_pbdSolver->setBendStiffness(std::max(
					PBDFloorControlParam::bendMin,
					g_pbdSolver->getBendStiffness() - PBDFloorControlParam::bendStep
				));
				logPBDFloorControlState("bend-");
				break;
			case 'w':
			case 'W':
				g_pbdSolver->setBendStiffness(std::min(
					PBDFloorControlParam::bendMax,
					g_pbdSolver->getBendStiffness() + PBDFloorControlParam::bendStep
				));
				logPBDFloorControlState("bend+");
				break;
			case 'a':
			case 'A':
				g_pbdSolver->setDampingFactor(std::max(
					PBDFloorControlParam::dampingMin,
					g_pbdSolver->getDampingFactor() - PBDFloorControlParam::dampingStep
				));
				logPBDFloorControlState("damping-");
				break;
			case 's':
			case 'S':
				g_pbdSolver->setDampingFactor(std::min(
					PBDFloorControlParam::dampingMax,
					g_pbdSolver->getDampingFactor() + PBDFloorControlParam::dampingStep
				));
				logPBDFloorControlState("damping+");
				break;
			case '[':
				g_pbdFloorRuntime.pendingMeshResolution = previousPBDFloorResolution(g_pbdFloorRuntime.pendingMeshResolution);
				logPBDFloorControlState("pending-mesh-");
				break;
			case ']':
				g_pbdFloorRuntime.pendingMeshResolution = nextPBDFloorResolution(g_pbdFloorRuntime.pendingMeshResolution);
				logPBDFloorControlState("pending-mesh+");
				break;
			case '9': {
				const int nextIndex = std::max(0, g_pbdFloorRuntime.speedPresetIndex - 1);
				setPBDFloorRuntimeTimestep(PBDFloorControlParam::speedPresets[nextIndex], nextIndex, false);
				logPBDFloorControlState("speed-");
				break;
			}
			case '0': {
				const int nextIndex = std::min(
					static_cast<int>(PBDFloorControlParam::speedPresets.size()) - 1,
					g_pbdFloorRuntime.speedPresetIndex + 1
				);
				setPBDFloorRuntimeTimestep(PBDFloorControlParam::speedPresets[nextIndex], nextIndex, false);
				logPBDFloorControlState("speed+");
				break;
			}
			case 'r':
			case 'R':
				resetPBDFloorDemo(false);
				break;
			case 't':
			case 'T':
				resetPBDFloorDemo(true);
				break;
			case 'p':
			case 'P':
				g_pbdFloorRuntime.paused = !g_pbdFloorRuntime.paused;
				logPBDFloorControlState(g_pbdFloorRuntime.paused ? "pause" : "resume");
				break;
			case 'd':
			case 'D':
				g_enableDebugDiagnostics = !g_enableDebugDiagnostics;
				g_pbdFloorRuntime.debugEnabled = g_enableDebugDiagnostics;
				logPBDFloorControlState(g_enableDebugDiagnostics ? "debug-on" : "debug-off");
				break;
			default:
				return;
			}

			glutPostRedisplay();
			return;
		}

		if (g_mode != SimMode::PBDDropFloorDual) {
			return;
		}

		switch (key) {
		case '1':
			// Stretch is a distance-constraint stiffness control.
			g_pbdSolver->setStructuralStiffness(std::max(
				PBDDualControlParam::stretchMin,
				g_pbdSolver->getStructuralStiffness() - PBDDualControlParam::stretchStep
			));
			logPBDDualControlState("stretch-");
			break;
		case '2':
			g_pbdSolver->setStructuralStiffness(std::min(
				PBDDualControlParam::stretchMax,
				g_pbdSolver->getStructuralStiffness() + PBDDualControlParam::stretchStep
			));
			logPBDDualControlState("stretch+");
			break;
		case '3':
			// Shear is also a distance-constraint stiffness control.
			g_pbdSolver->setShearStiffness(std::max(
				PBDDualControlParam::shearMin,
				g_pbdSolver->getShearStiffness() - PBDDualControlParam::shearStep
			));
			logPBDDualControlState("shear-");
			break;
		case '4':
			g_pbdSolver->setShearStiffness(std::min(
				PBDDualControlParam::shearMax,
				g_pbdSolver->getShearStiffness() + PBDDualControlParam::shearStep
			));
			logPBDDualControlState("shear+");
			break;
		case '5':
			// Bend controls dihedral bending stiffness.
			g_pbdSolver->setBendStiffness(std::max(
				PBDDualControlParam::bendMin,
				g_pbdSolver->getBendStiffness() - PBDDualControlParam::bendStep
			));
			logPBDDualControlState("bend-");
			break;
		case '6':
			g_pbdSolver->setBendStiffness(std::min(
				PBDDualControlParam::bendMax,
				g_pbdSolver->getBendStiffness() + PBDDualControlParam::bendStep
			));
			logPBDDualControlState("bend+");
			break;
		case '[':
			// Mesh resolution changes particle, constraint, and contact count and
			// is applied only on reset when the cloth system is rebuilt.
			g_pbdDualRuntime.pendingMeshResolution = previousPBDDualResolution(g_pbdDualRuntime.pendingMeshResolution);
			logPBDDualControlState("pending-mesh-");
			break;
		case ']':
			g_pbdDualRuntime.pendingMeshResolution = nextPBDDualResolution(g_pbdDualRuntime.pendingMeshResolution);
			logPBDDualControlState("pending-mesh+");
			break;
		case 'q':
		case 'Q':
			// Sphere radius changes the smooth obstacle collision size.
			g_pbdDualRuntime.pendingSphereRadius = std::max(
				PBDDualControlParam::sphereRadiusMin,
				g_pbdDualRuntime.pendingSphereRadius - PBDDualControlParam::objectSizeStep
			);
			logPBDDualControlState("pending-sphere-");
			break;
		case 'w':
		case 'W':
			g_pbdDualRuntime.pendingSphereRadius = std::min(
				PBDDualControlParam::sphereRadiusMax,
				g_pbdDualRuntime.pendingSphereRadius + PBDDualControlParam::objectSizeStep
			);
			logPBDDualControlState("pending-sphere+");
			break;
		case 'a':
		case 'A':
			// Cube size changes the sharp analytic box collider and visual cube together.
			g_pbdDualRuntime.pendingCubeSize = std::max(
				PBDDualControlParam::cubeSizeMin,
				g_pbdDualRuntime.pendingCubeSize - PBDDualControlParam::objectSizeStep
			);
			logPBDDualControlState("pending-cube-");
			break;
		case 's':
		case 'S':
			g_pbdDualRuntime.pendingCubeSize = std::min(
				PBDDualControlParam::cubeSizeMax,
				g_pbdDualRuntime.pendingCubeSize + PBDDualControlParam::objectSizeStep
			);
			logPBDDualControlState("pending-cube+");
			break;
		case '9': {
			const int nextIndex = std::max(0, g_pbdDualRuntime.speedPresetIndex - 1);
			setPBDDualRuntimeTimestep(PBDFloorControlParam::speedPresets[nextIndex], nextIndex, false);
			logPBDDualControlState("speed-");
			break;
		}
		case '0': {
			const int nextIndex = std::min(
				static_cast<int>(PBDFloorControlParam::speedPresets.size()) - 1,
				g_pbdDualRuntime.speedPresetIndex + 1
			);
			setPBDDualRuntimeTimestep(PBDFloorControlParam::speedPresets[nextIndex], nextIndex, false);
			logPBDDualControlState("speed+");
			break;
		}
		case 'r':
		case 'R':
			resetPBDDualDemo(false);
			break;
		case 't':
		case 'T':
			resetPBDDualDemo(true);
			break;
		case 'p':
		case 'P':
			g_pbdDualRuntime.paused = !g_pbdDualRuntime.paused;
			logPBDDualControlState(g_pbdDualRuntime.paused ? "pause" : "resume");
			break;
		default:
			return;
		}

		glutPostRedisplay();
		return;
	}

	switch (key) {
	case '1':
		// Stretch is a distance-constraint stiffness control.
		g_pbdSolver->setStructuralStiffness(std::max(
			PBDDropControlParam::stretchMin,
			g_pbdSolver->getStructuralStiffness() - PBDDropControlParam::stretchStep
		));
		logPBDDropControlState("stretch-");
		break;
	case '2':
		g_pbdSolver->setStructuralStiffness(std::min(
			PBDDropControlParam::stretchMax,
			g_pbdSolver->getStructuralStiffness() + PBDDropControlParam::stretchStep
		));
		logPBDDropControlState("stretch+");
		break;
	case '3':
		// Shear is also a distance-constraint stiffness control.
		g_pbdSolver->setShearStiffness(std::max(
			PBDDropControlParam::shearMin,
			g_pbdSolver->getShearStiffness() - PBDDropControlParam::shearStep
		));
		logPBDDropControlState("shear-");
		break;
	case '4':
		g_pbdSolver->setShearStiffness(std::min(
			PBDDropControlParam::shearMax,
			g_pbdSolver->getShearStiffness() + PBDDropControlParam::shearStep
		));
		logPBDDropControlState("shear+");
		break;
	case '5':
		// Bend controls dihedral bending stiffness.
		g_pbdSolver->setBendStiffness(std::max(
			PBDDropControlParam::bendMin,
			g_pbdSolver->getBendStiffness() - PBDDropControlParam::bendStep
		));
		logPBDDropControlState("bend-");
		break;
	case '6':
		g_pbdSolver->setBendStiffness(std::min(
			PBDDropControlParam::bendMax,
			g_pbdSolver->getBendStiffness() + PBDDropControlParam::bendStep
		));
		logPBDDropControlState("bend+");
		break;
	case '7':
		// Damping controls velocity energy decay and keeps gravity unchanged.
		g_pbdSolver->setDampingFactor(std::max(
			PBDDropControlParam::dampingMin,
			g_pbdSolver->getDampingFactor() - PBDDropControlParam::dampingStep
		));
		logPBDDropControlState("damping-");
		break;
	case '8':
		g_pbdSolver->setDampingFactor(std::min(
			PBDDropControlParam::dampingMax,
			g_pbdSolver->getDampingFactor() + PBDDropControlParam::dampingStep
		));
		logPBDDropControlState("damping+");
		break;
	case '[':
		g_pbdDropRuntime.pendingMeshResolution = previousPBDDropResolution(g_pbdDropRuntime.pendingMeshResolution);
		logPBDDropControlState("pending-mesh-");
		break;
	case ']':
		g_pbdDropRuntime.pendingMeshResolution = nextPBDDropResolution(g_pbdDropRuntime.pendingMeshResolution);
		logPBDDropControlState("pending-mesh+");
		break;
	case 'r':
	case 'R':
		resetPBDDropDemo(false);
		break;
	case 't':
	case 'T':
		resetPBDDropDemo(true);
		break;
	case 'p':
	case 'P':
		g_pbdDropRuntime.paused = !g_pbdDropRuntime.paused;
		logPBDDropControlState(g_pbdDropRuntime.paused ? "pause" : "resume");
		break;
	default:
		return;
	}

	glutPostRedisplay();
}

static void mouse(const int button, const int state, const int x, const int y) {
	g_mouseClickX = x;
	g_mouseClickY = g_windowHeight - y - 1;

	g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
	g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
	g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

	g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
	g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
	g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

	g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

	// TODO: move to UserInteraction class: add renderer member variable
	// pick point
	if (g_mouseLClickButton && UI != nullptr) {
		UI->setModelview(g_ModelViewMatrix);
		UI->setProjection(g_ProjectionMatrix);
		UI->grabPoint(g_mouseClickX, g_mouseClickY);
	}
	else if (UI != nullptr) UI->releasePoint();
}

static void motion(const int x, const int y) {
	const float dx = float(x - g_mouseClickX);
	const float dy = float (-(g_windowHeight - y - 1 - g_mouseClickY));

	if (g_mouseLClickButton && UI != nullptr) {
		//glm::vec3 ux(g_ModelViewMatrix * glm::vec4(1, 0, 0, 0));
		//glm::vec3 uy(g_ModelViewMatrix * glm::vec4(0, 1, 0, 0));
		glm::vec3 ux(0, 1, 0);
		glm::vec3 uy(0, 0, -1);
		UI->movePoint(0.01f * (dx * ux + dy * uy));
	}

	g_mouseClickX = x;
	g_mouseClickY = g_windowHeight - y - 1;
}

// C L O T H ///////////////////////////////////////////////////////////////////////
static void drawFloor() {
	Renderer renderer;
	renderer.setProgram(g_phongShader);
	renderer.setModelview(g_ModelViewMatrix);
	renderer.setProjection(g_ProjectionMatrix);
	g_phongShader->setUseFlagPattern(false);
	g_phongShader->setAlbedo(g_floor_albedo);
	g_phongShader->setAmbient(g_floor_ambient);
	g_phongShader->setLight(g_light);
	g_phongShader->setSpecularStrength(0.12f);
	g_phongShader->setShininess(18.0f);
	renderer.setProgramInput(g_floor_target);
	renderer.setElementCount(6);
	renderer.draw();
}

static void drawFloorShadows() {
	if (!hasSceneFloor() || g_shadowShader == nullptr) return;

	const float shadowPlaneHeight = g_floor_collision_height + g_floor_render_offset + 1e-3f;
	const glm::mat4 shadowModelView = g_ModelViewMatrix * floorShadowMatrix(shadowPlaneHeight, glm::normalize(g_light));

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_FALSE);

	Renderer renderer;
	renderer.setProgram(g_shadowShader);
	renderer.setProjection(g_ProjectionMatrix);
	renderer.setModelview(shadowModelView);
	g_shadowShader->setShadowColor(g_shadow_color);

	renderer.setProgramInput(g_render_target);
	renderer.setElementCount(g_clothMesh->ibuffLen());
	renderer.draw();

	if (hasSphereColliderVisual() && g_sphere_target != nullptr) {
		renderer.setProgramInput(g_sphere_target);
		renderer.setElementCount(g_sphere_index_count);
		renderer.draw();
	}
	if (hasCubeColliderVisual() && g_cube_target != nullptr) {
		renderer.setProgramInput(g_cube_target);
		renderer.setElementCount(g_cube_index_count);
		renderer.draw();
	}

	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
}

static void drawCloth() {
	Renderer renderer;
	renderer.setProgram(g_phongShader);
	renderer.setModelview(g_ModelViewMatrix);
	renderer.setProjection(g_ProjectionMatrix);
	g_phongShader->setUseFlagPattern(g_mode == SimMode::PBDHangWind);
	g_phongShader->setAlbedo(g_albedo);
	g_phongShader->setAmbient(g_ambient);
	g_phongShader->setLight(g_light);
	g_phongShader->setSpecularStrength(g_specular_strength);
	g_phongShader->setShininess(g_shininess);
	renderer.setProgramInput(g_render_target);
	renderer.setElementCount(g_clothMesh->ibuffLen());
	renderer.draw();
}

static void drawBitmapText(float x, float y, const std::string& text) {
	glRasterPos2f(x, y);
	for (char c : text) {
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, c);
	}
}

static void drawWindDirectionIndicator(const Eigen::Vector3f& normalizedDirection) {
	glm::vec2 horizontalDirection(normalizedDirection.x(), normalizedDirection.y());
	const float horizontalMagnitude = glm::length(horizontalDirection);

	const glm::vec2 boxMin(18.0f, 18.0f);
	const glm::vec2 boxMax(130.0f, 130.0f);
	const glm::vec2 center = 0.5f * (boxMin + boxMax);
	const float shaftLength = 34.0f;
	const float headLength = 12.0f;
	const float headWidth = 7.0f;
	if (horizontalMagnitude > 1e-5f) {
		horizontalDirection /= horizontalMagnitude;
	}
	const glm::vec2 tip = center + shaftLength * horizontalDirection;
	const glm::vec2 headBase = tip - headLength * horizontalDirection;
	const glm::vec2 normal(-horizontalDirection.y, horizontalDirection.x);

	const glm::vec2 verticalAnchor(boxMax.x + 26.0f, center.y);
	const float zMagnitude = std::min(1.0f, std::abs(normalizedDirection.z()));
	const float verticalShaftLength = 18.0f + 18.0f * zMagnitude;
	const float verticalHeadLength = 10.0f;
	const float verticalHeadWidth = 6.0f;
	const float verticalSign = (normalizedDirection.z() >= 0.0f) ? 1.0f : -1.0f;
	const glm::vec2 verticalDirection(0.0f, verticalSign);
	const glm::vec2 verticalTip = verticalAnchor + verticalShaftLength * verticalDirection;
	const glm::vec2 verticalHeadBase = verticalTip - verticalHeadLength * verticalDirection;

	glColor3f(0.72f, 0.78f, 0.88f);
	glLineWidth(1.5f);
	glBegin(GL_LINE_LOOP);
	glVertex2f(boxMin.x, boxMin.y);
	glVertex2f(boxMax.x, boxMin.y);
	glVertex2f(boxMax.x, boxMax.y);
	glVertex2f(boxMin.x, boxMax.y);
	glEnd();

	glBegin(GL_LINES);
	glVertex2f(center.x - 5.0f, center.y);
	glVertex2f(center.x + 5.0f, center.y);
	glVertex2f(center.x, center.y - 5.0f);
	glVertex2f(center.x, center.y + 5.0f);
	glEnd();

	glColor3f(0.95f, 0.97f, 1.0f);
	if (horizontalMagnitude > 1e-5f) {
		glLineWidth(2.5f);
		glBegin(GL_LINES);
		glVertex2f(center.x, center.y);
		glVertex2f(tip.x, tip.y);
		glEnd();

		glBegin(GL_TRIANGLES);
		glVertex2f(tip.x, tip.y);
		glVertex2f(headBase.x + headWidth * normal.x, headBase.y + headWidth * normal.y);
		glVertex2f(headBase.x - headWidth * normal.x, headBase.y - headWidth * normal.y);
		glEnd();
	}

	glLineWidth(1.5f);
	glBegin(GL_LINES);
	glVertex2f(verticalAnchor.x, boxMin.y);
	glVertex2f(verticalAnchor.x, boxMax.y);
	glEnd();

	if (zMagnitude > 1e-5f) {
		glLineWidth(2.5f);
		glBegin(GL_LINES);
		glVertex2f(verticalAnchor.x, verticalAnchor.y);
		glVertex2f(verticalTip.x, verticalTip.y);
		glEnd();

		glBegin(GL_TRIANGLES);
		glVertex2f(verticalTip.x, verticalTip.y);
		glVertex2f(verticalHeadBase.x + verticalHeadWidth, verticalHeadBase.y);
		glVertex2f(verticalHeadBase.x - verticalHeadWidth, verticalHeadBase.y);
		glEnd();
	}

	glLineWidth(1.0f);
	drawBitmapText(boxMin.x + 30.0f, boxMin.y - 14.0f, "Wind");
	drawBitmapText(verticalAnchor.x - 7.0f, boxMin.y - 14.0f, "Z");
}

static void drawPBDHangOverlay() {
	if (g_mode != SimMode::PBDHang || g_pbdSolver == nullptr) return;

	const GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, static_cast<double>(g_windowWidth), 0.0, static_cast<double>(g_windowHeight));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);

	std::ostringstream valueLine;
	valueLine << std::fixed << std::setprecision(3)
		<< "stretch=" << g_pbdSolver->getStructuralStiffness()
		<< "  shear=" << g_pbdSolver->getShearStiffness()
		<< "  bend=" << g_pbdSolver->getBendStiffness()
		<< "  damping=" << g_pbdSolver->getDampingFactor();

	std::ostringstream iterLine;
	iterLine << "iters=" << g_pbdSolver->getSolverIterations()
		<< "  state=" << (g_pbdHangRuntime.paused ? "paused" : "running");

	drawBitmapText(16.0f, g_windowHeight - 22.0f, "PBD hang tuning: 1/2 stretch  3/4 shear  5/6 bend  7/8 damping  R reset cloth  T reset params  P pause");
	drawBitmapText(16.0f, g_windowHeight - 40.0f, valueLine.str());
	drawBitmapText(16.0f, g_windowHeight - 58.0f, iterLine.str());

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	if (depthEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
}

static void drawPBDWindOverlay() {
	if (g_mode != SimMode::PBDHangWind || g_pbdSolver == nullptr) return;

	const GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, static_cast<double>(g_windowWidth), 0.0, static_cast<double>(g_windowHeight));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);

	const Eigen::Vector3f& rawDirection = g_pbdWindRuntime.currentDirectionInput;
	const Eigen::Vector3f& normalizedDirection = g_pbdWindRuntime.appliedWindDirection;
	std::ostringstream line1;
	line1 << std::fixed << std::setprecision(3)
		<< "PBD Wind Demo"
		<< "  speed=" << g_pbdWindRuntime.currentWindSpeed
		<< "  range=[0, 15]"
		<< "  state=" << (g_pbdWindRuntime.paused ? "paused" : "running");

	std::ostringstream line2;
	line2 << std::fixed << std::setprecision(3)
		<< "dir=(" << rawDirection.x() << ", " << rawDirection.y() << ", " << rawDirection.z() << ")"
		<< "  normalized=(" << normalizedDirection.x() << ", " << normalizedDirection.y() << ", " << normalizedDirection.z() << ")";

	drawBitmapText(16.0f, g_windowHeight - 22.0f, "Controls: 1/2 speed  3/4 dir-x  5/6 dir-y  7/8 dir-z  R reset cloth  T reset wind  P pause");
	drawBitmapText(16.0f, g_windowHeight - 40.0f, line1.str());
	drawBitmapText(16.0f, g_windowHeight - 58.0f, line2.str());
	drawWindDirectionIndicator(normalizedDirection);
	if (std::abs(normalizedDirection.z()) > PBDWindControlParam::verticalWarningThreshold) {
		drawBitmapText(16.0f, g_windowHeight - 76.0f, "Warning: wind has a vertical component. For flag-style motion, use z near 0.");
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	if (depthEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
}

static void drawPBDDropOverlay() {
	if (g_mode != SimMode::PBDDrop || g_pbdSolver == nullptr) return;

	const GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, static_cast<double>(g_windowWidth), 0.0, static_cast<double>(g_windowHeight));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);

	std::ostringstream valueLine;
	valueLine << std::fixed << std::setprecision(3)
		<< "radius=" << g_pbdDropRuntime.sphereRadius
		<< "  iters=" << g_pbdSolver->getSolverIterations()
		<< "  stretch=" << g_pbdSolver->getStructuralStiffness()
		<< "  shear=" << g_pbdSolver->getShearStiffness();

	std::ostringstream valueLine2;
	valueLine2 << std::fixed << std::setprecision(3)
		<< "bend=" << g_pbdSolver->getBendStiffness()
		<< "  damping=" << g_pbdSolver->getDampingFactor()
		<< "  mesh=" << g_pbdDropRuntime.currentMeshResolution << "x" << g_pbdDropRuntime.currentMeshResolution
		<< "  state=" << (g_pbdDropRuntime.paused ? "paused" : "running");

	drawBitmapText(16.0f, g_windowHeight - 22.0f, "PBD drop tuning: 1/2 stretch  3/4 shear  5/6 bend  7/8 damping  [/ ] pending mesh  R apply reset  T reset params  P pause");
	drawBitmapText(16.0f, g_windowHeight - 40.0f, valueLine.str());
	drawBitmapText(16.0f, g_windowHeight - 58.0f, valueLine2.str());
	if (g_pbdDropRuntime.pendingMeshResolution != g_pbdDropRuntime.currentMeshResolution) {
		std::ostringstream pendingLine;
		pendingLine << "Pending mesh: "
			<< g_pbdDropRuntime.pendingMeshResolution << "x" << g_pbdDropRuntime.pendingMeshResolution
			<< ", press R to apply";
		drawBitmapText(16.0f, g_windowHeight - 76.0f, pendingLine.str());
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	if (depthEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
}

static void drawPBDFloorOverlay() {
	if (g_mode != SimMode::PBDDropFloor || g_pbdSolver == nullptr) return;

	const GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, static_cast<double>(g_windowWidth), 0.0, static_cast<double>(g_windowHeight));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);

	std::ostringstream line1;
	line1 << std::fixed << std::setprecision(4)
		<< "PBD Drop-Floor Self-Collision Demo"
		<< "  iters=" << g_pbdFloorRuntime.solverIterations
		<< "  dt=" << g_pbdFloorRuntime.currentTimestep;
	if (!g_pbdFloorRuntime.customTimestep) {
		line1 << " (" << PBDFloorControlParam::speedPresetLabels[g_pbdFloorRuntime.speedPresetIndex] << ")";
	}

	std::ostringstream line2;
	line2 << std::fixed << std::setprecision(3)
		<< "self-k=" << g_pbdSolver->getSelfCollisionStiffness()
		<< "  max-contacts=" << g_pbdSolver->getMaxSelfCollisionContactsPerVertex()
		<< "  self-thickness=" << g_pbdSolver->getSelfCollisionThickness()
		<< "  floor-friction=" << g_pbdSolver->getPlaneFriction();

	std::ostringstream line3;
	line3 << std::fixed << std::setprecision(3)
		<< "bend=" << g_pbdSolver->getBendStiffness()
		<< "  damping=" << g_pbdSolver->getDampingFactor()
		<< "  mesh=" << g_pbdFloorRuntime.currentMeshResolution << "x" << g_pbdFloorRuntime.currentMeshResolution
		<< "  debug=" << (g_enableDebugDiagnostics ? "on" : "off")
		<< "  state=" << (g_pbdFloorRuntime.paused ? "paused" : "running");

	drawBitmapText(16.0f, g_windowHeight - 22.0f, line1.str());
	drawBitmapText(16.0f, g_windowHeight - 40.0f, "Controls: 1/2 self-k  3/4 max contacts  5/6 thickness  7/8 friction  Q/W bend  A/S damping  [/ ] mesh  9/0 speed  R reset  T defaults  P pause  D debug");
	drawBitmapText(16.0f, g_windowHeight - 58.0f, line2.str());
	drawBitmapText(16.0f, g_windowHeight - 76.0f, line3.str());
	if (g_pbdFloorRuntime.pendingMeshResolution != g_pbdFloorRuntime.currentMeshResolution) {
		std::ostringstream pendingLine;
		pendingLine << "Pending mesh: " << g_pbdFloorRuntime.pendingMeshResolution << "x" << g_pbdFloorRuntime.pendingMeshResolution << ", press R to apply";
		drawBitmapText(16.0f, g_windowHeight - 94.0f, pendingLine.str());
	}
	if (g_pbdFloorRuntime.speedPresetIndex > 0) {
		drawBitmapText(16.0f, g_windowHeight - 112.0f, "Warning: faster timestep may increase penetration, jitter, or missed self-collision");
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	if (depthEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
}

static void drawPBDDualOverlay() {
	if (g_mode != SimMode::PBDDropFloorDual || g_pbdSolver == nullptr) return;

	const GLboolean depthEnabled = glIsEnabled(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, static_cast<double>(g_windowWidth), 0.0, static_cast<double>(g_windowHeight));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(1.0f, 1.0f, 1.0f);

	std::ostringstream line1;
	line1 << std::fixed << std::setprecision(4)
		<< "PBD Dual-Obstacle Demo"
		<< "  iters=" << g_pbdDualRuntime.solverIterations
		<< "  dt=" << g_pbdDualRuntime.currentTimestep
		<< "  stretch=" << g_pbdSolver->getStructuralStiffness()
		<< "  shear=" << g_pbdSolver->getShearStiffness();
	if (!g_pbdDualRuntime.customTimestep) {
		line1 << " (" << PBDFloorControlParam::speedPresetLabels[g_pbdDualRuntime.speedPresetIndex] << ")";
	}

	std::ostringstream line2;
	line2 << std::fixed << std::setprecision(3)
		<< "bend=" << g_pbdSolver->getBendStiffness()
		<< "  mesh=" << g_pbdDualRuntime.currentMeshResolution << "x" << g_pbdDualRuntime.currentMeshResolution
		<< "  sphere=" << g_pbdDualRuntime.currentSphereRadius
		<< "  cube=" << g_pbdDualRuntime.currentCubeSize
		<< "  state=" << (g_pbdDualRuntime.paused ? "paused" : "running");

	drawBitmapText(16.0f, g_windowHeight - 22.0f, "Controls: 1/2 stretch  3/4 shear  5/6 bend  [/ ] mesh  Q/W sphere  A/S cube  9/0 speed  R reset cloth  T reset params  P pause");
	drawBitmapText(16.0f, g_windowHeight - 40.0f, line1.str());
	drawBitmapText(16.0f, g_windowHeight - 58.0f, line2.str());

	const bool hasPendingMesh = g_pbdDualRuntime.pendingMeshResolution != g_pbdDualRuntime.currentMeshResolution;
	const bool hasPendingSphere = std::abs(g_pbdDualRuntime.pendingSphereRadius - g_pbdDualRuntime.currentSphereRadius) > 1e-5f;
	const bool hasPendingCube = std::abs(g_pbdDualRuntime.pendingCubeSize - g_pbdDualRuntime.currentCubeSize) > 1e-5f;
	float y = g_windowHeight - 76.0f;
	if (hasPendingMesh) {
		std::ostringstream pendingLine;
		pendingLine << "Pending mesh: " << g_pbdDualRuntime.pendingMeshResolution << "x" << g_pbdDualRuntime.pendingMeshResolution;
		drawBitmapText(16.0f, y, pendingLine.str());
		y -= 18.0f;
	}
	if (hasPendingSphere) {
		std::ostringstream pendingLine;
		pendingLine << std::fixed << std::setprecision(3) << "Pending sphere radius: " << g_pbdDualRuntime.pendingSphereRadius;
		drawBitmapText(16.0f, y, pendingLine.str());
		y -= 18.0f;
	}
	if (hasPendingCube) {
		std::ostringstream pendingLine;
		pendingLine << std::fixed << std::setprecision(3) << "Pending cube size: " << g_pbdDualRuntime.pendingCubeSize;
		drawBitmapText(16.0f, y, pendingLine.str());
		y -= 18.0f;
	}
	if (hasPendingMesh || hasPendingSphere || hasPendingCube) {
		drawBitmapText(16.0f, y, "Pending changes: press R to apply");
		y -= 18.0f;
	}
	if (g_pbdDualRuntime.speedPresetIndex > 0) {
		drawBitmapText(16.0f, y, "Warning: faster timestep may increase penetration, jitter, or missed self-collision");
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	if (depthEnabled) {
		glEnable(GL_DEPTH_TEST);
	}
}

static void animateCloth(int value) {
	if (isPBDMode()) {
		const bool paused = (g_mode == SimMode::PBDHang && g_pbdHangRuntime.paused)
			|| (g_mode == SimMode::PBDHangWind && g_pbdWindRuntime.paused)
			|| (g_mode == SimMode::PBDDrop && g_pbdDropRuntime.paused)
			|| (g_mode == SimMode::PBDDropFloor && g_pbdFloorRuntime.paused)
			|| (g_mode == SimMode::PBDDropFloorDual && g_pbdDualRuntime.paused);
		if (!paused) {
			const unsigned int iterationCount = (isFloorDemo() || g_mode == SimMode::PBDHang || g_mode == SimMode::PBDDrop)
				? g_pbdSolver->getSolverIterations()
				: static_cast<unsigned int>(PBDSystemParam::n_iter);
			g_pbdSolver->solve(iterationCount);
			++g_pbdFrameCounter;
			if (g_enableDebugDiagnostics) {
				logPBDSelfCollisionDiagnostics();
			}
		}
	}
	else {
		g_solver->solve(g_iter);
		g_solver->solve(g_iter);

		CgSatisfyVisitor visitor;
		visitor.satisfy(*g_cgRootNode);
	}

	// update normals
	g_clothMesh->request_face_normals();
	g_clothMesh->update_normals();
	g_clothMesh->release_face_normals();

	// update target
	updateRenderTarget();

	// redisplay
	glutPostRedisplay();

	// reset timer
	glutTimerFunc(g_animation_timer, animateCloth, 0);
}

static void logPBDSelfCollisionDiagnostics() {
	if (g_pbdSolver == nullptr) return;
	if (PBDDebugParam::debugPrintPeriod == 0u) return;
	if ((g_pbdFrameCounter % PBDDebugParam::debugPrintPeriod) != 0u) return;

	const SelfCollisionDebugStats& stats = g_pbdSolver->getSelfCollisionDebugStats();
	const char* modeLabel =
		(g_mode == SimMode::PBDDropFloorDual)
			? "pbd drop-floor-dual"
			: (isFloorDemo()
				? "pbd drop-floor"
				: (g_mode == SimMode::PBDDrop ? "pbd drop" : (g_mode == SimMode::PBDHangWind ? "pbd hang_wind" : "pbd hang")));
	std::cout
		<< "[" << modeLabel << " self-collision] frame=" << g_pbdFrameCounter
		<< " generated=" << stats.generatedContacts
		<< " initial-violations=" << stats.initiallyViolatedContacts
		<< " remaining-violations=" << stats.remainingViolatedContacts
		<< " max-pen-before=" << stats.maxInitialPenetration
		<< " max-pen-after=" << stats.maxRemainingPenetration
		<< std::endl;
}

// S C E N E  U P D A T E ///////////////////////////////////////////////////////////
static void updateProjection() {
	g_ProjectionMatrix = glm::perspective(PI / 4.0f,
		g_windowWidth * 1.0f / g_windowHeight, 0.01f, 1000.0f);
}

static void updateRenderTarget() {
	// update vertex positions
	g_render_target->setPositionData(g_clothMesh->vbuff(), g_clothMesh->vbuffLen());

	// update vertex normals
	g_render_target->setNormalData(g_clothMesh->nbuff(), g_clothMesh->vbuffLen());

}

// C L E A N  U P //////////////////////////////////////////////////////////////////
static void cleanUp() {
	// delete mesh
	delete g_clothMesh;

	// delete UI
	delete g_pickRenderer;
	delete UI;

	// delete render target
	delete g_render_target;
	delete g_floor_target;
	delete g_sphere_target;
	delete g_cube_target;

	// delete mass-spring system
	delete g_system;
	delete g_solver;
	delete g_pbdSystem;
	delete g_pbdSolver;
	delete g_shadowShader;

	// delete constraint graph
	// TODO
}

// E R R O R S /////////////////////////////////////////////////////////////////////
void checkGlErrors() {
	const GLenum errCode = glGetError();

	if (errCode != GL_NO_ERROR) {
		std::string error("GL Error: ");
		error += reinterpret_cast<const char*>(gluErrorString(errCode));
		std::cerr << error << std::endl;
		throw std::runtime_error(error);
	}
}