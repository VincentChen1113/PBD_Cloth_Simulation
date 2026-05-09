#include <GL/glew.h>
#include <GL/glut.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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
static const float g_mouse_drag_tolerance_scale = 0.75f;
static const float g_mouse_drag_tolerance_min = 0.6f;

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
static unsigned int g_pbdHangIterationsOverride = 0u;

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

namespace PBDWindCliParam {
	static const float minValue = 0.0f; // Safe lower bound for user-provided --wind-speed.
	static const float maxValue = 15.0f; // Safe upper bound for user-provided --wind-speed.
	static const glm::vec3 windDirection(1.0f, 0.0f, 0.0f); // wind direction
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
static glm::vec3 dragClampTolerance();
static void configurePBDHangSolver(float stretch, float shear, float bend, float damping, unsigned int iterations, bool captureDefaults);
static void resetPBDHangDemo(bool resetParameters);
static void logPBDHangControlState(const std::string& reason);
static void drawBitmapText(float x, float y, const std::string& text);
static void drawPBDHangOverlay();
static glm::mat4 floorShadowMatrix(float planeHeight, const glm::vec3& lightDirection);
static void orientClothForFloorDrop();
static void orientClothFlatForDualFloorDrop();
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
	static const float sphereRadius = 0.30f;
	static const AnalyticBoxDefinition cube = {
		glm::vec3(-0.42f, 0.0f, g_floor_collision_height + 0.44f),
		glm::vec3(0.44f, 0.44f, 0.44f)
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
		"Usage: ./fast-mass-spring [mass-spring|ms] [hang|drop] [--self-thickness value] [--debug], ./fast-mass-spring pbd [hang|hang-wind|drop|drop-floor|drop-floor-dual] [--self-thickness value] [--debug] [--wind-speed value], or ./fast-mass-spring [ms-hang|ms-drop|pbd-hang|pbd-hang-wind|pbd-drop|pbd-drop-floor|pbd-drop-floor-dual] [--self-thickness value] [--debug] [--wind-speed value]"
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

		if (arg == "--iters") {
			if (g_mode != SimMode::PBDHang) {
				throw std::runtime_error("--iters is only valid for the pbd hang demo");
			}
			if (i + 1 >= argc) {
				throw std::runtime_error("Missing value after --iters");
			}
			if (g_pbdHangIterationsOverride != 0u) {
				throw std::runtime_error("Specify --iters only once");
			}

			std::stringstream valueStream(argv[++i]);
			int iterations = 0;
			valueStream >> iterations;
			// --iters controls the number of PBD projection passes per timestep
			// for the hang demo only, so keep it within a safe startup range.
			if (!valueStream || !valueStream.eof()
				|| iterations < static_cast<int>(PBDHangCliParam::minIterations)
				|| iterations > static_cast<int>(PBDHangCliParam::maxIterations)) {
				throw std::runtime_error("--iters expects an integer value in [1, 80]");
			}

			g_pbdHangIterationsOverride = static_cast<unsigned int>(iterations);
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

static glm::vec3 dragClampTolerance() {
	const float tolerance = std::max(
		g_mouse_drag_tolerance_min,
		g_mouse_drag_tolerance_scale * activeClothWidth()
	);
	return glm::vec3(tolerance);
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

static void orientClothFlatForDualFloorDrop() {
	if (g_clothMesh == nullptr || g_render_target == nullptr) return;

	const float sphereTop = PBDDualObstacleDemoParam::sphereCenter.z + PBDDualObstacleDemoParam::sphereRadius;
	const float cubeTop = PBDDualObstacleDemoParam::cube.center.z + PBDDualObstacleDemoParam::cube.halfExtents.z;
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

	if (g_mode != SimMode::PBDHangWind) {
		return;
	}

	float* const positions = g_clothMesh->vbuff();
	const unsigned int vertexCount = g_clothMesh->n_vertices();
	if (positions != nullptr && vertexCount > 0u) {
		glm::vec3 minBounds(positions[0], positions[1], positions[2]);
		glm::vec3 maxBounds = minBounds;
		for (unsigned int vertex = 1u; vertex < vertexCount; ++vertex) {
			const glm::vec3 position(
				positions[3 * vertex + 0],
				positions[3 * vertex + 1],
				positions[3 * vertex + 2]
			);
			minBounds = glm::min(minBounds, position);
			maxBounds = glm::max(maxBounds, position);
		}

		const glm::vec3 tolerance = dragClampTolerance();
		UI->setDragBounds(minBounds - tolerance, maxBounds + tolerance);
	}
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
	const unsigned int n = PBDSystemParam::n;
	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);
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
	PBDWindConfig windConfig;
	windConfig.inputMode = PBDWindInputMode::Speed;
	windConfig.windDirection = Eigen::Vector3f(
		PBDWindCliParam::windDirection.x,
		PBDWindCliParam::windDirection.y,
		PBDWindCliParam::windDirection.z
	);
	windConfig.baseSpeed = g_windSpeed;
	windConfig.baseAcceleration = 0.0f;
	const float gustReference = g_windSpeed;
	windConfig.gustAmplitude = PBDWindCliParam::gustFraction * gustReference;
	windConfig.gustFrequency = PBDWindCliParam::gustFrequency;
	windConfig.noiseStrength = PBDWindCliParam::noiseFraction * gustReference;
	windConfig.dragCoefficient = PBDWindCliParam::dragCoefficient;
	windConfig.liftCoefficient = PBDWindCliParam::liftCoefficient;
	windConfig.airDensity = PBDWindCliParam::airDensity;
	windConfig.maxWindSpeed = PBDWindCliParam::maxValue;
	g_pbdSolver->setWindConfig(windConfig);
	g_pbdSolver->setDampingFactor(PBDWindCliParam::dampingFactor);
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), PBDWindCliParam::bendStiffness);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);
	for (unsigned int row = 0; row < n; ++row) {
		g_pbdSolver->pinPoint(row * n);
	}
	std::cout
		<< "[pbd hang-wind] "
		<< "wind-speed="
		<< g_windSpeed
		<< ", direction=(" << PBDWindCliParam::windDirection.x << ", "
		<< PBDWindCliParam::windDirection.y << ", "
		<< PBDWindCliParam::windDirection.z << ")"
		<< ", gust-amplitude=" << windConfig.gustAmplitude
		<< ", gust-frequency=" << windConfig.gustFrequency
		<< ", noise-strength=" << windConfig.noiseStrength
		<< ", Cd=" << windConfig.dragCoefficient
		<< ", Cl=" << windConfig.liftCoefficient
		<< ", rho=" << windConfig.airDensity
		<< std::endl;
	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, n);
}

static void demo_pbd_drop() {
	const unsigned int n = PBDSystemParam::n;
	const glm::vec3 sphereCenter(0.0f, 0.0f, -1.0f);
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

	// build PBD system from mass-spring system, then initialize PBD solver and constraints
	mass_spring_system* temp = builder.getResult();
	g_pbdSystem = buildPBDSystem(*temp);
	delete temp;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), PBDSystemParam::k_bend);
	g_pbdSolver->addSphereCollider(Eigen::Vector3f(sphereCenter.x, sphereCenter.y, sphereCenter.z), PBDSystemParam::sphere_radius);
	initSphereColliderVisual(PBDSystemParam::sphere_radius, sphereCenter);
	initMouseInteraction(g_pbdSolver, n);
}

static void demo_pbd_drop_floor() {
	const unsigned int n = PBDSystemParam::n;
	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);
	orientClothForFloorDrop();
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
	g_pbdSystem->time_step = PBDFloorDemoParam::h;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	g_pbdSolver->setSolverIterations(PBDFloorDemoParam::n_iter);
	g_pbdSolver->setSelfCollisionStiffness(PBDFloorDemoParam::selfCollisionStiffness);
	g_pbdSolver->setMaxSelfCollisionContactsPerVertex(PBDFloorDemoParam::maxSelfCollisionContactsPerVertex);
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), PBDSystemParam::k_bend);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);
	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, n);
}

static void demo_pbd_drop_floor_dual() {
	const unsigned int n = PBDSystemParam::n;
	const Eigen::Vector3f floorPoint(0.0f, 0.0f, g_floor_collision_height);
	const Eigen::Vector3f floorNormal(0.0f, 0.0f, 1.0f);
	orientClothFlatForDualFloorDrop();
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
	g_pbdSystem->time_step = PBDFloorDemoParam::h;
	g_pbdSolver = new PBDSolver(g_pbdSystem, g_clothMesh->vbuff());
	g_pbdSolver->setSolverIterations(PBDFloorDemoParam::n_iter);
	g_pbdSolver->setSelfCollisionStiffness(PBDFloorDemoParam::selfCollisionStiffness);
	g_pbdSolver->setMaxSelfCollisionContactsPerVertex(PBDFloorDemoParam::maxSelfCollisionContactsPerVertex);
	if (g_selfCollisionThicknessOverride > 0.0f) {
		g_pbdSolver->setSelfCollisionThickness(g_selfCollisionThicknessOverride);
	}
	g_pbdSolver->addStructuralConstraints(builder.getStructIndex(), PBDSystemParam::k_stretch);
	g_pbdSolver->addShearConstraints(builder.getShearIndex(), PBDSystemParam::k_shear);
	g_pbdSolver->addBendConstraints(builder.getBendIndex(), PBDSystemParam::k_bend);
	g_pbdSolver->addPlaneCollider(floorPoint, floorNormal);
	g_pbdSolver->addSphereCollider(
		Eigen::Vector3f(
			PBDDualObstacleDemoParam::sphereCenter.x,
			PBDDualObstacleDemoParam::sphereCenter.y,
			PBDDualObstacleDemoParam::sphereCenter.z
		),
		PBDDualObstacleDemoParam::sphereRadius
	);
	g_pbdSolver->addBoxCollider(
		Eigen::Vector3f(
			PBDDualObstacleDemoParam::cube.center.x,
			PBDDualObstacleDemoParam::cube.center.y,
			PBDDualObstacleDemoParam::cube.center.z
		),
		Eigen::Vector3f(
			PBDDualObstacleDemoParam::cube.halfExtents.x,
			PBDDualObstacleDemoParam::cube.halfExtents.y,
			PBDDualObstacleDemoParam::cube.halfExtents.z
		)
	);
	initSphereColliderVisual(PBDDualObstacleDemoParam::sphereRadius, PBDDualObstacleDemoParam::sphereCenter);
	// Render and collision use the same shared analytic box definition.
	initCubeColliderVisual(PBDDualObstacleDemoParam::cube.center, PBDDualObstacleDemoParam::cube.halfExtents);
	g_pbdFrameCounter = 0u;
	initMouseInteraction(g_pbdSolver, n);
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
	if (g_mode != SimMode::PBDHang || g_pbdSolver == nullptr) return;

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

static void animateCloth(int value) {
	if (isPBDMode()) {
		if (!(g_mode == SimMode::PBDHang && g_pbdHangRuntime.paused)) {
			const unsigned int iterationCount = (isFloorDemo() || g_mode == SimMode::PBDHang)
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