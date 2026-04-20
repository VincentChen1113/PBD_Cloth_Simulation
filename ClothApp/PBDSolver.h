#pragma once
#include "MassSpringSolver.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_set>

// -----------------------------
// Simple constraint data types for prototyping PBD solver
// -----------------------------
struct DistanceConstraint {
	unsigned int i;       // particle index 1
	unsigned int j;       // particle index 2
	float rest_length;    // target distance
	float stiffness;      // [0, 1], optional scaling
};

struct FixedPointConstraint {
	unsigned int i;             // fixed particle index
	Eigen::Vector3f fixedPosition;   // target fixed position
};

struct SphereCollider {
	Eigen::Vector3f center;
	float radius;
};

// -----------------------------
// PBD system struct
// -----------------------------
struct pbd_system {
    typedef std::pair<unsigned int, unsigned int> Edge;
    typedef std::vector<Edge> EdgeList;

    unsigned int n_points;
    unsigned int n_constraints;
    float time_step;

    EdgeList spring_list;
    Eigen::VectorXf rest_lengths;
    Eigen::VectorXf masses;
};

// -----------------------------
// PBD Solver
// -----------------------------
class PBDSolver : public FixedPointController {
private:
	typedef Eigen::Vector3f Vector3f;   // 3D vector type
	typedef Eigen::VectorXf VectorXf;   // dynamic float vector type
	typedef std::pair<unsigned int, unsigned int> Edge; // edge type for spring constraints

	// system / render buffer
	pbd_system* system;     // pointer to PBD system
	float* vbuff;

	// particle state
	std::vector<Vector3f> x;        // current positions
	std::vector<Vector3f> p;        // predicted positions
	std::vector<Vector3f> v;        // velocities
	std::vector<float> invMass;     // inverse mass w = 1 / m

	// constraints
	std::vector<DistanceConstraint> structuralConstraints;      // equality
	std::vector<DistanceConstraint> shearConstraints;           // equality
	std::vector<DistanceConstraint> bendConstraints;            // equality
	std::vector<FixedPointConstraint> fixedConstraints;         // equality
	std::vector<SphereCollider> sphereColliders;                // inequality

	// simulation parameters
    unsigned int solverIterations;
	float dampingFactor;
	float collisionEps;
	Vector3f gravity;

	// internal steps
	void initializeState();
	void writeBackToVBuff();

	void applyExternalForces(float dt);
	void dampVelocities();
	void predictPositions(float dt);

	void projectFixedConstraints();
	void projectDistanceConstraints(const std::vector<DistanceConstraint>& constraints);
	void projectSphereCollisions();

	void updateVelocities(float dt);
	void commitPositions();

public:
	PBDSolver(pbd_system* system, float* vbuff);

	// one simulation step
	void step(float dt);

	// optional compatibility wrapper
	void solve(unsigned int n);

	// setup helpers
	void pinPoint(unsigned int i);
	virtual void fixPoint(unsigned int i) override;
	virtual void releasePoint(unsigned int i) override;
	void addSphereCollider(const Vector3f& center, float radius);

	// build constraint lists from builder indices
	void addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);
	void addShearConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);
	void addBendConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);

	// accessors
	std::vector<Vector3f>& getPositions() { return x; }
	std::vector<Vector3f>& getPredictedPositions() { return p; }
	std::vector<Vector3f>& getVelocities() { return v; }
};