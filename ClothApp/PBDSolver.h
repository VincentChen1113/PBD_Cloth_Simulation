#pragma once
#include "MassSpringSolver.h"
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// -----------------------------
// Generic PBD constraint system
// -----------------------------
// In Muller et al., each constraint j is described by:
// - a cardinality n_j
// - a scalar constraint function C_j(.)
// - particle indices i_1 ... i_n
// - a stiffness k in [0, 1]
// - an equality or inequality type
//
// For a course project, a polymorphic base class is a good fit because each
// constraint naturally owns its own C(.) and gradient code. That keeps the
// solver loop close to the paper and avoids a large switch statement.
enum class PBDConstraintType {
	Equality,
	Inequality
};

class PBDConstraint {
protected:
	typedef Eigen::Vector3f Vector3f;

	std::vector<unsigned int> particleIndices; // involved particles {i_1, ..., i_n}
	float stiffnessValue;                      // k in [0, 1]
	PBDConstraintType constraintType;          // equality or inequality

public:
	PBDConstraint(
		std::vector<unsigned int> particleIndices,
		float stiffnessValue,
		PBDConstraintType constraintType
	);
	virtual ~PBDConstraint() = default;

	std::size_t cardinality() const;
	const std::vector<unsigned int>& indices() const;
	float stiffness() const;
	void setStiffness(float stiffness);
	PBDConstraintType type() const;

	// Evaluate the scalar constraint C(p).
	virtual float evaluate(const std::vector<Vector3f>& positions) const = 0;

	// Return gradients [grad_{p_i1} C, ..., grad_{p_in} C].
	virtual void gradients(
		const std::vector<Vector3f>& positions,
		std::vector<Vector3f>& outGradients
	) const = 0;

	// Equality constraints project toward C = 0.
	// Inequality constraints project only when violated, i.e. C < 0.
	virtual bool isViolated(const std::vector<Vector3f>& positions, float epsilon) const;

	// Generic PBD projection step:
	//   s = C / sum_j w_j * |grad_{p_j} C|^2
	//   Delta p_i = -k' * s * w_i * grad_{p_i} C
	virtual void project(
		std::vector<Vector3f>& positions,
		const std::vector<float>& invMass,
		unsigned int solverIterations,
		float epsilon
	) const;
};

// Distance constraint:
//   C(p1, p2) = |p1 - p2| - d
class DistanceConstraint : public PBDConstraint {
private:
	float restLength;

public:
	DistanceConstraint(unsigned int i, unsigned int j, float restLength, float stiffness);

	virtual float evaluate(const std::vector<Vector3f>& positions) const override;
	virtual void gradients(
		const std::vector<Vector3f>& positions,
		std::vector<Vector3f>& outGradients
	) const override;
};

// Dihedral bending constraint:
//   C_bend = theta - theta_0
// where theta is the dihedral angle between the two triangles incident to a
// shared cloth edge. Unlike a distance-based bend proxy, this directly
// measures folding and is therefore much less coupled to stretching.
class DihedralBendConstraint : public PBDConstraint {
private:
	float restAngle;

public:
	DihedralBendConstraint(
		unsigned int edge0,
		unsigned int edge1,
		unsigned int opposite0,
		unsigned int opposite1,
		float restAngle,
		float stiffness
	);

	float angle(const std::vector<Vector3f>& positions, bool* valid = nullptr) const;

	virtual float evaluate(const std::vector<Vector3f>& positions) const override;
	virtual void gradients(
		const std::vector<Vector3f>& positions,
		std::vector<Vector3f>& outGradients
	) const override;
};

// Fixed point constraint:
//   C(p_i) = p_i - p_fixed
// This is vector-valued in full generality, so for a practical cloth solver we
// keep one object and project it directly to the target position.
class FixedPointConstraint : public PBDConstraint {
private:
	Eigen::Vector3f fixedPosition;

public:
	FixedPointConstraint(unsigned int i, const Eigen::Vector3f& fixedPosition, float stiffness = 1.0f);

	void setFixedPosition(const Eigen::Vector3f& position);
	const Eigen::Vector3f& position() const;

	virtual float evaluate(const std::vector<Vector3f>& positions) const override;
	virtual void gradients(
		const std::vector<Vector3f>& positions,
		std::vector<Vector3f>& outGradients
	) const override;
	virtual void project(
		std::vector<Vector3f>& positions,
		const std::vector<float>& invMass,
		unsigned int solverIterations,
		float epsilon
	) const override;
};

struct SphereCollider {
	Eigen::Vector3f center;
	float radius;
};

struct PlaneCollider {
	Eigen::Vector3f point;
	Eigen::Vector3f normal;
};

// Generated collision constraint:
// - sphere contact:          C(p_i) = |p_i - c| - r
// - plane/static contact:    C(p_i) = (p_i - q_c) . n_c
// - self vertex-triangle:    C(q, p1, p2, p3) = (q - p1) . n - h
// Both are inequality constraints and are satisfied when C >= 0.
class CollisionConstraint : public PBDConstraint {
private:
	enum class CollisionKind {
		Sphere,
		Plane,
		SelfVertexTriangle
	};

	CollisionKind collisionKind;
	Eigen::Vector3f center;
	float radius;
	float offset;
	Eigen::Vector3f planeNormal;
	Eigen::Vector3f selfCollisionNormal;
	Eigen::Vector3f selfCollisionBarycentric;

	bool selfCollisionGeometry(
		const std::vector<Vector3f>& positions,
		Eigen::Vector3f& outNormal,
		Eigen::Vector3f& outBarycentric
	) const;

public:
	CollisionConstraint(unsigned int i, const Eigen::Vector3f& center, float radius, float stiffness = 1.0f);
	CollisionConstraint(
		unsigned int i,
		const Eigen::Vector3f& planePoint,
		const Eigen::Vector3f& planeNormal,
		float stiffness = 1.0f
	);
	CollisionConstraint(
		unsigned int vertex,
		unsigned int p1,
		unsigned int p2,
		unsigned int p3,
		float thickness,
		const Eigen::Vector3f& normal,
		const Eigen::Vector3f& barycentric,
		float stiffness = 1.0f
	);

	virtual float evaluate(const std::vector<Vector3f>& positions) const override;
	virtual void gradients(
		const std::vector<Vector3f>& positions,
		std::vector<Vector3f>& outGradients
	) const override;
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
	std::vector<unsigned int> triangle_indices;
};

struct SelfCollisionDebugStats {
	unsigned int generatedContacts = 0u;
	unsigned int initiallyViolatedContacts = 0u;
	unsigned int remainingViolatedContacts = 0u;
	float maxInitialPenetration = 0.0f;
	float maxRemainingPenetration = 0.0f;
};

// -----------------------------
// PBD Solver
// -----------------------------
class PBDSolver : public FixedPointController {
private:
	typedef Eigen::Vector3f Vector3f;     // 3D vector type
	typedef std::pair<unsigned int, unsigned int> Edge;
	typedef std::unique_ptr<PBDConstraint> ConstraintPtr;
	typedef std::vector<ConstraintPtr> ConstraintList;

	// system / render buffer
	pbd_system* system;                   // pointer to PBD system
	float* vbuff;

	// particle state
	std::vector<Vector3f> restX;          // reference positions used to preserve the original cloth side in self-collision
	std::vector<Vector3f> x;              // current positions x_i
	std::vector<Vector3f> p;              // predicted positions p_i
	std::vector<Vector3f> v;              // velocities v_i
	std::vector<float> invMass;           // inverse mass w_i = 1 / m_i

	// Persistent constraints are part of the cloth model and exist every frame.
	ConstraintList persistentConstraints;
	std::unordered_map<unsigned int, FixedPointConstraint*> fixedPointConstraints;
	std::vector<PBDConstraint*> structuralConstraints;
	std::vector<PBDConstraint*> shearConstraints;
	std::vector<PBDConstraint*> bendConstraints;

	// Collision primitives persist, but actual collision constraints are generated
	// fresh each step from the predicted positions x -> p.
	std::vector<SphereCollider> sphereColliders;
	std::vector<PlaneCollider> planeColliders;
	ConstraintList generatedCollisionConstraints;
	std::vector<CollisionConstraint*> generatedSelfCollisionConstraints;
	std::vector<Vector3f> planeContactPoints;
	std::vector<Vector3f> planeContactNormals;
	std::vector<float> planeContactSignedDistances;
	std::vector<std::unordered_set<unsigned int>> meshAdjacency;
	SelfCollisionDebugStats selfCollisionDebugStats;

	// simulation parameters
	unsigned int solverIterations;
	float dampingFactor;
	float collisionEps;
	float structuralStiffness;
	float shearStiffness;
	float bendStiffness;
	float planeFriction;
	float selfCollisionThickness;
	float selfCollisionStiffness;
	float selfCollisionCellSize;
	unsigned int maxSelfCollisionContactsPerVertex;
	float velocitySleepThreshold;
	Vector3f gravity;

	// -----------------------------
	// 1. Constructor / Initialization
	// -----------------------------
	void initializeState();

	// -----------------------------
	// 3. Core PBD Pipeline Stages
	// -----------------------------
	void applyExternalForces(float dt);
	void dampVelocities();
	void predictPositions(float dt);
	void projectConstraints(const ConstraintList& constraints);
	void updateVelocities(float dt);

	// -----------------------------
	// 5. Collision Handling
	// -----------------------------
	void generateCollisionConstraints();
	void applyPlaneContactVelocityDamping();

	// -----------------------------
	// 6. Self-Collision Utilities / Helpers
	// -----------------------------
	void generateSelfCollisionConstraints();

	// -----------------------------
	// 4. Persistent Constraint Setup / Constraint Construction
	// -----------------------------
	void addDistanceConstraints(
		const std::vector<unsigned int>& indices,
		float stiffness,
		std::vector<PBDConstraint*>* constraintGroup = nullptr
	);
	void addDihedralBendConstraints(float stiffness);
	void setConstraintGroupStiffness(const std::vector<PBDConstraint*>& constraints, float stiffness);

	// -----------------------------
	// 8. Debug / Diagnostics
	// -----------------------------
	void updateSelfCollisionDebugStats(const std::vector<Vector3f>& positions, bool afterProjection);

	// -----------------------------
	// 9. Rendering / Buffer Synchronization
	// -----------------------------
	void writeBackToVBuff();
	void commitPositions();

public:
	// -----------------------------
	// 1. Constructor / Initialization
	// -----------------------------
	PBDSolver(pbd_system* system, float* vbuff);

	// -----------------------------
	// 2. Main Simulation Entry / Solver Loop
	// -----------------------------
	void step(float dt);
	void solve(unsigned int n);

	// -----------------------------
	// 4. Persistent Constraint Setup / Constraint Construction
	// -----------------------------
	void pinPoint(unsigned int i);
	virtual void fixPoint(unsigned int i) override;
	virtual void releasePoint(unsigned int i) override;
	void addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);
	void addShearConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);
	void addBendConstraints(const std::vector<unsigned int>& indices, float stiffness = 1.0f);

	// -----------------------------
	// 5. Collision Handling
	// -----------------------------
	void addSphereCollider(const Vector3f& center, float radius);
	void addPlaneCollider(const Vector3f& point, const Vector3f& normal);

	// -----------------------------
	// 7. Parameter / Tuning Interface
	// -----------------------------
	void setGravity(float gravityMagnitude);
	float getGravity() const;
	void setDampingFactor(float damping);
	float getDampingFactor() const;
	void setPlaneFriction(float friction);
	float getPlaneFriction() const;
	void setSolverIterations(unsigned int iterations);
	unsigned int getSolverIterations() const;
	void setVelocitySleepThreshold(float threshold);
	float getVelocitySleepThreshold() const;
	void setSelfCollisionThickness(float thickness) {
		if (thickness <= 0.0f) return;
		selfCollisionThickness = std::max(thickness, collisionEps);
	}
	void setStructuralStiffness(float stiffness);
	float getStructuralStiffness() const;
	void setShearStiffness(float stiffness);
	float getShearStiffness() const;
	void setBendStiffness(float stiffness);
	float getBendStiffness() const;
	void setSelfCollisionStiffness(float stiffness);
	float getSelfCollisionStiffness() const;
	float getSelfCollisionThickness() const;
	void setMaxSelfCollisionContactsPerVertex(unsigned int maxContacts);
	unsigned int getMaxSelfCollisionContactsPerVertex() const;

	// -----------------------------
	// 8. Debug / Diagnostics
	// -----------------------------
	const SelfCollisionDebugStats& getSelfCollisionDebugStats() const { return selfCollisionDebugStats; }

	// -----------------------------
	// 9. Rendering / Buffer Synchronization
	// -----------------------------
	// accessors
	std::vector<Vector3f>& getPositions() { return x; }
	std::vector<Vector3f>& getPredictedPositions() { return p; }
	std::vector<Vector3f>& getVelocities() { return v; }
};