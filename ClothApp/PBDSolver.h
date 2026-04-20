#pragma once
#include "MassSpringSolver.h"
#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
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

// Sphere collision constraint:
//   C(p_i) = |p_i - c| - r
// This is an inequality constraint and is satisfied when C >= 0.
class SphereCollisionConstraint : public PBDConstraint {
private:
	Eigen::Vector3f center;
	float radius;

public:
	SphereCollisionConstraint(unsigned int i, const Eigen::Vector3f& center, float radius, float stiffness = 1.0f);

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
	std::vector<Vector3f> x;              // current positions x_i
	std::vector<Vector3f> p;              // predicted positions p_i
	std::vector<Vector3f> v;              // velocities v_i
	std::vector<float> invMass;           // inverse mass w_i = 1 / m_i

	// Persistent constraints are part of the cloth model and exist every frame.
	ConstraintList persistentConstraints;
	std::unordered_map<unsigned int, FixedPointConstraint*> fixedPointConstraints;

	// Collision primitives persist, but actual collision constraints are generated
	// fresh each step from the predicted positions x -> p.
	std::vector<SphereCollider> sphereColliders;
	ConstraintList generatedCollisionConstraints;

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
	void generateCollisionConstraints();
	void projectConstraints(const ConstraintList& constraints);

	void updateVelocities(float dt);
	void commitPositions();

	void addDistanceConstraints(const std::vector<unsigned int>& indices, float stiffness);

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