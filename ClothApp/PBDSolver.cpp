#include "PBDSolver.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>

namespace PBDDefaultParam {
	static const unsigned int solverIterations = 10;
	static const float dampingFactor = 0.01f;
	static const float collisionEps = 1e-4f;
	static const float collisionStiffness = 1.0f;
	static const Eigen::Vector3f gravity(0.0f, 0.0f, -9.8f);
}

namespace {
// Apply the iteration-corrected stiffness
//   k' = 1 - (1 - k)^(1 / n_s)
// so that the effective stiffness stays consistent when the number of solver
// iterations n_s changes.
float correctedStiffness(float stiffness, unsigned int solverIterations) {
	if (stiffness <= 0.0f) return 0.0f;
	if (stiffness >= 1.0f) return 1.0f;
	if (solverIterations == 0u) return stiffness;
	return 1.0f - std::pow(1.0f - stiffness, 1.0f / static_cast<float>(solverIterations));
}

struct SharedEdgeKey {
	unsigned int a;
	unsigned int b;

	bool operator==(const SharedEdgeKey& other) const {
		return a == other.a && b == other.b;
	}
};

struct SharedEdgeKeyHash {
	std::size_t operator()(const SharedEdgeKey& key) const {
		return (static_cast<std::size_t>(key.a) << 32) ^ static_cast<std::size_t>(key.b);
	}
};

struct PendingTriangleEdge {
	unsigned int edge0;
	unsigned int edge1;
	unsigned int opposite;
};

float dihedralAngleFromPositions(
	/*
	C_bend(p1, p2, p3, p4) = 
			acos( 
				( (p2 - p1) x (p3 - p1) ) . ( (p2 - p1) x (p4 - p1) ) 
									/ 
				( |(p2 - p1) x (p3 - p1)| * |(p2 - p1) x (p4 - p1)| ) 
			)
	*/
	const std::vector<Eigen::Vector3f>& positions,
	unsigned int edge0,
	unsigned int edge1,
	unsigned int opposite0,
	unsigned int opposite1,
	bool* valid = nullptr
) {
	if (valid != nullptr) *valid = false;
	if (edge0 >= positions.size() || edge1 >= positions.size()
		|| opposite0 >= positions.size() || opposite1 >= positions.size()) {
		return 0.0f;
	}

	const Eigen::Vector3f& p1 = positions[edge0];
	const Eigen::Vector3f& p2 = positions[edge1];
	const Eigen::Vector3f& p3 = positions[opposite0];
	const Eigen::Vector3f& p4 = positions[opposite1];

	// Formulate bending on the two triangles adjacent to a shared edge. 
	// (p1, p2) is the shared edge, and p3/p4 are the opposite vertices. 
	const Eigen::Vector3f normal1 = (p2 - p1).cross(p3 - p1);
	const Eigen::Vector3f normal2 = (p2 - p1).cross(p4 - p1);
	const float area1 = normal1.norm();
	const float area2 = normal2.norm();

	// Degenerate triangles have undefined normals and therefore no meaningful dihedral angle. 
	// In this case we skip evaluation/projection for robustness.
	if (area1 <= 1e-8f || area2 <= 1e-8f) {
		return 0.0f;
	}

	// The bend constraint uses theta = acos(n1 . n2). Clamp the dot product to
	// stay in the valid acos domain under floating-point roundoff.
	const float normalizedDot = normal1.dot(normal2) / (area1 * area2);
	const float dotValue = std::max(-1.0f, std::min(1.0f, normalizedDot));
	if (valid != nullptr) *valid = true;
	return std::acos(dotValue);
}
}

PBDConstraint::PBDConstraint(
	std::vector<unsigned int> particleIndices,
	float stiffnessValue,
	PBDConstraintType constraintType
)
	: particleIndices(std::move(particleIndices)),
	  stiffnessValue(stiffnessValue),
	  constraintType(constraintType) {}

std::size_t PBDConstraint::cardinality() const {
	return particleIndices.size();
}

const std::vector<unsigned int>& PBDConstraint::indices() const {
	return particleIndices;
}

float PBDConstraint::stiffness() const {
	return stiffnessValue;
}

PBDConstraintType PBDConstraint::type() const {
	return constraintType;
}

bool PBDConstraint::isViolated(const std::vector<Vector3f>& positions, float epsilon) const {
	const float value = evaluate(positions);

	if (constraintType == PBDConstraintType::Equality) {
		return std::abs(value) > epsilon;
	}

	// Inequality constraints are only projected when violated.
	// For C(p) >= 0, violation means C(p) < 0.
	return value < 0.0f;
}

void PBDConstraint::project(
	std::vector<Vector3f>& positions,
	const std::vector<float>& invMass,
	unsigned int solverIterations,
	float epsilon
) const {
	if (!isViolated(positions, epsilon)) return;

	std::vector<Vector3f> gradientValues;
	gradients(positions, gradientValues);
	if (gradientValues.size() != particleIndices.size()) return;

	/*
	Generic PBD projection from the paper:
	   	q_j = grad_{p_j} C(p)
	   	s = C(positions) / ( sum_j ( w_j * | grad_{p_j} C(positions) |^2 ) )
	   	Delta p_i = -k' * s * w_i * grad_{p_i} C(positions)
	Every concrete constraint in the solver only needs to provide C(.) and its gradients q_j; 
	the correction rule itself stays identical for: 
		distance, collision, fixed-point, and dihedral bending constraints.
	*/
	
	float denominator = 0.0f;
	for (std::size_t j = 0; j < particleIndices.size(); ++j) {
		const unsigned int particle = particleIndices[j];
		if (particle >= positions.size() || particle >= invMass.size()) return;
		denominator += invMass[particle] * gradientValues[j].squaredNorm();
	}

	if (denominator <= epsilon) return;

	const float constraintValue = evaluate(positions);
	const float scale = -correctedStiffness(stiffnessValue, solverIterations)
		* (constraintValue / denominator);

	for (std::size_t j = 0; j < particleIndices.size(); ++j) {
		const unsigned int particle = particleIndices[j];
		positions[particle] += scale * invMass[particle] * gradientValues[j];
	}
}

DistanceConstraint::DistanceConstraint(unsigned int i, unsigned int j, float restLength, float stiffness)
	: PBDConstraint(std::vector<unsigned int>{ i, j }, stiffness, PBDConstraintType::Equality),
	  restLength(restLength) {}

float DistanceConstraint::evaluate(const std::vector<Vector3f>& positions) const {
	// Distance constraint:
	//   C(p1, p2) = |p1 - p2| - d
	const Vector3f delta = positions[particleIndices[0]] - positions[particleIndices[1]];
	return delta.norm() - restLength;
}

void DistanceConstraint::gradients(
	const std::vector<Vector3f>& positions,
	std::vector<Vector3f>& outGradients
) const {
	outGradients.assign(2, Vector3f::Zero());

	const Vector3f delta = positions[particleIndices[0]] - positions[particleIndices[1]];
	const float length = delta.norm();
	if (length <= 1e-8f) return;

	// grad_{p1} C(positions) = (p1 - p2) / |p1 - p2|
	// grad_{p2} C(positions) = -grad_{p1} C(positions)
	const Vector3f direction = delta / length;
	outGradients[0] = direction;
	outGradients[1] = -direction;
}

DihedralBendConstraint::DihedralBendConstraint(
	unsigned int edge0,
	unsigned int edge1,
	unsigned int opposite0,
	unsigned int opposite1,
	float restAngle,
	float stiffness
)
	: PBDConstraint(
		std::vector<unsigned int>{ edge0, edge1, opposite0, opposite1 },
		stiffness,
		PBDConstraintType::Equality
	),
	  restAngle(restAngle) {}

float DihedralBendConstraint::angle(const std::vector<Vector3f>& positions, bool* valid) const {
	return dihedralAngleFromPositions(
		positions,
		particleIndices[0],
		particleIndices[1],
		particleIndices[2],
		particleIndices[3],
		valid
	);
}

float DihedralBendConstraint::evaluate(const std::vector<Vector3f>& positions) const {
	bool valid = false;
	const float currentAngle = angle(positions, &valid);
	if (!valid) return 0.0f;
	
	// C_bend(p1, p2, p3, p4) = theta - theta_0,
	// 		where theta is the current dihedral angle between the two incident triangles, 
	// 		and theta_0 is the rest angle measured in the reference pose.
	// This formulation measure the deviation from the rest angle, 
	// and therefore projects toward the rest angle instead of toward a flat fold.
	return currentAngle - restAngle;
}

void DihedralBendConstraint::gradients(
	const std::vector<Vector3f>& positions,
	std::vector<Vector3f>& outGradients
) const {
	outGradients.assign(4, Vector3f::Zero());
	if (particleIndices.size() != 4) return;

	const unsigned int iShared0 = particleIndices[0];
	const unsigned int iShared1 = particleIndices[1];
	const unsigned int iOpp0 = particleIndices[2];
	const unsigned int iOpp1 = particleIndices[3];
	if (iShared0 >= positions.size() || iShared1 >= positions.size()
		|| iOpp0 >= positions.size() || iOpp1 >= positions.size()) {
		return;
	}

	// Local particle order in this solver is:
	//   [0] shared edge vertex p1
	//   [1] shared edge vertex p2
	//   [2] opposite vertex p3 of the first triangle
	//   [3] opposite vertex p4 of the second triangle
	//
	// For the analytic derivative it is convenient to switch to the standard
	// dihedral notation used in many PBD derivations:
	//   p0 = opposite vertex of triangle 1
	//   p1 = opposite vertex of triangle 2
	//   p2 = shared edge vertex 1
	//   p3 = shared edge vertex 2
	const Vector3f& p0 = positions[iOpp0];
	const Vector3f& p1 = positions[iOpp1];
	const Vector3f& p2 = positions[iShared0];
	const Vector3f& p3 = positions[iShared1];

	// The two normalized face normals encode the fold between adjacent
	// triangles. Their dot product is cos(theta), where theta is the dihedral
	// angle across the shared edge.
	Vector3f n1 = (p2 - p0).cross(p3 - p0);
	Vector3f n2 = (p3 - p1).cross(p2 - p1);
	const float n1Len = n1.norm();
	const float n2Len = n2.norm();

	const float eps = 1e-6f;
	if (n1Len < eps || n2Len < eps) return;

	n1 /= n1Len;
	n2 /= n2Len;
	const float area1 = 0.5f * n1Len;
	const float area2 = 0.5f * n2Len;

	float d = n1.dot(n2);
	d = std::max(-1.0f, std::min(1.0f, d));

	// C = acos(d) - theta_0, so by the chain rule:
	//   grad C = -grad d / sqrt(1 - d^2)
	// The denominator becomes singular near perfectly flat or fully folded
	// states, so clamp it away from zero for robustness.
	const float sinTheta = std::sqrt(std::max(1.0f - d * d, eps));

	// These q-vectors are the analytic gradients of d = cos(theta) with respect
	// to the four dihedral vertices. They replace the previous finite-difference
	// approximation while leaving the generic PBD projection rule unchanged.
	// The Muller-style formulas are expressed in terms of triangle areas. Since
	// |(a x b)| is twice the triangle area, use 0.5 * |cross| here rather than
	// the raw cross-product norm to avoid an overall factor-of-two error.
	const Vector3f q0 =
		((p2 - p1).cross(n1) + n2.cross(p2 - p1) * d) / area2;
	const Vector3f q1 =
		-((p3 - p1).cross(n1) + n2.cross(p3 - p1) * d) / area2;
	const Vector3f q2 =
		-((p3 - p0).cross(n2) + n1.cross(p3 - p0) * d) / area1;
	const Vector3f q3 =
		((p2 - p0).cross(n2) + n1.cross(p2 - p0) * d) / area1;

	const Vector3f g0 = -q0 / sinTheta;
	const Vector3f g1 = -q1 / sinTheta;
	const Vector3f g2 = -q2 / sinTheta;
	const Vector3f g3 = -q3 / sinTheta;

	// Map the paper-style order back to this solver's local particle order.
	outGradients[0] = g2;
	outGradients[1] = g3;
	outGradients[2] = g0;
	outGradients[3] = g1;
}

FixedPointConstraint::FixedPointConstraint(unsigned int i, const Eigen::Vector3f& fixedPosition, float stiffness)
	: PBDConstraint(std::vector<unsigned int>{ i }, stiffness, PBDConstraintType::Equality),
	  fixedPosition(fixedPosition) {}

void FixedPointConstraint::setFixedPosition(const Eigen::Vector3f& position) {
	fixedPosition = position;
}

const Eigen::Vector3f& FixedPointConstraint::position() const {
	return fixedPosition;
}

float FixedPointConstraint::evaluate(const std::vector<Vector3f>& positions) const {
	// For logging/debugging, measure the distance to the target anchor.
	return (positions[particleIndices[0]] - fixedPosition).norm();
}

void FixedPointConstraint::gradients(
	const std::vector<Vector3f>& positions,
	std::vector<Vector3f>& outGradients
) const {
	outGradients.assign(1, Vector3f::Zero());
	const Vector3f delta = positions[particleIndices[0]] - fixedPosition;
	const float length = delta.norm();
	if (length <= 1e-8f) return;
	outGradients[0] = delta / length;
}

void FixedPointConstraint::project(
	std::vector<Vector3f>& positions,
	const std::vector<float>& invMass,
	unsigned int solverIterations,
	float epsilon
) const {
	const unsigned int particle = particleIndices[0];
	if (particle >= positions.size() || particle >= invMass.size()) return;

	// A fixed point is treated as a moving positional anchor.
	// Using k' here keeps the formulation consistent with the paper:
	//   p_i <- p_i + k' * (p_fixed - p_i)
	const float kPrime = correctedStiffness(stiffnessValue, solverIterations);
	if ((positions[particle] - fixedPosition).norm() <= epsilon) return;
	positions[particle] += kPrime * (fixedPosition - positions[particle]);
}

SphereCollisionConstraint::SphereCollisionConstraint(
	unsigned int i,
	const Eigen::Vector3f& center,
	float radius,
	float stiffness
)
	: PBDConstraint(std::vector<unsigned int>{ i }, stiffness, PBDConstraintType::Inequality),
	  center(center),
	  radius(radius) {}

float SphereCollisionConstraint::evaluate(const std::vector<Vector3f>& positions) const {
	// Collision inequality:
	//   C(p_i) = |p_i - c| - r >= 0
	return (positions[particleIndices[0]] - center).norm() - radius;
}

void SphereCollisionConstraint::gradients(
	const std::vector<Vector3f>& positions,
	std::vector<Vector3f>& outGradients
) const {
	outGradients.assign(1, Vector3f::Zero());
	Vector3f delta = positions[particleIndices[0]] - center;
	const float length = delta.norm();

	if (length <= 1e-8f) {
		outGradients[0] = Vector3f(0.0f, 0.0f, 1.0f);
		return;
	}

	// grad_{p_i} C = (p_i - c) / |p_i - c|
	outGradients[0] = delta / length;
}

PBDSolver::PBDSolver(pbd_system* system, float* vbuff)
	: system(system), vbuff(vbuff),
	  solverIterations(PBDDefaultParam::solverIterations),
	  dampingFactor(PBDDefaultParam::dampingFactor),
	  collisionEps(PBDDefaultParam::collisionEps),
	  gravity(PBDDefaultParam::gravity) {
	assert(system != nullptr);
	assert(vbuff != nullptr);

	initializeState();
}

void PBDSolver::initializeState() {
	const unsigned int n = system->n_points;
	// Pseudo-code (1)-(3): initialize x, p, v, w.
	x.resize(n);
	p.resize(n);
	v.resize(n);
	invMass.resize(n);

	for (unsigned int i = 0; i < n; ++i) {
		// Read initial position from render vertex buffer.
		Vector3f pos(
			vbuff[3 * i + 0],
			vbuff[3 * i + 1],
			vbuff[3 * i + 2]
		);

		x[i] = pos;
		p[i] = pos;

		// Initial velocity = 0.
		v[i] = Vector3f(0.0f, 0.0f, 0.0f);

		// Inverse mass w_i = 1 / m_i.
		const float mass = system->masses[i];
		invMass[i] = (mass > 0.0f) ? (1.0f / mass) : 0.0f;
	}
}

void PBDSolver::writeBackToVBuff() {
	const unsigned int n = system->n_points;

	for (unsigned int i = 0; i < n; ++i) {
		vbuff[3 * i + 0] = x[i][0];    // x
		vbuff[3 * i + 1] = x[i][1];    // y
		vbuff[3 * i + 2] = x[i][2];    // z
	}
}

void PBDSolver::applyExternalForces(float dt) {
	const unsigned int n = system->n_points;
	// Pseudo-code (5):
	//   v_i <- v_i + dt * f_ext_i / m_i
	// Here gravity is stored directly as an acceleration vector.
	for (unsigned int i = 0; i < n; ++i) {
		if (invMass[i] == 0.0f) continue; // fixed particles do not accelerate
		v[i] += dt * gravity;
	}
}

void PBDSolver::dampVelocities() {
	const unsigned int n = system->n_points;
	if (n == 0) return;

	// (1) Compute total mass and center of mass position:
	//     x_cm = (sum_i x_i m_i) / (sum_i m_i)
	float totalMass = 0.0f;
	Vector3f xcm(0.0f, 0.0f, 0.0f);

	for (unsigned int i = 0; i < n; ++i) {
		const float mass = system->masses[i];
		totalMass += mass;
		xcm += mass * x[i];
	}

	if (totalMass <= 0.0f) return;
	xcm /= totalMass;

	// (2) Compute center of mass velocity:
	//     v_cm = (sum_i v_i m_i) / (sum_i m_i)
	Vector3f vcm(0.0f, 0.0f, 0.0f);
	for (unsigned int i = 0; i < n; ++i) {
		const float mass = system->masses[i];
		vcm += mass * v[i];
	}
	vcm /= totalMass;

	// (3) Compute angular momentum:
	//     L = sum_i r_i x (m_i v_i), where r_i = x_i - x_cm
	Vector3f angularMomentum(0.0f, 0.0f, 0.0f);

	// (4) Compute inertia tensor:
	//     I = sum_i m_i (|r_i|^2 E - r_i r_i^T)
	Eigen::Matrix3f inertia = Eigen::Matrix3f::Zero();

	for (unsigned int i = 0; i < n; ++i) {
		const float mass = system->masses[i];
		const Vector3f r = x[i] - xcm;

		angularMomentum += r.cross(mass * v[i]);

		const float r2 = r.squaredNorm();
		inertia += mass * (r2 * Eigen::Matrix3f::Identity() - r * r.transpose());
	}

	// (5) Compute angular velocity:
	//     omega = I^{-1} L
	Vector3f omega(0.0f, 0.0f, 0.0f);
	if (std::abs(inertia.determinant()) > 1e-8f) {
		omega = inertia.inverse() * angularMomentum;
	}

	// (6)-(8) For each vertex:
	//     delta_v_i = v_cm + omega x r_i - v_i
	//     v_i <- v_i + k_damping * delta_v_i
	for (unsigned int i = 0; i < n; ++i) {
		if (invMass[i] == 0.0f) continue; // keep fixed particles unchanged

		const Vector3f r = x[i] - xcm;
		const Vector3f targetVelocity = vcm + omega.cross(r);
		v[i] += dampingFactor * (targetVelocity - v[i]);
	}
}

void PBDSolver::predictPositions(float dt) {
	const unsigned int n = system->n_points;
	for (unsigned int i = 0; i < n; ++i) {
		if (invMass[i] == 0.0f) {
			p[i] = x[i];
			continue;
		}

		// Pseudo-code (7): predict positions
		//   p_i = x_i + dt * v_i
		p[i] = x[i] + dt * v[i];
	}
}

void PBDSolver::generateCollisionConstraints() {
	// The paper separates collision detection from constraint projection.
	// Persistent colliders stay in sphereColliders, while actual contact
	// constraints are generated per step from predicted positions p.
	generatedCollisionConstraints.clear();

	for (const SphereCollider& collider : sphereColliders) {
		for (unsigned int i = 0; i < system->n_points; ++i) {
			if (invMass[i] == 0.0f) continue;

			const Vector3f delta = p[i] - collider.center;
			if (delta.norm() >= collider.radius) continue;

			generatedCollisionConstraints.push_back(
				std::make_unique<SphereCollisionConstraint>(
					i,
					collider.center,
					collider.radius + collisionEps,
					PBDDefaultParam::collisionStiffness
				)
			);
		}
	}
}

void PBDSolver::projectConstraints(const ConstraintList& constraints) {
	for (const ConstraintPtr& constraint : constraints) {
		constraint->project(p, invMass, solverIterations, collisionEps);
	}
}

void PBDSolver::updateVelocities(float dt) {
	// Pseudo-code (16):
	//   v_i = (p_i - x_i) / dt
	if (dt <= 0.0f) return;

	const unsigned int n = system->n_points;
	for (unsigned int i = 0; i < n; ++i) {
		if (invMass[i] == 0.0f) {
			v[i] = Vector3f(0.0f, 0.0f, 0.0f);
			continue;
		}

		v[i] = (p[i] - x[i]) / dt;
	}
}

void PBDSolver::commitPositions() {
	const unsigned int n = system->n_points;
	for (unsigned int i = 0; i < n; ++i) {
		x[i] = p[i];
	}

	writeBackToVBuff();
}

void PBDSolver::step(float dt) {
	// Paper-aligned solver structure:
	// 1. apply external forces
	// 2. damp velocities
	// 3. predict positions
	// 4. generate collision constraints
	// 5. iterate projections
	// 6. update velocities
	// 7. commit positions
	applyExternalForces(dt);
	dampVelocities();
	predictPositions(dt);
	generateCollisionConstraints();

	for (unsigned int iteration = 0; iteration < solverIterations; ++iteration) {
		projectConstraints(persistentConstraints);
		projectConstraints(generatedCollisionConstraints);
	}

	updateVelocities(dt);
	commitPositions();
}

void PBDSolver::solve(unsigned int n) {
	// Wrapper used by the app to override the number of constraint-projection
	// iterations for a single simulation time step.
	//
	// Important: n is NOT the number of outer time steps. This function still
	// advances the simulation by exactly one dt via step(system->time_step).
	// Instead, n temporarily replaces solverIterations, which controls the
	// inner PBD loop inside step():
	//   for each solver iteration:
	//     project persistent constraints
	//     project generated collision constraints
	//
	// After that one step is finished, the previous default iteration count is
	// restored. This lets the caller choose the quality/cost of one frame update
	// without permanently changing solver configuration.
	const unsigned int previousIterations = solverIterations;
	solverIterations = n;
	step(system->time_step);
	solverIterations = previousIterations;
}

void PBDSolver::pinPoint(unsigned int i) {
	fixPoint(i);
}

void PBDSolver::fixPoint(unsigned int i) {
	if (i >= x.size()) return;

	const Vector3f fixedPosition(
		vbuff[3 * i + 0],
		vbuff[3 * i + 1],
		vbuff[3 * i + 2]
	);

	// Mouse dragging acts like a moving fixed point. If the point is already
	// fixed, only update its target position. Otherwise add a new persistent
	// fixed-point constraint.
	auto existing = fixedPointConstraints.find(i);
	if (existing != fixedPointConstraints.end()) {
		existing->second->setFixedPosition(fixedPosition);
	}
	else {
		auto constraint = std::make_unique<FixedPointConstraint>(i, fixedPosition, 1.0f);
		fixedPointConstraints[i] = constraint.get();
		persistentConstraints.push_back(std::move(constraint));
	}

	// Mark the point as immovable for all other constraints.
	x[i] = fixedPosition;
	p[i] = fixedPosition;
	v[i] = Vector3f(0.0f, 0.0f, 0.0f);
	invMass[i] = 0.0f;
}

void PBDSolver::releasePoint(unsigned int i) {
	if (i >= x.size()) return;

	auto existing = fixedPointConstraints.find(i);
	if (existing == fixedPointConstraints.end()) return;

	FixedPointConstraint* target = existing->second;
	fixedPointConstraints.erase(existing);

	persistentConstraints.erase(
		std::remove_if(
			persistentConstraints.begin(),
			persistentConstraints.end(),
			[target](const ConstraintPtr& constraint) { return constraint.get() == target; }
		),
		persistentConstraints.end()
	);

	const float mass = system->masses[i];
	invMass[i] = (mass > 0.0f) ? (1.0f / mass) : 0.0f;
}

void PBDSolver::addSphereCollider(const Vector3f& center, float radius) {
	// Store persistent collision geometry. Actual contact constraints are
	// generated each step from predicted positions.
	sphereColliders.push_back(SphereCollider{ center, radius });
}

void PBDSolver::addDistanceConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	for (unsigned int index : indices) {
		if (index >= system->spring_list.size()) continue;

		const Edge& edge = system->spring_list[index];
		persistentConstraints.push_back(
			std::make_unique<DistanceConstraint>(
				edge.first,
				edge.second,
				system->rest_lengths[index],
				stiffness
			)
		);
	}
}

void PBDSolver::addDihedralBendConstraints(float stiffness) {
	if (system->triangle_indices.size() < 6 || system->triangle_indices.size() % 3 != 0) return;

	std::unordered_map<SharedEdgeKey, PendingTriangleEdge, SharedEdgeKeyHash> pendingEdges;
	pendingEdges.reserve(system->triangle_indices.size());

	// Build one bending constraint per interior mesh edge. Each triangle is read
	// from the render mesh index buffer, and when the same undirected edge is
	// seen a second time we have found the adjacent triangle pair required by the
	// paper's 4-particle dihedral constraint.
	for (std::size_t triangle = 0; triangle < system->triangle_indices.size(); triangle += 3) {
		const std::array<unsigned int, 3> vertices = {
			system->triangle_indices[triangle + 0],
			system->triangle_indices[triangle + 1],
			system->triangle_indices[triangle + 2]
		};

		for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex) {
			const unsigned int edge0 = vertices[edgeIndex];
			const unsigned int edge1 = vertices[(edgeIndex + 1) % 3];
			const unsigned int opposite = vertices[(edgeIndex + 2) % 3];

			SharedEdgeKey key{ std::min(edge0, edge1), std::max(edge0, edge1) };
			auto existing = pendingEdges.find(key);
			if (existing == pendingEdges.end()) {
				pendingEdges.emplace(key, PendingTriangleEdge{ edge0, edge1, opposite });
				continue;
			}

			const PendingTriangleEdge firstTriangle = existing->second;
			pendingEdges.erase(existing);

			bool valid = false;
			// The rest angle theta_0 is taken from the initial cloth configuration,
			// so the solver preserves the reference fold across this shared edge.
			const float restAngle = dihedralAngleFromPositions(
				x,
				firstTriangle.edge0,
				firstTriangle.edge1,
				firstTriangle.opposite,
				opposite,
				&valid
			);
			if (!valid) continue;

			persistentConstraints.push_back(
				std::make_unique<DihedralBendConstraint>(
					firstTriangle.edge0,
					firstTriangle.edge1,
					firstTriangle.opposite,
					opposite,
					restAngle,
					stiffness
				)
			);
		}
	}
}

void PBDSolver::addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	addDistanceConstraints(indices, stiffness);
}

void PBDSolver::addShearConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	addDistanceConstraints(indices, stiffness);
}

void PBDSolver::addBendConstraints(const std::vector<unsigned int>&, float stiffness) {
	// Bending is generated from adjacent triangle pairs rather than from longer
	// spring edges. This measures the cloth's fold angle directly and therefore
	// remains meaningful even when in-plane stretching changes edge lengths.
	addDihedralBendConstraints(stiffness);


	// addDistanceConstraints(indices, stiffness); for distance-based bending --- IGNORE ---
}