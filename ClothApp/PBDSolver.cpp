#include "PBDSolver.h"
#include <algorithm>
#include <cassert>
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

	// Generic PBD projection from the paper:
	//   s = C / sum_j w_j * |grad_{p_j} C|^2
	//   Delta p_i = -k' * s * w_i * grad_{p_i} C
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

	// grad_{p1} C = (p1 - p2) / |p1 - p2|
	// grad_{p2} C = -grad_{p1} C
	const Vector3f direction = delta / length;
	outGradients[0] = direction;
	outGradients[1] = -direction;
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

void PBDSolver::addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	addDistanceConstraints(indices, stiffness);
}

void PBDSolver::addShearConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	addDistanceConstraints(indices, stiffness);
}

void PBDSolver::addBendConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	addDistanceConstraints(indices, stiffness);
}