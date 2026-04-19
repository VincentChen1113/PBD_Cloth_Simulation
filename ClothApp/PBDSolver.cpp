#include "PBDSolver.h"
#include <cassert>
#include <cmath>

namespace PBDDefaultParam {
	static const unsigned int solverIterations = 10;
	static const float dampingFactor = 0.01f;
	static const float collisionEps = 1e-4f;
	static const Eigen::Vector3f gravity(0.0f, 0.0f, -9.8f);
}

PBDSolver::PBDSolver(pbd_system* system, float* vbuff)
	: system(system), vbuff(vbuff),
	  solverIterations(PBDDefaultParam::solverIterations),
	  dampingFactor(PBDDefaultParam::dampingFactor),
	  collisionEps(PBDDefaultParam::collisionEps),
	  gravity(PBDDefaultParam::gravity)
{
	assert(system != nullptr);
	assert(vbuff != nullptr);

	initializeState();
}

void PBDSolver::initializeState() {
	const unsigned int n = system->n_points;

    // resize state vectors to n elements
	x.resize(n);
	p.resize(n);
	v.resize(n);
	invMass.resize(n);

	for (unsigned int i = 0; i < n; ++i) {
		// read initial position from render vertex buffer
		Vector3f pos(
			vbuff[3 * i + 0],
			vbuff[3 * i + 1],
			vbuff[3 * i + 2]
		);

		x[i] = pos;
		p[i] = pos;

		// initial velocity = zero
		v[i] = Vector3f(0.0f, 0.0f, 0.0f);

		// inverse mass
		float mass = system->masses[i];
		if (mass > 0.0f) {
			invMass[i] = 1.0f / mass;
		}
		else {
			invMass[i] = 0.0f;
		}
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

    // apply gravity as external force for now
    // need further extension to support other types of forces (e.g. wind) and user interaction
	for (unsigned int i = 0; i < n; ++i) {
		// fixed particles: invMass = 0
		if (invMass[i] == 0.0f) continue;

		// gravity treated as acceleration directly
		v[i] += dt * gravity;
	}
}

void PBDSolver::dampVelocities() {
	const unsigned int n = system->n_points;
	if (n == 0) return;

	// (1) Compute total mass and center of mass position:
	// x_cm = (sum_i x_i m_i) / (sum_i m_i)
	float totalMass = 0.0f;
	Vector3f xcm(0.0f, 0.0f, 0.0f);

	for (unsigned int i = 0; i < n; ++i) {
		float mass = system->masses[i];
		totalMass += mass;
		xcm += mass * x[i];
	}

	if (totalMass <= 0.0f) return;
	xcm /= totalMass;

	// (2) Compute center of mass velocity:
	// v_cm = (sum_i v_i m_i) / (sum_i m_i)
	Vector3f vcm(0.0f, 0.0f, 0.0f);
	for (unsigned int i = 0; i < n; ++i) {
		float mass = system->masses[i];
		vcm += mass * v[i];
	}
	vcm /= totalMass;

	// (3) Compute angular momentum:
	// L = sum_i r_i x (m_i v_i), where r_i = x_i - x_cm
	Vector3f L(0.0f, 0.0f, 0.0f);

	// (4) Compute inertia tensor:
	// I = sum_i m_i * (|r_i|^2 * E - r_i r_i^T)
	Eigen::Matrix3f I = Eigen::Matrix3f::Zero();

	for (unsigned int i = 0; i < n; ++i) {
		float mass = system->masses[i];
		Vector3f r = x[i] - xcm;            // particle position relative to center of mass

		L += r.cross(mass * v[i]);          // cross product gives rotational contribution

		float r2 = r.squaredNorm();
		I += mass * (r2 * Eigen::Matrix3f::Identity() - r * r.transpose());
	}

	// (5) Compute angular velocity:
	// omega = I^{-1} L
	Vector3f omega(0.0f, 0.0f, 0.0f);

	// Avoid unstable inverse if inertia is singular or nearly singular
	if (I.determinant() != 0.0f) {
		omega = I.inverse() * L;
	}

	// (6)-(8) For each vertex:
	// delta_v_i = v_cm + omega x r_i - v_i
	// v_i <- v_i + k_damping * delta_v_i
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

		p[i] = x[i] + dt * v[i];
	}
}

void PBDSolver::projectFixedConstraints() {
	for (const FixedPointConstraint& constraint : fixedConstraints) {
		if (constraint.i >= p.size()) continue;
		p[constraint.i] = constraint.fixedPosition;
	}
}

void PBDSolver::projectDistanceConstraints(const std::vector<DistanceConstraint>& constraints) {
	for (const DistanceConstraint& constraint : constraints) {
		if (constraint.i >= p.size() || constraint.j >= p.size()) continue;

		const float wi = invMass[constraint.i];
		const float wj = invMass[constraint.j];
		const float wsum = wi + wj;
		if (wsum <= 0.0f) continue;

		Vector3f delta = p[constraint.i] - p[constraint.j];
		const float length = delta.norm();
		if (length <= collisionEps) continue;

		const float correctionScale = constraint.stiffness * (length - constraint.rest_length) / length;
		delta *= correctionScale;

		p[constraint.i] -= (wi / wsum) * delta;
		p[constraint.j] += (wj / wsum) * delta;
	}
}

void PBDSolver::projectSphereCollisions() {
	for (const SphereCollider& collider : sphereColliders) {
		const float targetRadius = collider.radius + collisionEps;
		for (unsigned int i = 0; i < system->n_points; ++i) {
			if (invMass[i] == 0.0f) continue;

			Vector3f offset = p[i] - collider.center;
			const float distance = offset.norm();
			if (distance >= targetRadius) continue;

			if (distance <= collisionEps) {
				offset = Vector3f(0.0f, 0.0f, 1.0f);
			}
			else {
				offset /= distance;
			}

			p[i] = collider.center + targetRadius * offset;
		}
	}
}

void PBDSolver::updateVelocities(float dt) {
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
	applyExternalForces(dt);
	dampVelocities();
	predictPositions(dt);

	for (unsigned int iteration = 0; iteration < solverIterations; ++iteration) {
		projectFixedConstraints();
		projectDistanceConstraints(structuralConstraints);
		projectDistanceConstraints(shearConstraints);
		projectDistanceConstraints(bendConstraints);
		projectSphereCollisions();
	}

	projectFixedConstraints();
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
	if (i >= x.size()) return;

	for (const FixedPointConstraint& constraint : fixedConstraints) {
		if (constraint.i == i) return;
	}

	invMass[i] = 0.0f;
	fixedConstraints.push_back(FixedPointConstraint{ i, x[i] });
}

void PBDSolver::addSphereCollider(const Vector3f& center, float radius) {
	sphereColliders.push_back(SphereCollider{ center, radius });
}

void PBDSolver::addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	for (unsigned int index : indices) {
		if (index >= system->spring_list.size()) continue;
		const Edge& edge = system->spring_list[index];
		structuralConstraints.push_back(DistanceConstraint{ edge.first, edge.second, system->rest_lengths[index], stiffness });
	}
}

void PBDSolver::addShearConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	for (unsigned int index : indices) {
		if (index >= system->spring_list.size()) continue;
		const Edge& edge = system->spring_list[index];
		shearConstraints.push_back(DistanceConstraint{ edge.first, edge.second, system->rest_lengths[index], stiffness });
	}
}

void PBDSolver::addBendConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	for (unsigned int index : indices) {
		if (index >= system->spring_list.size()) continue;
		const Edge& edge = system->spring_list[index];
		bendConstraints.push_back(DistanceConstraint{ edge.first, edge.second, system->rest_lengths[index], stiffness });
	}
}