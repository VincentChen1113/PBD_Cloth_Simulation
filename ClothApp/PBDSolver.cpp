#include "PBDSolver.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace PBDDefaultParam {
	static const unsigned int solverIterations = 15;
	static const float dampingFactor = 0.07f;
	static const float collisionEps = 1e-4f;
	static const float collisionStiffness = 1.0f;
	static const float contactFriction = 0.15f;
	static const float selfCollisionStiffness = 0.3f;
	static const unsigned int maxSelfCollisionContactsPerVertex = 4u;    // 12u
	static const float velocitySleepThreshold = 5e-3f;
	static const Eigen::Vector3f gravity(0.0f, 0.0f, -9.81f);
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

struct SpatialHashKey {
	int x;
	int y;
	int z;

	bool operator==(const SpatialHashKey& other) const {
		return x == other.x && y == other.y && z == other.z;
	}
};

struct SpatialHashKeyHash {
	std::size_t operator()(const SpatialHashKey& key) const {
		const std::size_t hx = std::hash<int>{}(key.x);
		const std::size_t hy = std::hash<int>{}(key.y);
		const std::size_t hz = std::hash<int>{}(key.z);
		return hx ^ (hy << 1) ^ (hz << 2);
	}
};

SpatialHashKey hashPosition(const Eigen::Vector3f& position, float cellSize) {
	assert(cellSize > 0.0f);
	return SpatialHashKey{
		static_cast<int>(std::floor(position.x() / cellSize)),
		static_cast<int>(std::floor(position.y() / cellSize)),
		static_cast<int>(std::floor(position.z() / cellSize))
	};
}

void insertSweptVertexIntoHash(
	const Eigen::Vector3f& previousPosition,
	const Eigen::Vector3f& predictedPosition,
	float padding,
	float cellSize,
	unsigned int vertexIndex,
	std::unordered_map<SpatialHashKey, std::vector<unsigned int>, SpatialHashKeyHash>& verticesByCell
) {
	const Eigen::Vector3f minCorner = previousPosition.cwiseMin(predictedPosition)
		- Eigen::Vector3f::Constant(padding);
	const Eigen::Vector3f maxCorner = previousPosition.cwiseMax(predictedPosition)
		+ Eigen::Vector3f::Constant(padding);

	const SpatialHashKey minCell = hashPosition(minCorner, cellSize);
	const SpatialHashKey maxCell = hashPosition(maxCorner, cellSize);
	for (int cellX = minCell.x; cellX <= maxCell.x; ++cellX) {
		for (int cellY = minCell.y; cellY <= maxCell.y; ++cellY) {
			for (int cellZ = minCell.z; cellZ <= maxCell.z; ++cellZ) {
				verticesByCell[SpatialHashKey{ cellX, cellY, cellZ }].push_back(vertexIndex);
			}
		}
	}
}

bool barycentricCoordinates(
	const Eigen::Vector3f& point,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c,
	Eigen::Vector3f& barycentric
) {
	const Eigen::Vector3f v0 = b - a;
	const Eigen::Vector3f v1 = c - a;
	const Eigen::Vector3f v2 = point - a;

	const float d00 = v0.dot(v0);
	const float d01 = v0.dot(v1);
	const float d11 = v1.dot(v1);
	const float d20 = v2.dot(v0);
	const float d21 = v2.dot(v1);
	const float denominator = d00 * d11 - d01 * d01;
	if (std::abs(denominator) <= 1e-8f) {
		return false;
	}

	const float w1 = (d11 * d20 - d01 * d21) / denominator;
	const float w2 = (d00 * d21 - d01 * d20) / denominator;
	const float w0 = 1.0f - w1 - w2;
	barycentric = Eigen::Vector3f(w0, w1, w2);
	return true;
}

bool pointProjectsInsideTriangle(
	const Eigen::Vector3f& point,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c,
	Eigen::Vector3f& barycentric
) {
	if (!barycentricCoordinates(point, a, b, c, barycentric)) {
		return false;
	}

	const float tolerance = -1e-4f;
	return barycentric.x() >= tolerance
		&& barycentric.y() >= tolerance
		&& barycentric.z() >= tolerance;
}

bool triangleNormalAndSignedDistance(
	const std::vector<Eigen::Vector3f>& positions,
	unsigned int vertexIndex,
	unsigned int p1Index,
	unsigned int p2Index,
	unsigned int p3Index,
	Eigen::Vector3f& outNormal,
	float& outSignedDistance
) {
	if (vertexIndex >= positions.size()
		|| p1Index >= positions.size()
		|| p2Index >= positions.size()
		|| p3Index >= positions.size()) {
		return false;
	}

	const Eigen::Vector3f& p1 = positions[p1Index];
	const Eigen::Vector3f& p2 = positions[p2Index];
	const Eigen::Vector3f& p3 = positions[p3Index];
	outNormal = (p2 - p1).cross(p3 - p1);
	const float normalLength = outNormal.norm();
	if (normalLength <= 1e-8f) {
		return false;
	}

	outNormal /= normalLength;
	outSignedDistance = (positions[vertexIndex] - p1).dot(outNormal);
	return true;
}

bool previousFrameSideAndNormal(
	const std::vector<Eigen::Vector3f>& previousPositions,
	const std::vector<Eigen::Vector3f>& referencePositions,
	unsigned int vertexIndex,
	unsigned int p1Index,
	unsigned int p2Index,
	unsigned int p3Index,
	bool& flipNormal,
	float& outSignedDistance
) {
	Eigen::Vector3f normal;

	// Use the previous frame to decide which side the vertex came from so the
	// contact direction follows the paper's above/below logic. The undeformed
	// pose remains a fallback if the previous triangle is degenerate or nearly
	// coplanar with the vertex.
	const bool previousValid = triangleNormalAndSignedDistance(
		previousPositions,
		vertexIndex,
		p1Index,
		p2Index,
		p3Index,
		normal,
		outSignedDistance
	);
	if (!previousValid || std::abs(outSignedDistance) <= 1e-6f) {
		if (!triangleNormalAndSignedDistance(
			referencePositions,
			vertexIndex,
			p1Index,
			p2Index,
			p3Index,
			normal,
			outSignedDistance
		)) {
			return false;
		}
	}

	flipNormal = outSignedDistance < 0.0f;
	if (flipNormal) outSignedDistance = -outSignedDistance;
	return true;
}

float averageEdgeLength(const Eigen::VectorXf& restLengths) {
	float sum = 0.0f;
	unsigned int count = 0u;
	for (int i = 0; i < restLengths.size(); ++i) {
		const float restLength = restLengths[i];
		if (restLength <= 1e-6f) continue;
		sum += restLength;
		++count;
	}
	if (count == 0u) return 0.0f;
	return sum / static_cast<float>(count);
}

Eigen::Vector3f closestPointOnPlane(
	const Eigen::Vector3f& point,
	const Eigen::Vector3f& planePoint,
	const Eigen::Vector3f& planeNormal,
	float* outSignedDistance = nullptr
) {
	const float signedDistance = (point - planePoint).dot(planeNormal);
	if (outSignedDistance != nullptr) {
		*outSignedDistance = signedDistance;
	}
	return point - signedDistance * planeNormal;
}

bool segmentPlaneContactPoint(
	const Eigen::Vector3f& previousPosition,
	const Eigen::Vector3f& predictedPosition,
	const Eigen::Vector3f& planePoint,
	const Eigen::Vector3f& planeNormal,
	float previousSignedDistance,
	float predictedSignedDistance,
	Eigen::Vector3f& outContactPoint
) {
	const float denominator = previousSignedDistance - predictedSignedDistance;
	if (std::abs(denominator) <= 1e-8f) {
		return false;
	}

	const float alpha = std::max(0.0f, std::min(1.0f, previousSignedDistance / denominator));
	outContactPoint = previousPosition + alpha * (predictedPosition - previousPosition);
	float residual = 0.0f;
	outContactPoint = closestPointOnPlane(outContactPoint, planePoint, planeNormal, &residual);
	return true;
}

void applySphereContactDamping(
	std::vector<Eigen::Vector3f>& positions,
	const std::vector<Eigen::Vector3f>& previousPositions,
	const std::vector<float>& invMass,
	const std::vector<SphereCollider>& sphereColliders,
	float collisionEps
) {
	if (sphereColliders.empty()) return;

	const float friction = std::max(0.0f, std::min(1.0f, PBDDefaultParam::contactFriction));
	if (friction <= 0.0f) return;

	const float contactBand = 2.0f * collisionEps;
	for (const SphereCollider& collider : sphereColliders) {
		for (unsigned int i = 0; i < positions.size() && i < previousPositions.size() && i < invMass.size(); ++i) {
			if (invMass[i] == 0.0f) continue;

			Eigen::Vector3f radial = positions[i] - collider.center;
			const float radialLength = radial.norm();
			if (radialLength > collider.radius + contactBand) continue;

			if (radialLength <= 1e-8f) {
				radial = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			}
			else {
				radial /= radialLength;
			}

			const Eigen::Vector3f displacement = positions[i] - previousPositions[i];
			const Eigen::Vector3f normalDisplacement = displacement.dot(radial) * radial;
			const Eigen::Vector3f tangentialDisplacement = displacement - normalDisplacement;
			positions[i] = previousPositions[i] + normalDisplacement + (1.0f - friction) * tangentialDisplacement;
		}
	}
}

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

void PBDConstraint::setStiffness(float stiffness) {
	stiffnessValue = std::max(0.0f, std::min(1.0f, stiffness));
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
	// The denominator becomes singular near perfectly flat or fully folded states, 
	// so clamp it away from zero for robustness (avoid division by zero when d = +1 or d = -1).
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

CollisionConstraint::CollisionConstraint(
	unsigned int i,
	const Eigen::Vector3f& center,
	float radius,
	float stiffness
)
	: PBDConstraint(std::vector<unsigned int>{ i }, stiffness, PBDConstraintType::Inequality),
	  collisionKind(CollisionKind::Sphere),
	  center(center),
	  radius(radius),
	  offset(0.0f),
	  planeNormal(Eigen::Vector3f::Zero()),
	  selfCollisionNormal(Eigen::Vector3f::Zero()),
	  selfCollisionBarycentric(Eigen::Vector3f::Zero()) {}

CollisionConstraint::CollisionConstraint(
	unsigned int i,
	const Eigen::Vector3f& planePoint,
	const Eigen::Vector3f& planeNormal,
	float stiffness
)
	: PBDConstraint(std::vector<unsigned int>{ i }, stiffness, PBDConstraintType::Inequality),
	  collisionKind(CollisionKind::Plane),
	  center(planePoint),
	  radius(0.0f),
	  offset(0.0f),
	  planeNormal(planeNormal.normalized()),
	  selfCollisionNormal(Eigen::Vector3f::Zero()),
	  selfCollisionBarycentric(Eigen::Vector3f::Zero()) {}

CollisionConstraint::CollisionConstraint(
	unsigned int vertex,
	unsigned int p1,
	unsigned int p2,
	unsigned int p3,
	float thickness,
	const Eigen::Vector3f& normal,
	const Eigen::Vector3f& barycentric,
	float stiffness
)
	: PBDConstraint(
		std::vector<unsigned int>{ vertex, p1, p2, p3 },
		stiffness,
		PBDConstraintType::Inequality
	),
	  collisionKind(CollisionKind::SelfVertexTriangle),
	  center(Eigen::Vector3f::Zero()),
	  radius(0.0f),
	  offset(thickness),
	  planeNormal(Eigen::Vector3f::Zero()),
	  selfCollisionNormal(normal),
	  selfCollisionBarycentric(barycentric) {}

float CollisionConstraint::evaluate(const std::vector<Vector3f>& positions) const {
	if (collisionKind == CollisionKind::Sphere) {
		// Collision inequality:
		//   C(p_i) = |p_i - c| - r >= 0
		return (positions[particleIndices[0]] - center).norm() - radius;
	}

	if (collisionKind == CollisionKind::Plane) {
		return (positions[particleIndices[0]] - center).dot(planeNormal);
	}

	if (particleIndices.size() != 4) return 0.0f;

	const Vector3f& q = positions[particleIndices[0]];
	const Vector3f& p1 = positions[particleIndices[1]];

	// Self-collision inequality from Muller et al. 2007, Sec. 4.3:
	//   C(q, p1, p2, p3) = (q - p1) . n - h >= 0
	return (q - p1).dot(selfCollisionNormal) - offset;
}

void CollisionConstraint::gradients(
	const std::vector<Vector3f>& positions,
	std::vector<Vector3f>& outGradients
) const {
	if (collisionKind == CollisionKind::Sphere) {
		outGradients.assign(1, Vector3f::Zero());
		Vector3f delta = positions[particleIndices[0]] - center;
		const float length = delta.norm();

		if (length <= 1e-8f) {
			outGradients[0] = Vector3f(0.0f, 0.0f, 1.0f);
			return;
		}

		// grad_{p_i} C = (p_i - c) / |p_i - c|
		outGradients[0] = delta / length;
		return;
	}

	if (collisionKind == CollisionKind::Plane) {
		outGradients.assign(1, Vector3f::Zero());
		outGradients[0] = planeNormal;
		return;
	}

	outGradients.assign(4, Vector3f::Zero());
	if (particleIndices.size() != 4) return;

	// Using barycentric weights for the triangle vertices reproduces the paper's
	// point-triangle correction split inside the generic PBD projection rule.
	outGradients[0] = selfCollisionNormal;
	outGradients[1] = -selfCollisionBarycentric[0] * selfCollisionNormal;
	outGradients[2] = -selfCollisionBarycentric[1] * selfCollisionNormal;
	outGradients[3] = -selfCollisionBarycentric[2] * selfCollisionNormal;
}

// -----------------------------
// 1. Constructor / Initialization
// -----------------------------

PBDSolver::PBDSolver(pbd_system* system, float* vbuff)
	: system(system), vbuff(vbuff),
	  solverIterations(PBDDefaultParam::solverIterations),
	  dampingFactor(PBDDefaultParam::dampingFactor),
	  collisionEps(PBDDefaultParam::collisionEps),
	  structuralStiffness(1.0f),
	  shearStiffness(0.7f),
	  bendStiffness(0.03f),
	  planeFriction(PBDDefaultParam::contactFriction),
	  selfCollisionThickness(PBDDefaultParam::collisionEps),
	  selfCollisionStiffness(PBDDefaultParam::selfCollisionStiffness),
	  selfCollisionCellSize(PBDDefaultParam::collisionEps),
	  maxSelfCollisionContactsPerVertex(PBDDefaultParam::maxSelfCollisionContactsPerVertex),
	  velocitySleepThreshold(PBDDefaultParam::velocitySleepThreshold),
	  gravity(PBDDefaultParam::gravity) {
	assert(system != nullptr);
	assert(vbuff != nullptr);

	initializeState();
	meshAdjacency.resize(system->n_points);
	for (const Edge& edge : system->spring_list) {
		if (edge.first >= meshAdjacency.size() || edge.second >= meshAdjacency.size()) continue;
		meshAdjacency[edge.first].insert(edge.second);
		meshAdjacency[edge.second].insert(edge.first);
	}

	float minRestLength = std::numeric_limits<float>::max();
	for (int i = 0; i < system->rest_lengths.size(); ++i) {
		const float restLength = system->rest_lengths[i];
		if (restLength > 1e-6f) {
			minRestLength = std::min(minRestLength, restLength);
		}
	}
	const float meanRestLength = averageEdgeLength(system->rest_lengths);

	if (minRestLength < std::numeric_limits<float>::max()) {
		// Self-collision should model a thin cloth thickness, not half an edge
		// length. Large thickness inflates folded cloth and causes false
		// repulsion between nearby layers.
		selfCollisionThickness = std::max(0.05f * minRestLength, collisionEps);
	}
	if (meanRestLength > 0.0f) {
		// Broad-phase hashing is more reliable when cells match the cloth's edge
		// scale instead of the much smaller thickness band.
		selfCollisionCellSize = std::max(meanRestLength, collisionEps);
	}
}

void PBDSolver::initializeState() {
	const unsigned int n = system->n_points;
	// Pseudo-code (1)-(3): initialize x, p, v, w.
	restX.resize(n);
	x.resize(n);
	p.resize(n);
	v.resize(n);
	invMass.resize(n);
	planeContactPoints.assign(n, Vector3f::Zero());
	planeContactNormals.assign(n, Vector3f(0.0f, 0.0f, 1.0f));
	planeContactSignedDistances.assign(n, std::numeric_limits<float>::infinity());

	for (unsigned int i = 0; i < n; ++i) {
		// Read initial position from render vertex buffer.
		Vector3f pos(
			vbuff[3 * i + 0],
			vbuff[3 * i + 1],
			vbuff[3 * i + 2]
		);

		// Preserve the initial pose as the reference side for future
		// vertex-triangle self-collision tests.
		restX[i] = pos;
		x[i] = pos;
		p[i] = pos;

		// Initial velocity = 0.
		v[i] = Vector3f(0.0f, 0.0f, 0.0f);

		// Inverse mass w_i = 1 / m_i.
		const float mass = system->masses[i];
		invMass[i] = (mass > 0.0f) ? (1.0f / mass) : 0.0f;
	}
}

// -----------------------------
// 2. Main Simulation Entry / Solver Loop
// -----------------------------

void PBDSolver::step(float dt) {
	// Paper-aligned solver structure:
	// 1. apply external forces
	// 2. damp velocities
	// 3. predict positions
	// 4. generate collision constraints
	// 5. iterate projections
	// 6. update velocities from projected positions
	// 7. apply post-collision velocity manipulation (friction)
	// 8. commit positions
	applyExternalForces(dt);
	dampVelocities();
	predictPositions(dt);
	generateCollisionConstraints();

	for (unsigned int iteration = 0; iteration < solverIterations; ++iteration) {
		projectConstraints(persistentConstraints);
		projectConstraints(generatedCollisionConstraints);
	}
	applySphereContactDamping(p, x, invMass, sphereColliders, collisionEps);
	updateSelfCollisionDebugStats(p, true);

	updateVelocities(dt);
	applyPlaneContactVelocityDamping();
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

// -----------------------------
// 3. Core PBD Pipeline Stages
// -----------------------------

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

	// Use strictly dissipative damping so residual cloth motion decays instead
	// of preserving rigid-body modes indefinitely.
	const float factor = std::max(0.0f, 1.0f - dampingFactor);
	for (unsigned int i = 0; i < n; ++i) {
		if (invMass[i] == 0.0f) continue;
		v[i] *= factor;
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
		if (v[i].norm() < velocitySleepThreshold) {
			v[i] = Vector3f::Zero();
		}
	}
}

// -----------------------------
// 4. Persistent Constraint Setup / Constraint Construction
// -----------------------------

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

void PBDSolver::addDistanceConstraints(
	const std::vector<unsigned int>& indices,
	float stiffness,
	std::vector<PBDConstraint*>* constraintGroup
) {
	for (unsigned int index : indices) {
		if (index >= system->spring_list.size()) continue;

		const Edge& edge = system->spring_list[index];
		auto constraint = std::make_unique<DistanceConstraint>(
			edge.first,
			edge.second,
			system->rest_lengths[index],
			stiffness
		);
		if (constraintGroup != nullptr) {
			constraintGroup->push_back(constraint.get());
		}
		persistentConstraints.push_back(std::move(constraint));
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

			auto constraint = std::make_unique<DihedralBendConstraint>(
				firstTriangle.edge0,
				firstTriangle.edge1,
				firstTriangle.opposite,
				opposite,
				restAngle,
				stiffness
			);
			bendConstraints.push_back(constraint.get());
			persistentConstraints.push_back(std::move(constraint));
		}
	}
}

void PBDSolver::addStructuralConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	structuralStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	addDistanceConstraints(indices, structuralStiffness, &structuralConstraints);
}

void PBDSolver::addShearConstraints(const std::vector<unsigned int>& indices, float stiffness) {
	shearStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	addDistanceConstraints(indices, shearStiffness, &shearConstraints);
}

void PBDSolver::addBendConstraints(const std::vector<unsigned int>&, float stiffness) {
	// Bending is generated from adjacent triangle pairs rather than from longer
	// spring edges. This measures the cloth's fold angle directly and therefore
	// remains meaningful even when in-plane stretching changes edge lengths.
	bendStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	addDihedralBendConstraints(bendStiffness);


	// addDistanceConstraints(indices, stiffness); for distance-based bending --- IGNORE ---
}

// -----------------------------
// 5. Collision Handling
// -----------------------------

void PBDSolver::generateCollisionConstraints() {
	// The paper separates collision detection from constraint projection.
	// Persistent colliders stay in sphereColliders, while actual contact
	// constraints are generated per step from predicted positions p.
	generatedCollisionConstraints.clear();
	generatedSelfCollisionConstraints.clear();
	selfCollisionDebugStats = SelfCollisionDebugStats{};
	planeContactPoints.assign(system->n_points, Vector3f::Zero());
	planeContactNormals.assign(system->n_points, Vector3f(0.0f, 0.0f, 1.0f));
	planeContactSignedDistances.assign(system->n_points, std::numeric_limits<float>::infinity());

	for (const SphereCollider& collider : sphereColliders) {
		for (unsigned int i = 0; i < system->n_points; ++i) {
			if (invMass[i] == 0.0f) continue;

			const Vector3f delta = p[i] - collider.center;
			if (delta.norm() >= collider.radius) continue;

			generatedCollisionConstraints.push_back(
				std::make_unique<CollisionConstraint>(
					i,
					collider.center,
					collider.radius + collisionEps,
					PBDDefaultParam::collisionStiffness
				)
			);
		}
	}

	for (const PlaneCollider& collider : planeColliders) {
		for (unsigned int i = 0; i < system->n_points; ++i) {
			if (invMass[i] == 0.0f) continue;

			const float previousSignedDistance = (x[i] - collider.point).dot(collider.normal);
			const float predictedSignedDistance = (p[i] - collider.point).dot(collider.normal);
			const bool crossedPlane = previousSignedDistance > collisionEps && predictedSignedDistance < 0.0f;
			const bool insideOrNearPlane = predictedSignedDistance < collisionEps;
			if (!crossedPlane && !insideOrNearPlane) continue;

			Vector3f contactPoint = Vector3f::Zero();
			if (crossedPlane) {
				if (!segmentPlaneContactPoint(
					x[i],
					p[i],
					collider.point,
					collider.normal,
					previousSignedDistance,
					predictedSignedDistance,
					contactPoint
				)) {
					contactPoint = closestPointOnPlane(p[i], collider.point, collider.normal);
				}
			}
			else {
				// Static fallback: when the predicted point is already on or below the
				// floor, use the plane closest point as the contact anchor.
				contactPoint = closestPointOnPlane(p[i], collider.point, collider.normal);
			}

			if (predictedSignedDistance < planeContactSignedDistances[i]) {
				planeContactSignedDistances[i] = predictedSignedDistance;
				planeContactPoints[i] = contactPoint;
				planeContactNormals[i] = collider.normal;
			}

			generatedCollisionConstraints.push_back(
				std::make_unique<CollisionConstraint>(
					i,
					contactPoint,
					collider.normal,
					PBDDefaultParam::collisionStiffness
				)
			);
		}
	}

	generateSelfCollisionConstraints();
	updateSelfCollisionDebugStats(p, false);
}

void PBDSolver::addSphereCollider(const Vector3f& center, float radius) {
	// Store persistent collision geometry. Actual contact constraints are
	// generated each step from predicted positions.
	sphereColliders.push_back(SphereCollider{ center, radius });
}

void PBDSolver::addPlaneCollider(const Vector3f& point, const Vector3f& normal) {
	const float normalLength = normal.norm();
	if (normalLength <= 1e-8f) return;
	planeColliders.push_back(PlaneCollider{ point, normal / normalLength });
}

void PBDSolver::applyPlaneContactVelocityDamping() {
	if (planeContactSignedDistances.empty()) return;

	const float friction = std::max(0.0f, std::min(1.0f, planeFriction));
	const float tangentialSleepSpeed = velocitySleepThreshold;
	for (unsigned int i = 0; i < v.size() && i < invMass.size()
		&& i < planeContactSignedDistances.size()
		&& i < planeContactNormals.size(); ++i) {
		if (invMass[i] == 0.0f) continue;
		if (!std::isfinite(planeContactSignedDistances[i])) continue;

		const Vector3f& contactNormal = planeContactNormals[i];
		const float normalSpeed = v[i].dot(contactNormal);
		const float separatingSpeed = std::max(0.0f, normalSpeed);
		const Vector3f normalVelocity = separatingSpeed * contactNormal;
		Vector3f tangentialVelocity = v[i] - normalSpeed * contactNormal;
		if (friction > 0.0f) {
			tangentialVelocity *= (1.0f - friction);
		}
		if (tangentialVelocity.norm() <= tangentialSleepSpeed) {
			tangentialVelocity = Vector3f::Zero();
		}

		v[i] = normalVelocity + tangentialVelocity;
	}
}

// -----------------------------
// 6. Self-Collision Utilities / Helpers
// -----------------------------

void PBDSolver::generateSelfCollisionConstraints() {
	if (system->triangle_indices.size() < 3 || selfCollisionCellSize <= 0.0f) return;

	std::unordered_map<SpatialHashKey, std::vector<unsigned int>, SpatialHashKeyHash> verticesByCell;
	verticesByCell.reserve(p.size());
	std::vector<unsigned int> contactsPerVertex(system->n_points, 0u);

	for (unsigned int vertex = 0; vertex < p.size(); ++vertex) {
		insertSweptVertexIntoHash(
			x[vertex],
			p[vertex],
			selfCollisionThickness,
			selfCollisionCellSize,
			vertex,
			verticesByCell
		);
	}

	for (std::size_t triangle = 0; triangle + 2 < system->triangle_indices.size(); triangle += 3) {
		const unsigned int p1Index = system->triangle_indices[triangle + 0];
		const unsigned int p2Index = system->triangle_indices[triangle + 1];
		const unsigned int p3Index = system->triangle_indices[triangle + 2];

		const Vector3f& p1 = p[p1Index];
		const Vector3f& p2 = p[p2Index];
		const Vector3f& p3 = p[p3Index];

		Vector3f currentNormal = (p2 - p1).cross(p3 - p1);
		const float currentNormalLength = currentNormal.norm();
		if (currentNormalLength <= 1e-8f) continue;
		currentNormal /= currentNormalLength;

		const Vector3f& x1 = x[p1Index];
		const Vector3f& x2 = x[p2Index];
		const Vector3f& x3 = x[p3Index];
		const Vector3f minCorner = x1.cwiseMin(x2).cwiseMin(x3)
			.cwiseMin(p1).cwiseMin(p2).cwiseMin(p3)
			- Vector3f::Constant(selfCollisionThickness);
		const Vector3f maxCorner = x1.cwiseMax(x2).cwiseMax(x3)
			.cwiseMax(p1).cwiseMax(p2).cwiseMax(p3)
			+ Vector3f::Constant(selfCollisionThickness);

		const SpatialHashKey minCell = hashPosition(minCorner, selfCollisionCellSize);
		const SpatialHashKey maxCell = hashPosition(maxCorner, selfCollisionCellSize);
		std::unordered_set<unsigned int> processedVertices;

		for (int cellX = minCell.x; cellX <= maxCell.x; ++cellX) {
			for (int cellY = minCell.y; cellY <= maxCell.y; ++cellY) {
				for (int cellZ = minCell.z; cellZ <= maxCell.z; ++cellZ) {
					const SpatialHashKey cellKey{ cellX, cellY, cellZ };
					auto cellVertices = verticesByCell.find(cellKey);
					if (cellVertices == verticesByCell.end()) continue;

					for (unsigned int vertexIndex : cellVertices->second) {
						if (!processedVertices.insert(vertexIndex).second) continue;
						if (contactsPerVertex[vertexIndex] >= maxSelfCollisionContactsPerVertex) continue;
						if (vertexIndex == p1Index || vertexIndex == p2Index || vertexIndex == p3Index) continue;
						if (meshAdjacency[vertexIndex].count(p1Index) != 0
							|| meshAdjacency[vertexIndex].count(p2Index) != 0
							|| meshAdjacency[vertexIndex].count(p3Index) != 0) {
							continue;
						}

						const Vector3f& q = p[vertexIndex];
						const float unsignedDistance = std::abs((q - p1).dot(currentNormal));
						if (unsignedDistance >= selfCollisionThickness) continue;

						bool flipNormal = false;
						float previousSignedDistance = 0.0f;
						if (!previousFrameSideAndNormal(
							x,
							restX,
							vertexIndex,
							p1Index,
							p2Index,
							p3Index,
							flipNormal,
							previousSignedDistance
						)) {
							continue;
						}

						// Reorient the contact to the side the vertex occupied in the
						// previous frame before building the current-step inequality.
						const unsigned int orientedP2 = flipNormal ? p3Index : p2Index;
						const unsigned int orientedP3 = flipNormal ? p2Index : p3Index;
						const Vector3f& orientedP1 = p[p1Index];
						const Vector3f& orientedP2Pos = p[orientedP2];
						const Vector3f& orientedP3Pos = p[orientedP3];

						Vector3f orientedNormal = (orientedP2Pos - orientedP1).cross(orientedP3Pos - orientedP1);
						const float orientedNormalLength = orientedNormal.norm();
						if (orientedNormalLength <= 1e-8f) continue;
						orientedNormal /= orientedNormalLength;

						const float signedDistance = (q - orientedP1).dot(orientedNormal);
						if (signedDistance >= selfCollisionThickness) continue;

						// Generate a contact when the vertex crosses the plane over the
						// step, enters the thickness band, or is still overlapping from a
						// previously missed/self-persistent contact.
						const bool crossedPlane = previousSignedDistance > collisionEps && signedDistance < 0.0f;
						const bool enteredThicknessBand = previousSignedDistance >= selfCollisionThickness
							&& signedDistance < selfCollisionThickness;
						const bool persistentOverlap = previousSignedDistance < selfCollisionThickness;
						if (!crossedPlane && !enteredThicknessBand && !persistentOverlap) continue;

						const Vector3f projectedPoint = q - signedDistance * orientedNormal;
						Eigen::Vector3f barycentric;
						if (!pointProjectsInsideTriangle(
							projectedPoint,
							orientedP1,
							orientedP2Pos,
							orientedP3Pos,
							barycentric
						)) {
							continue;
						}

						auto constraint = std::make_unique<CollisionConstraint>(
							vertexIndex,
							p1Index,
							orientedP2,
							orientedP3,
							selfCollisionThickness,
							orientedNormal,
							barycentric,
							selfCollisionStiffness
						);
						generatedSelfCollisionConstraints.push_back(constraint.get());
						generatedCollisionConstraints.push_back(std::move(constraint));
						// Limiting contacts per vertex reduces conflicting constraints in
						// dense folds, which helps suppress pinching and spike artifacts.
						++contactsPerVertex[vertexIndex];
					}
				}
			}
		}
	}
}

// -----------------------------
// 7. Parameter / Tuning Interface
// -----------------------------

void PBDSolver::setConstraintGroupStiffness(const std::vector<PBDConstraint*>& constraints, float stiffness) {
	const float clamped = std::max(0.0f, std::min(1.0f, stiffness));
	for (PBDConstraint* constraint : constraints) {
		if (constraint != nullptr) {
			constraint->setStiffness(clamped);
		}
	}
}

void PBDSolver::setGravity(float gravityMagnitude) {
	gravity = Vector3f(0.0f, 0.0f, -std::max(0.0f, gravityMagnitude));
}

float PBDSolver::getGravity() const {
	return -gravity.z();
}

void PBDSolver::setDampingFactor(float damping) {
	dampingFactor = std::max(0.0f, std::min(1.0f, damping));
}

float PBDSolver::getDampingFactor() const {
	return dampingFactor;
}

void PBDSolver::setPlaneFriction(float friction) {
	planeFriction = std::max(0.0f, std::min(1.0f, friction));
}

float PBDSolver::getPlaneFriction() const {
	return planeFriction;
}

void PBDSolver::setSolverIterations(unsigned int iterations) {
	solverIterations = std::max(1u, iterations);
}

unsigned int PBDSolver::getSolverIterations() const {
	return solverIterations;
}

void PBDSolver::setVelocitySleepThreshold(float threshold) {
	velocitySleepThreshold = std::max(0.0f, threshold);
}

float PBDSolver::getVelocitySleepThreshold() const {
	return velocitySleepThreshold;
}

void PBDSolver::setStructuralStiffness(float stiffness) {
	structuralStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	setConstraintGroupStiffness(structuralConstraints, structuralStiffness);
}

float PBDSolver::getStructuralStiffness() const {
	return structuralStiffness;
}

void PBDSolver::setShearStiffness(float stiffness) {
	shearStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	setConstraintGroupStiffness(shearConstraints, shearStiffness);
}

float PBDSolver::getShearStiffness() const {
	return shearStiffness;
}

void PBDSolver::setBendStiffness(float stiffness) {
	bendStiffness = std::max(0.0f, std::min(1.0f, stiffness));
	setConstraintGroupStiffness(bendConstraints, bendStiffness);
}

float PBDSolver::getBendStiffness() const {
	return bendStiffness;
}

void PBDSolver::setSelfCollisionStiffness(float stiffness) {
	selfCollisionStiffness = std::max(0.0f, std::min(1.0f, stiffness));
}

float PBDSolver::getSelfCollisionStiffness() const {
	return selfCollisionStiffness;
}

float PBDSolver::getSelfCollisionThickness() const {
	return selfCollisionThickness;
}

void PBDSolver::setMaxSelfCollisionContactsPerVertex(unsigned int maxContacts) {
	maxSelfCollisionContactsPerVertex = std::max(1u, maxContacts);
}

unsigned int PBDSolver::getMaxSelfCollisionContactsPerVertex() const {
	return maxSelfCollisionContactsPerVertex;
}

// -----------------------------
// 8. Debug / Diagnostics
// -----------------------------

void PBDSolver::updateSelfCollisionDebugStats(const std::vector<Vector3f>& positions, bool afterProjection) {
	if (!afterProjection) {
		selfCollisionDebugStats.generatedContacts = static_cast<unsigned int>(generatedSelfCollisionConstraints.size());
		selfCollisionDebugStats.initiallyViolatedContacts = 0u;
		selfCollisionDebugStats.maxInitialPenetration = 0.0f;
	}
	else {
		selfCollisionDebugStats.remainingViolatedContacts = 0u;
		selfCollisionDebugStats.maxRemainingPenetration = 0.0f;
	}

	for (const CollisionConstraint* constraint : generatedSelfCollisionConstraints) {
		if (constraint == nullptr) continue;
		const float value = constraint->evaluate(positions);
		if (value >= 0.0f) continue;

		const float penetration = -value;
		if (!afterProjection) {
			++selfCollisionDebugStats.initiallyViolatedContacts;
			selfCollisionDebugStats.maxInitialPenetration = std::max(selfCollisionDebugStats.maxInitialPenetration, penetration);
		}
		else {
			++selfCollisionDebugStats.remainingViolatedContacts;
			selfCollisionDebugStats.maxRemainingPenetration = std::max(selfCollisionDebugStats.maxRemainingPenetration, penetration);
		}
	}
}

// -----------------------------
// 9. Rendering / Buffer Synchronization
// -----------------------------

void PBDSolver::writeBackToVBuff() {
	const unsigned int n = system->n_points;

	for (unsigned int i = 0; i < n; ++i) {
		vbuff[3 * i + 0] = x[i][0];    // x
		vbuff[3 * i + 1] = x[i][1];    // y
		vbuff[3 * i + 2] = x[i][2];    // z
	}
}

void PBDSolver::commitPositions() {
	const unsigned int n = system->n_points;
	for (unsigned int i = 0; i < n; ++i) {
		x[i] = p[i];
	}

	writeBackToVBuff();
}