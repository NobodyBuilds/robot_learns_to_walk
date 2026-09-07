#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h>

#include "norender.h"
#include "renderdata.h"
#include "render.h"
#include "vars.h"

// Bullet Physics is the only physics implementation in this file. CUDA is
// used only for the existing raw-triangle renderer and device observations.
// PPO can later observe the device root pose/velocities, all shoulder/hip
// pitch-sideways-twist vars, elbow/knee/waist joint vars, robotMoveX/Y/Z, and
// robotPartTouchingGround plus the named contact flags. The action values can
// use those same device vars; this file does not implement PPO.

//ai was used for this as im not capable for adding a whole physics system from scratch 

#define PI 3.14159265358979323846f
#define RAD(x) ((x) * PI / 180.0f)
#define NUM_PARTS 15
#define TRIS_PER_PART 12
#define TOTAL_TRIS (NUM_PARTS * TRIS_PER_PART)
#define ROBOT_INTEROP_ID 1

#define WAIST_MIN      -30.0f
#define WAIST_MAX       50.0f
#define HIP_MIN         -30.0f
#define HIP_MAX         100.0f
#define HIPROLL_MIN     -10.0f
#define HIPROLL_MAX      45.0f
#define KNEE_MIN          0.0f
#define KNEE_MAX        140.0f
#define SHOULDER_MIN    -60.0f
#define SHOULDER_MAX    170.0f
#define SHOULDER_SIDE_MIN  -90.0f
#define SHOULDER_SIDE_MAX   90.0f
#define SHOULDER_TWIST_MIN -90.0f
#define SHOULDER_TWIST_MAX  90.0f
#define HIP_TWIST_MIN       -45.0f
#define HIP_TWIST_MAX        45.0f
#define ELBOW_MIN         0.0f
#define ELBOW_MAX       150.0f

enum RobotAnchor {
    A_HEADTOP = 0, A_HEADBASE, A_CHEST, A_PELVIS,
    A_LSHOULDER, A_RSHOULDER, A_LELBOW, A_RELBOW,
    A_LWRIST, A_RWRIST, A_LHIP, A_RHIP, A_LKNEE, A_RKNEE,
    A_LANKLE, A_RANKLE, A_LTOE, A_RTOE, NUM_ANCHORS
};

enum RobotPart {
    PART_PELVIS = 0, PART_TORSO, PART_HEAD,
    PART_LUPPERLEG, PART_LLOWERLEG, PART_LFOOT,
    PART_RUPPERLEG, PART_RLOWERLEG, PART_RFOOT,
    PART_LUPPERARM, PART_LFOREARM, PART_LHAND,
    PART_RUPPERARM, PART_RFOREARM, PART_RHAND
};

enum MotorInput {
    MOTOR_LEFT_SHOULDER, MOTOR_RIGHT_SHOULDER,
    MOTOR_LEFT_ELBOW, MOTOR_RIGHT_ELBOW,
    MOTOR_LEFT_HIP, MOTOR_RIGHT_HIP,
    MOTOR_LEFT_KNEE, MOTOR_RIGHT_KNEE
};

struct PartSpec {
    float x0, x1, y0, y1, z0, z1;
    int anchorA, anchorB;
};

static const PartSpec PART_SPECS[NUM_PARTS] = {
    { -0.40f, 0.40f, -0.24f, 0.12f, -0.22f, 0.22f, A_PELVIS, A_CHEST },
    { -0.34f, 0.34f,  0.10f, 1.02f, -0.20f, 0.24f, A_PELVIS, A_CHEST },
    { -0.23f, 0.23f,  0.02f, 0.48f, -0.20f, 0.21f, A_HEADBASE, A_HEADTOP },
    { -0.15f, 0.15f, -0.05f, 1.00f, -0.17f, 0.17f, A_LHIP, A_LKNEE },
    { -0.12f, 0.12f,  0.00f, 0.91f, -0.14f, 0.14f, A_LKNEE, A_LANKLE },
    { -0.13f, 0.13f, -0.15f, 0.34f, -0.09f, 0.05f, A_LANKLE, A_LTOE },
    { -0.15f, 0.15f, -0.05f, 1.00f, -0.17f, 0.17f, A_RHIP, A_RKNEE },
    { -0.12f, 0.12f,  0.00f, 0.91f, -0.14f, 0.14f, A_RKNEE, A_RANKLE },
    { -0.13f, 0.13f, -0.15f, 0.34f, -0.09f, 0.05f, A_RANKLE, A_RTOE },
    { -0.12f, 0.12f, -0.05f, 0.63f, -0.12f, 0.12f, A_LSHOULDER, A_LELBOW },
    { -0.095f,0.095f,  0.00f, 0.58f, -0.10f, 0.10f, A_LELBOW, A_LWRIST },
    { -0.085f,0.085f,  0.60f, 0.88f, -0.08f, 0.08f, A_LELBOW, A_LWRIST },
    { -0.12f, 0.12f, -0.05f, 0.63f, -0.12f, 0.12f, A_RSHOULDER, A_RELBOW },
    { -0.095f,0.095f,  0.00f, 0.58f, -0.10f, 0.10f, A_RELBOW, A_RWRIST },
    { -0.085f,0.085f,  0.60f, 0.88f, -0.08f, 0.08f, A_RELBOW, A_RWRIST },
};

static const float REST_POSE[NUM_ANCHORS][3] = {
    { 0.00f,  1.75f, 0.00f }, { 0.00f,  1.25f, 0.00f },
    { 0.00f,  1.05f, 0.00f }, { 0.00f,  0.00f, 0.00f },
    {-0.62f,  1.02f, 0.00f }, { 0.62f,  1.02f, 0.00f },
    {-0.74f,  0.40f, 0.00f }, { 0.74f,  0.40f, 0.00f },
    {-0.80f, -0.18f, 0.00f }, { 0.80f, -0.18f, 0.00f },
    {-0.28f, -0.02f, 0.00f }, { 0.28f, -0.02f, 0.00f },
    {-0.31f, -1.02f, 0.00f }, { 0.31f, -1.02f, 0.00f },
    {-0.32f, -1.93f, 0.00f }, { 0.32f, -1.93f, 0.00f },
    {-0.32f, -2.02f, 0.35f }, { 0.32f, -2.02f, 0.35f },
};

struct RobotRenderTransform {
    float3 origin;
    float3 axisX;
    float3 axisY;
    float3 axisZ;
};

// rawTriangles3D stores one RGB color for the whole triangle. noRender's
// interop VBO is a regular vertex stream, so the color must be expanded to
// every vertex before OpenGL consumes it.
struct InteropRawTriangle3D {
    float data[18]; // (x, y, z, r, g, b) for each of 3 vertices
};

struct BulletPart {
    btRigidBody* body = nullptr;
    btBoxShape* shape = nullptr;
};

struct HingeMotor {
    btHingeConstraint* joint = nullptr;
    MotorInput input;
    float low;
    float high;
};

enum BallMotorInput {
    BALL_LEFT_SHOULDER, BALL_RIGHT_SHOULDER,
    BALL_LEFT_HIP, BALL_RIGHT_HIP
};

struct BallMotor {
    btGeneric6DofSpring2Constraint* joint = nullptr;
    BallMotorInput input;
    btVector3 lower;
    btVector3 upper;
};

static btDiscreteDynamicsWorld* g_world = nullptr;
static btDefaultCollisionConfiguration* g_collisionConfig = nullptr;
static btCollisionDispatcher* g_dispatcher = nullptr;
static btBroadphaseInterface* g_broadphase = nullptr;
static btSequentialImpulseConstraintSolver* g_solver = nullptr;
static btRigidBody* g_groundBody = nullptr;
static btBoxShape* g_groundShape = nullptr;
static btGeneric6DofSpring2Constraint* g_waist = nullptr;
static BulletPart g_parts[NUM_PARTS];
static std::vector<btTypedConstraint*> g_constraints;
static std::vector<HingeMotor> g_hingeMotors;
static std::vector<BallMotor> g_ballMotors;
static float g_shapeScale = 1.0f;
static cudaGraphicsResource* g_robotInteropResource = nullptr;
static bool g_robotInteropReady = false;
static std::chrono::steady_clock::time_point g_lastPhysicsUpdate;
static bool g_havePhysicsUpdateTime = false;

// Preserved device-side controls used by the existing syncvar interface.
__device__ float g_ragdollTimer = 0.0f;
__device__ float g_muscleScale = 1.0f;

float h_ragdollTimer = 0.0f;
float h_muscleStrength = 1.0f;
rawTriangles3D* d_robot_geom = nullptr;
RobotRenderTransform h_robot_transforms[NUM_PARTS] = {};
RobotRenderTransform* d_robot_transforms = nullptr;

static float safeScale(float scale) { return scale > 0.001f ? scale : 1.0f; }
static float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

// vars.h keeps drag as a per-120 Hz velocity-retention value.  Convert that
// to Bullet's per-second damping coefficient so a value such as 0.98 does not
// accidentally behave like almost-zero damping.
static float bulletLinearDamping() {
    float retention = clampf(h_drag, 0.0f, 0.999999f);
    return clampf(1.0f - std::pow(retention, 120.0f), 0.0f, 1.0f);
}

static btVector3 restPoint(int index, const btVector3& root, float scale) {
    return root + btVector3(REST_POSE[index][0], REST_POSE[index][1], REST_POSE[index][2]) * scale;
}

static btMatrix3x3 basisAlongY(const btVector3& direction) {
    btVector3 y = direction.normalized();
    btVector3 reference = std::fabs(y.dot(btVector3(0, 0, 1))) > 0.9f
        ? btVector3(0, 1, 0) : btVector3(0, 0, 1);
    btVector3 x = y.cross(reference).normalized();
    btVector3 z = x.cross(y).normalized();
    return btMatrix3x3(x.x(), y.x(), z.x(), x.y(), y.y(), z.y(), x.z(), y.z(), z.z());
}

static btMatrix3x3 basisAroundAxis(const btVector3& axis) {
    btVector3 a = axis.normalized();
    btVector3 u, v;
    btPlaneSpace1(a, u, v);
    return btMatrix3x3(u.x(), v.x(), a.x(), u.y(), v.y(), a.y(), u.z(), v.z(), a.z());
}

static btTransform worldFrame(const btVector3& pivot, const btMatrix3x3& basis) {
    btTransform frame;
    frame.setBasis(basis);
    frame.setOrigin(pivot);
    return frame;
}

static btTransform localFrame(btRigidBody* body, const btTransform& frame) {
    return body->getWorldTransform().inverse() * frame;
}

static btRigidBody* makeRigidBody(int part, float mass, const btVector3& center,
    const btMatrix3x3& basis, float scale) {
    const PartSpec& spec = PART_SPECS[part];
    btVector3 halfExtents((spec.x1 - spec.x0) * 0.5f,
        (spec.y1 - spec.y0) * 0.5f, (spec.z1 - spec.z0) * 0.5f);
    btBoxShape* shape = new btBoxShape(halfExtents);
    shape->setLocalScaling(btVector3(scale, scale, scale));
    shape->setMargin(0.015f);
    btVector3 inertia(0, 0, 0);
    shape->calculateLocalInertia(mass, inertia);
    btRigidBody::btRigidBodyConstructionInfo info(mass, nullptr, shape, inertia);
    btRigidBody* body = new btRigidBody(info);
    body->setWorldTransform(worldFrame(center, basis));
    body->setFriction(clampf(h_friction, 0.0f, 1.0f));
    body->setRestitution(clampf(h_bounce, 0.0f, 1.0f));
    body->setDamping(bulletLinearDamping(), 0.04f);
    body->setActivationState(DISABLE_DEACTIVATION);
    body->setUserIndex(part);
    g_parts[part] = { body, shape };
    g_world->addRigidBody(body);
    return body;
}

static void addFixedJoint(btRigidBody* parent, btRigidBody* child, const btVector3& pivot) {
    btTransform frame = worldFrame(pivot, parent->getWorldTransform().getBasis());
    btFixedConstraint* joint = new btFixedConstraint(
        *parent, *child, localFrame(parent, frame), localFrame(child, frame));
    g_world->addConstraint(joint, true);
    g_constraints.push_back(joint);
}

static void addHingeJoint(btRigidBody* parent, btRigidBody* child, const btVector3& pivot,
    const btVector3& axis, float low, float high, MotorInput input) {
    btTransform frame = worldFrame(pivot, basisAroundAxis(axis));
    btHingeConstraint* joint = new btHingeConstraint(
        *parent, *child, localFrame(parent, frame), localFrame(child, frame), true);
    joint->setLimit(RAD(low), RAD(high));
    joint->setMaxMotorImpulse(30.0f);
    joint->enableMotor(true);
    g_world->addConstraint(joint, true);
    g_constraints.push_back(joint);
    g_hingeMotors.push_back({ joint, input, low, high });
}

static void addBallJoint(btRigidBody* parent, btRigidBody* child, const btVector3& pivot,
    const btVector3& lower, const btVector3& upper, BallMotorInput input) {
    btTransform frame = worldFrame(pivot, btMatrix3x3::getIdentity());
    btGeneric6DofSpring2Constraint* joint = new btGeneric6DofSpring2Constraint(
        *parent, *child, localFrame(parent, frame), localFrame(child, frame), RO_XYZ);
    joint->setLinearLowerLimit(btVector3(0, 0, 0));
    joint->setLinearUpperLimit(btVector3(0, 0, 0));
    joint->setAngularLowerLimit(lower);
    joint->setAngularUpperLimit(upper);
    for (int axis = 3; axis <= 5; axis++) {
        joint->enableMotor(axis, true);
        joint->setServo(axis, true);
        joint->setTargetVelocity(axis, 8.0f);
        joint->setMaxMotorForce(axis, 35.0f);
    }
    g_world->addConstraint(joint, true);
    g_constraints.push_back(joint);
    g_ballMotors.push_back({ joint, input, lower, upper });
}

static void addWaistJoint(btRigidBody* pelvis, btRigidBody* torso, const btVector3& pivot) {
    btTransform frame = worldFrame(pivot, btMatrix3x3::getIdentity());
    g_waist = new btGeneric6DofSpring2Constraint(
        *pelvis, *torso, localFrame(pelvis, frame), localFrame(torso, frame));
    g_waist->setLinearLowerLimit(btVector3(0, 0, 0));
    g_waist->setLinearUpperLimit(btVector3(0, 0, 0));
    g_waist->setAngularLowerLimit(btVector3(RAD(WAIST_MIN), 0.0f, RAD(HIPROLL_MIN)));
    g_waist->setAngularUpperLimit(btVector3(RAD(WAIST_MAX), 0.0f, RAD(HIPROLL_MAX)));
    for (int axis = 3; axis <= 5; axis++) {
        g_waist->enableMotor(axis, true);
        g_waist->setServo(axis, true);
        g_waist->setTargetVelocity(axis, 8.0f);
        g_waist->setMaxMotorForce(axis, 40.0f);
    }
    g_world->addConstraint(g_waist, true);
    g_constraints.push_back(g_waist);
}

static void destroyBulletWorld() {
    if (g_world) {
        for (btTypedConstraint* constraint : g_constraints) g_world->removeConstraint(constraint);
        for (btTypedConstraint* constraint : g_constraints) delete constraint;
        g_constraints.clear();
        g_hingeMotors.clear();
        g_ballMotors.clear();
        g_waist = nullptr;
        for (BulletPart& part : g_parts) {
            if (part.body) g_world->removeRigidBody(part.body);
            delete part.body;
            delete part.shape;
            part = {};
        }
        if (g_groundBody) g_world->removeRigidBody(g_groundBody);
        delete g_groundBody;
        delete g_groundShape;
        g_groundBody = nullptr;
        g_groundShape = nullptr;
    }
    delete g_world;
    delete g_solver;
    delete g_broadphase;
    delete g_dispatcher;
    delete g_collisionConfig;
    g_world = nullptr;
    g_solver = nullptr;
    g_broadphase = nullptr;
    g_dispatcher = nullptr;
    g_collisionConfig = nullptr;
}

static void createBulletWorld() {
    destroyBulletWorld();
    g_collisionConfig = new btDefaultCollisionConfiguration();
    g_dispatcher = new btCollisionDispatcher(g_collisionConfig);
    g_broadphase = new btDbvtBroadphase();
    g_solver = new btSequentialImpulseConstraintSolver();
    g_world = new btDiscreteDynamicsWorld(g_dispatcher, g_broadphase, g_solver, g_collisionConfig);
    g_world->setGravity(btVector3(0, h_gravity, 0));

    g_groundShape = new btBoxShape(btVector3(floorwidth * 0.5f, 0.5f, floorheight * 0.5f));
    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(floorX, floorY - 0.5f, floorZ));
    btRigidBody::btRigidBodyConstructionInfo groundInfo(0.0f, nullptr, g_groundShape);
    g_groundBody = new btRigidBody(groundInfo);
    g_groundBody->setWorldTransform(groundTransform);
    g_groundBody->setFriction(clampf(h_friction, 0.0f, 1.0f));
    g_groundBody->setRestitution(clampf(h_bounce, 0.0f, 1.0f));
    g_groundBody->setUserIndex(-1);
    g_world->addRigidBody(g_groundBody);

    const float scale = safeScale(h_robotScale);
    const btVector3 root(h_robotX, h_robotY, h_robotZ);
    for (int part = 0; part < NUM_PARTS; part++) {
        const PartSpec& spec = PART_SPECS[part];
        btVector3 a = restPoint(spec.anchorA, root, scale);
        btVector3 b = restPoint(spec.anchorB, root, scale);
        btMatrix3x3 basis = basisAlongY(b - a);
        btVector3 localCenter(0, (spec.y0 + spec.y1) * 0.5f * scale,
            (spec.z0 + spec.z1) * 0.5f * scale);
        btVector3 center = a + basis * localCenter;
        float mass = 1.0f;
        if (part == PART_PELVIS || part == PART_TORSO) mass = 8.0f;
        else if (part == PART_HEAD) mass = 2.0f;
        else if (part == PART_LUPPERLEG || part == PART_RUPPERLEG) mass = 3.0f;
        else if (part == PART_LLOWERLEG || part == PART_RLOWERLEG) mass = 2.0f;
        makeRigidBody(part, mass, center, basis, scale);
    }

    addWaistJoint(g_parts[PART_PELVIS].body, g_parts[PART_TORSO].body,
        restPoint(A_PELVIS, root, scale));
    addFixedJoint(g_parts[PART_TORSO].body, g_parts[PART_HEAD].body,
        restPoint(A_HEADBASE, root, scale));
    addBallJoint(g_parts[PART_PELVIS].body, g_parts[PART_LUPPERLEG].body,
        restPoint(A_LHIP, root, scale),
        btVector3(RAD(HIP_MIN), RAD(HIPROLL_MIN), RAD(HIP_TWIST_MIN)),
        btVector3(RAD(HIP_MAX), RAD(HIPROLL_MAX), RAD(HIP_TWIST_MAX)), BALL_LEFT_HIP);
    addBallJoint(g_parts[PART_PELVIS].body, g_parts[PART_RUPPERLEG].body,
        restPoint(A_RHIP, root, scale),
        btVector3(RAD(HIP_MIN), RAD(HIPROLL_MIN), RAD(HIP_TWIST_MIN)),
        btVector3(RAD(HIP_MAX), RAD(HIPROLL_MAX), RAD(HIP_TWIST_MAX)), BALL_RIGHT_HIP);
    addHingeJoint(g_parts[PART_LUPPERLEG].body, g_parts[PART_LLOWERLEG].body,
        restPoint(A_LKNEE, root, scale), btVector3(1, 0, 0), KNEE_MIN, KNEE_MAX, MOTOR_LEFT_KNEE);
    addHingeJoint(g_parts[PART_RUPPERLEG].body, g_parts[PART_RLOWERLEG].body,
        restPoint(A_RKNEE, root, scale), btVector3(1, 0, 0), KNEE_MIN, KNEE_MAX, MOTOR_RIGHT_KNEE);
    addFixedJoint(g_parts[PART_LLOWERLEG].body, g_parts[PART_LFOOT].body,
        restPoint(A_LANKLE, root, scale));
    addFixedJoint(g_parts[PART_RLOWERLEG].body, g_parts[PART_RFOOT].body,
        restPoint(A_RANKLE, root, scale));
    addBallJoint(g_parts[PART_TORSO].body, g_parts[PART_LUPPERARM].body,
        restPoint(A_LSHOULDER, root, scale),
        btVector3(RAD(SHOULDER_MIN), RAD(SHOULDER_SIDE_MIN), RAD(SHOULDER_TWIST_MIN)),
        btVector3(RAD(SHOULDER_MAX), RAD(SHOULDER_SIDE_MAX), RAD(SHOULDER_TWIST_MAX)), BALL_LEFT_SHOULDER);
    addBallJoint(g_parts[PART_TORSO].body, g_parts[PART_RUPPERARM].body,
        restPoint(A_RSHOULDER, root, scale),
        btVector3(RAD(SHOULDER_MIN), RAD(SHOULDER_SIDE_MIN), RAD(SHOULDER_TWIST_MIN)),
        btVector3(RAD(SHOULDER_MAX), RAD(SHOULDER_SIDE_MAX), RAD(SHOULDER_TWIST_MAX)), BALL_RIGHT_SHOULDER);
    addHingeJoint(g_parts[PART_LUPPERARM].body, g_parts[PART_LFOREARM].body,
        restPoint(A_LELBOW, root, scale), btVector3(1, 0, 0), ELBOW_MIN, ELBOW_MAX, MOTOR_LEFT_ELBOW);
    addHingeJoint(g_parts[PART_RUPPERARM].body, g_parts[PART_RFOREARM].body,
        restPoint(A_RELBOW, root, scale), btVector3(1, 0, 0), ELBOW_MIN, ELBOW_MAX, MOTOR_RIGHT_ELBOW);
    addFixedJoint(g_parts[PART_LFOREARM].body, g_parts[PART_LHAND].body,
        restPoint(A_LWRIST, root, scale));
    addFixedJoint(g_parts[PART_RFOREARM].body, g_parts[PART_RHAND].body,
        restPoint(A_RWRIST, root, scale));
}

static float motorTarget(MotorInput input) {
    switch (input) {
    case MOTOR_LEFT_SHOULDER: return clampf(h_leftShoulderJoint, SHOULDER_MIN, SHOULDER_MAX);
    case MOTOR_RIGHT_SHOULDER: return clampf(h_rightShoulderJoint, SHOULDER_MIN, SHOULDER_MAX);
    case MOTOR_LEFT_ELBOW: return clampf(h_leftElbowJoint, ELBOW_MIN, ELBOW_MAX);
    case MOTOR_RIGHT_ELBOW: return clampf(h_rightElbowJoint, ELBOW_MIN, ELBOW_MAX);
    case MOTOR_LEFT_HIP: return clampf(h_leftUpperLegJoint, HIP_MIN, HIP_MAX);
    case MOTOR_RIGHT_HIP: return clampf(h_rightUpperLegJoint, HIP_MIN, HIP_MAX);
    case MOTOR_LEFT_KNEE: return clampf(h_leftKneeJoint, KNEE_MIN, KNEE_MAX);
    case MOTOR_RIGHT_KNEE: return clampf(h_rightKneeJoint, KNEE_MIN, KNEE_MAX);
    }
    return 0.0f;
}

static btVector3 ballMotorTarget(BallMotorInput input) {
    switch (input) {
    case BALL_LEFT_SHOULDER:
        return btVector3(RAD(h_leftShoulderJoint), RAD(h_leftShoulderJointSideways),
            RAD(h_leftShoulderJointTwist));
    case BALL_RIGHT_SHOULDER:
        return btVector3(RAD(h_rightShoulderJoint), RAD(h_rightShoulderJointSideways),
            RAD(h_rightShoulderJointTwist));
    case BALL_LEFT_HIP:
        return btVector3(RAD(h_leftUpperLegJoint), RAD(h_leftHipJointSideways),
            RAD(h_leftHipJointTwist));
    case BALL_RIGHT_HIP:
        return btVector3(RAD(h_rightUpperLegJoint), RAD(h_rightHipJointSideways),
            RAD(h_rightHipJointTwist));
    }
    return btVector3(0, 0, 0);
}

static void driveBulletJoints(float dt) {
    bool motorsOn = h_muscleStrength > 0.001f && h_ragdollTimer <= 0.0f;
    for (const HingeMotor& motor : g_hingeMotors) {
        motor.joint->enableMotor(motorsOn);
        if (motorsOn) {
            float target = clampf(motorTarget(motor.input), motor.low, motor.high);
            motor.joint->setMaxMotorImpulse(30.0f * clampf(h_muscleStrength, 0.0f, 2.0f));
            motor.joint->setMotorTarget(RAD(target), dt);
        }
    }

    for (const BallMotor& motor : g_ballMotors) {
        for (int axis = 3; axis <= 5; axis++)
            motor.joint->enableMotor(axis, motorsOn);
        if (motorsOn) {
            btVector3 target = ballMotorTarget(motor.input);
            for (int axis = 0; axis < 3; axis++)
                target[axis] = clampf(float(target[axis]), float(motor.lower[axis]),
                    float(motor.upper[axis]));
            for (int axis = 0; axis < 3; axis++) {
                motor.joint->setServoTarget(axis + 3, target[axis]);
                motor.joint->setTargetVelocity(axis + 3, 8.0f);
                motor.joint->setMaxMotorForce(axis + 3, 35.0f * clampf(h_muscleStrength, 0.0f, 2.0f));
            }
        }
    }

    if (!g_waist) return;
    g_waist->enableMotor(3, motorsOn);
    g_waist->enableMotor(5, motorsOn);
    if (motorsOn) {
        g_waist->setServoTarget(3, RAD(clampf(h_hipjoints, WAIST_MIN, WAIST_MAX)));
        g_waist->setServoTarget(5, RAD(clampf(h_hipJointSideways, HIPROLL_MIN, HIPROLL_MAX)));
        g_waist->setMaxMotorForce(3, 40.0f * h_muscleStrength);
        g_waist->setMaxMotorForce(5, 40.0f * h_muscleStrength);
    }
}

static void applyRobotMovement(float frameDt) {
    btRigidBody* pelvis = g_parts[PART_PELVIS].body;
    if (!pelvis || frameDt <= 0.0f) return;

    // The command is robot-local, so forward/sideways/up movement follows the
    // robot instead of being locked to one world axis after it rotates or falls.
    btVector3 localAcceleration(
        clampf(h_robotMoveX, -25.0f, 25.0f),
        clampf(h_robotMoveY, -25.0f, 25.0f),
        clampf(h_robotMoveZ, -25.0f, 25.0f));
    btVector3 worldAcceleration = pelvis->getWorldTransform().getBasis() * localAcceleration;
    const btScalar maxAcceleration = 25.0f;
    if (worldAcceleration.length2() > maxAcceleration * maxAcceleration)
        worldAcceleration = worldAcceleration.normalized() * maxAcceleration;

    // An impulse gives the same acceleration across all fixed Bullet
    // substeps performed for this render frame.
    pelvis->applyCentralImpulse(worldAcceleration * pelvis->getMass() * frameDt);
}

static void updateBulletMaterial() {
    if (!g_world) return;
    float frictionValue = clampf(h_friction, 0.0f, 1.0f);
    float restitutionValue = clampf(h_bounce, 0.0f, 1.0f);
    float damping = bulletLinearDamping();
    if (g_groundBody) {
        g_groundBody->setFriction(frictionValue);
        g_groundBody->setRestitution(restitutionValue);
    }
    for (BulletPart& part : g_parts) {
        if (!part.body) continue;
        part.body->setFriction(frictionValue);
        part.body->setRestitution(restitutionValue);
        part.body->setDamping(damping, 0.04f);
    }
}

static void updateBulletScale() {
    float scale = safeScale(h_robotScale);
    if (std::fabs(scale - g_shapeScale) < 1e-5f) return;
    for (BulletPart& part : g_parts)
        if (part.shape) part.shape->setLocalScaling(btVector3(scale, scale, scale));
    g_shapeScale = scale;
}

static void pullJointTargetsFromDevice() {
    cudaMemcpyFromSymbol(&h_leftShoulderJoint, leftShoulderJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightShoulderJoint, rightShoulderJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftShoulderJointSideways, leftShoulderJointSideways, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightShoulderJointSideways, rightShoulderJointSideways, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftShoulderJointTwist, leftShoulderJointTwist, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightShoulderJointTwist, rightShoulderJointTwist, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftElbowJoint, leftElbowJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightElbowJoint, rightElbowJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_hipjoints, hipjoints, sizeof(float));
    cudaMemcpyFromSymbol(&h_hipJointSideways, hipJointSideways, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftUpperLegJoint, leftUpperLegJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightUpperLegJoint, rightUpperLegJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftHipJointSideways, leftHipJointSideways, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightHipJointSideways, rightHipJointSideways, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftHipJointTwist, leftHipJointTwist, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightHipJointTwist, rightHipJointTwist, sizeof(float));
    cudaMemcpyFromSymbol(&h_leftKneeJoint, leftKneeJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_rightKneeJoint, rightKneeJoint, sizeof(float));
    cudaMemcpyFromSymbol(&h_robotMoveX, robotMoveX, sizeof(float));
    cudaMemcpyFromSymbol(&h_robotMoveY, robotMoveY, sizeof(float));
    cudaMemcpyFromSymbol(&h_robotMoveZ, robotMoveZ, sizeof(float));
}

static btVector3 currentRobotRoot() {
    const btTransform& transform = g_parts[PART_PELVIS].body->getWorldTransform();
    float scale = safeScale(h_robotScale);
    btVector3 localCenter(0,
        (PART_SPECS[PART_PELVIS].y0 + PART_SPECS[PART_PELVIS].y1) * 0.5f * scale,
        (PART_SPECS[PART_PELVIS].z0 + PART_SPECS[PART_PELVIS].z1) * 0.5f * scale);
    return transform * (-localCenter);
}

static void teleportRobotToHostRoot() {
    if (!g_world || !g_parts[PART_PELVIS].body) return;
    btVector3 delta(btVector3(h_robotX, h_robotY, h_robotZ) - currentRobotRoot());
    for (BulletPart& part : g_parts) {
        btTransform transform = part.body->getWorldTransform();
        transform.setOrigin(transform.getOrigin() + delta);
        part.body->setWorldTransform(transform);
        part.body->setLinearVelocity(btVector3(0, 0, 0));
        part.body->setAngularVelocity(btVector3(0, 0, 0));
        part.body->activate(true);
    }
}

static void writeBulletStateToDevice() {
    if (!g_parts[PART_PELVIS].body) return;
    btRigidBody* pelvis = g_parts[PART_PELVIS].body;
    btVector3 root = currentRobotRoot();
    btVector3 velocity = pelvis->getLinearVelocity();
    btVector3 angularVelocity = pelvis->getAngularVelocity();
    btScalar yaw, pitch, roll;
    pelvis->getWorldTransform().getBasis().getEulerZYX(yaw, pitch, roll);
    float degrees = 180.0f / PI;
    float values[12] = {
        float(root.x()), float(root.y()), float(root.z()),
        float(velocity.x()), float(velocity.y()), float(velocity.z()),
        float(pitch) * degrees, float(yaw) * degrees, float(roll) * degrees,
        float(angularVelocity.x()), float(angularVelocity.y()), float(angularVelocity.z())
    };
    cudaMemcpyToSymbol(robotX, &values[0], sizeof(float));
    cudaMemcpyToSymbol(robotY, &values[1], sizeof(float));
    cudaMemcpyToSymbol(robotZ, &values[2], sizeof(float));
    cudaMemcpyToSymbol(robotVelX, &values[3], sizeof(float));
    cudaMemcpyToSymbol(robotVelY, &values[4], sizeof(float));
    cudaMemcpyToSymbol(robotVelZ, &values[5], sizeof(float));
    cudaMemcpyToSymbol(robotRotX, &values[6], sizeof(float));
    cudaMemcpyToSymbol(robotRotY, &values[7], sizeof(float));
    cudaMemcpyToSymbol(robotRotZ, &values[8], sizeof(float));
    cudaMemcpyToSymbol(robotAngularVelX, &values[9], sizeof(float));
    cudaMemcpyToSymbol(robotAngularVelY, &values[10], sizeof(float));
    cudaMemcpyToSymbol(robotAngularVelZ, &values[11], sizeof(float));
}

static void writeContactFlagsToDevice() {
    bool contacts[ROBOT_PART_COUNT] = {};
    bool any = false;
    btDispatcher* dispatcher = g_world->getDispatcher();
    int manifoldCount = dispatcher->getNumManifolds();
    for (int i = 0; i < manifoldCount; i++) {
        btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(i);
        const btCollisionObject* objectA = manifold->getBody0();
        const btCollisionObject* objectB = manifold->getBody1();
        int part = -1;
        if (objectA->getUserIndex() >= 0 && objectA->getUserIndex() < NUM_PARTS && objectB == g_groundBody)
            part = objectA->getUserIndex();
        if (objectB->getUserIndex() >= 0 && objectB->getUserIndex() < NUM_PARTS && objectA == g_groundBody)
            part = objectB->getUserIndex();
        if (part < 0) continue;
        for (int p = 0; p < manifold->getNumContacts(); p++) {
            if (manifold->getContactPoint(p).getDistance() <= 0.02f) {
                contacts[part] = true;
                any = true;
                break;
            }
        }
    }
    cudaMemcpyToSymbol(robotPartTouchingGround, contacts, sizeof(contacts));
    cudaMemcpyToSymbol(robotAnyPartTouchingGround, &any, sizeof(bool));
    cudaMemcpyToSymbol(robotPelvisTouchingGround, &contacts[PART_PELVIS], sizeof(bool));
    cudaMemcpyToSymbol(robotTorsoTouchingGround, &contacts[PART_TORSO], sizeof(bool));
    cudaMemcpyToSymbol(robotHeadTouchingGround, &contacts[PART_HEAD], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftUpperLegTouchingGround, &contacts[PART_LUPPERLEG], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftLowerLegTouchingGround, &contacts[PART_LLOWERLEG], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftFootTouchingGround, &contacts[PART_LFOOT], sizeof(bool));
    cudaMemcpyToSymbol(robotRightUpperLegTouchingGround, &contacts[PART_RUPPERLEG], sizeof(bool));
    cudaMemcpyToSymbol(robotRightLowerLegTouchingGround, &contacts[PART_RLOWERLEG], sizeof(bool));
    cudaMemcpyToSymbol(robotRightFootTouchingGround, &contacts[PART_RFOOT], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftUpperArmTouchingGround, &contacts[PART_LUPPERARM], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftForearmTouchingGround, &contacts[PART_LFOREARM], sizeof(bool));
    cudaMemcpyToSymbol(robotLeftHandTouchingGround, &contacts[PART_LHAND], sizeof(bool));
    cudaMemcpyToSymbol(robotRightUpperArmTouchingGround, &contacts[PART_RUPPERARM], sizeof(bool));
    cudaMemcpyToSymbol(robotRightForearmTouchingGround, &contacts[PART_RFOREARM], sizeof(bool));
    cudaMemcpyToSymbol(robotRightHandTouchingGround, &contacts[PART_RHAND], sizeof(bool));
}

static void updateRenderTransforms() {
    for (int part = 0; part < NUM_PARTS; part++) {
        const btTransform& transform = g_parts[part].body->getWorldTransform();
        const btMatrix3x3& basis = transform.getBasis();
        h_robot_transforms[part].origin = make_float3(
            float(transform.getOrigin().x()), float(transform.getOrigin().y()), float(transform.getOrigin().z()));
        btVector3 x = basis.getColumn(0), y = basis.getColumn(1), z = basis.getColumn(2);
        h_robot_transforms[part].axisX = make_float3(float(x.x()), float(x.y()), float(x.z()));
        h_robot_transforms[part].axisY = make_float3(float(y.x()), float(y.y()), float(y.z()));
        h_robot_transforms[part].axisZ = make_float3(float(z.x()), float(z.y()), float(z.z()));
    }
    cudaMemcpy(d_robot_transforms, h_robot_transforms, sizeof(h_robot_transforms), cudaMemcpyHostToDevice);
}

static bool mapRobotInteropBuffer(InteropRawTriangle3D*& output) {
    output = nullptr;
    if (!g_robotInteropReady || !g_robotInteropResource) return false;
    cudaError_t error = cudaGraphicsMapResources(1, &g_robotInteropResource, 0);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "robot interop map failed: %s\n", cudaGetErrorString(error));
        return false;
    }
    size_t bytes = 0;
    error = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&output), &bytes,
        g_robotInteropResource);
    if (error != cudaSuccess || !output || bytes < TOTAL_TRIS * sizeof(InteropRawTriangle3D)) {
        std::fprintf(stderr, "robot interop pointer failed: %s\n",
            cudaGetErrorString(error));
        cudaGraphicsUnmapResources(1, &g_robotInteropResource, 0);
        output = nullptr;
        return false;
    }
    return true;
}

static void unmapRobotInteropBuffer() {
    if (!g_robotInteropReady || !g_robotInteropResource) return;
    cudaError_t error = cudaGraphicsUnmapResources(1, &g_robotInteropResource, 0);
    if (error != cudaSuccess)
        std::fprintf(stderr, "robot interop unmap failed: %s\n", cudaGetErrorString(error));
}

static void addShadedBox(std::vector<rawTriangles3D>& tris,
    float x0, float x1, float y0, float y1, float z0, float z1,
    float taperX, float taperZ, float r, float g, float b,
    float frontR = -1.0f, float frontG = 0.0f, float frontB = 0.0f) {
    vertex3d c[8] = {
        {x0, y0, z0}, {x1, y0, z0}, {x1 - taperX, y1, z0 + taperZ}, {x0 + taperX, y1, z0 + taperZ},
        {x0, y0, z1}, {x1, y0, z1}, {x1 - taperX, y1, z1 - taperZ}, {x0 + taperX, y1, z1 - taperZ}
    };
    const int faces[6][6] = {{0,1,2,2,3,0},{1,5,6,6,2,1},{5,4,7,7,6,5},
        {4,0,3,3,7,4},{3,2,6,6,7,3},{4,5,1,1,0,4}};
    const float shade[6] = {0.55f,0.72f,0.95f,0.65f,1.0f,0.35f};
    for (int face = 0; face < 6; face++) {
        bool visor = face == 2 && frontR >= 0.0f;
        float fr = visor ? frontR : r * shade[face];
        float fg = visor ? frontG : g * shade[face];
        float fb = visor ? frontB : b * shade[face];
        for (int t = 0; t < 2; t++) {
            rawTriangles3D tri;
            tri.vertex1 = c[faces[face][t * 3 + 0]];
            tri.vertex2 = c[faces[face][t * 3 + 1]];
            tri.vertex3 = c[faces[face][t * 3 + 2]];
            tri.r = fr; tri.g = fg; tri.b = fb;
            tris.push_back(tri);
        }
    }
}

static void addCenteredPart(std::vector<rawTriangles3D>& tris, int part,
    float taperX, float taperZ, float r, float g, float b,
    float frontR = -1.0f, float frontG = 0.0f, float frontB = 0.0f) {
    const PartSpec& spec = PART_SPECS[part];
    size_t first = tris.size();
    addShadedBox(tris, spec.x0, spec.x1, spec.y0, spec.y1, spec.z0, spec.z1,
        taperX, taperZ, r, g, b, frontR, frontG, frontB);
    float centerY = (spec.y0 + spec.y1) * 0.5f;
    float centerZ = (spec.z0 + spec.z1) * 0.5f;
    for (size_t i = first; i < tris.size(); i++) {
        tris[i].vertex1.y -= centerY; tris[i].vertex1.z -= centerZ;
        tris[i].vertex2.y -= centerY; tris[i].vertex2.z -= centerZ;
        tris[i].vertex3.y -= centerY; tris[i].vertex3.z -= centerZ;
    }
}

static void createRobotGeometry(std::vector<rawTriangles3D>& tris) {
    addCenteredPart(tris, PART_PELVIS, 0.05f, 0.03f, 0.24f, 0.27f, 0.36f);
    addCenteredPart(tris, PART_TORSO, -0.13f, -0.03f, 0.32f, 0.44f, 0.72f);
    addCenteredPart(tris, PART_HEAD, 0.03f, 0.02f, 0.86f, 0.86f, 0.90f, 0.05f, 0.35f, 0.75f);
    addCenteredPart(tris, PART_LUPPERLEG, 0.04f, 0.03f, 0.28f, 0.32f, 0.55f);
    addCenteredPart(tris, PART_LLOWERLEG, 0.03f, 0.03f, 0.20f, 0.23f, 0.40f);
    addCenteredPart(tris, PART_LFOOT, 0.00f, 0.02f, 0.13f, 0.15f, 0.22f);
    addCenteredPart(tris, PART_RUPPERLEG, 0.04f, 0.03f, 0.28f, 0.32f, 0.55f);
    addCenteredPart(tris, PART_RLOWERLEG, 0.03f, 0.03f, 0.20f, 0.23f, 0.40f);
    addCenteredPart(tris, PART_RFOOT, 0.00f, 0.02f, 0.13f, 0.15f, 0.22f);
    addCenteredPart(tris, PART_LUPPERARM, 0.035f, 0.03f, 0.72f, 0.32f, 0.24f);
    addCenteredPart(tris, PART_LFOREARM, 0.03f, 0.03f, 0.82f, 0.42f, 0.30f);
    addCenteredPart(tris, PART_LHAND, 0.02f, 0.02f, 0.16f, 0.17f, 0.24f);
    addCenteredPart(tris, PART_RUPPERARM, 0.035f, 0.03f, 0.72f, 0.32f, 0.24f);
    addCenteredPart(tris, PART_RFOREARM, 0.03f, 0.03f, 0.82f, 0.42f, 0.30f);
    addCenteredPart(tris, PART_RHAND, 0.02f, 0.02f, 0.16f, 0.17f, 0.24f);
}

__device__ inline float3 add3(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline float3 scale3(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }

__device__ float3 transformVertex(const RobotRenderTransform& transform, vertex3d v, float scale) {
    return add3(transform.origin, add3(scale3(transform.axisX, v.x * scale),
        add3(scale3(transform.axisY, v.y * scale), scale3(transform.axisZ, v.z * scale))));
}

__global__ void renderRobotKernel(const rawTriangles3D* geometry,
    const RobotRenderTransform* transforms, InteropRawTriangle3D* output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= TOTAL_TRIS) return;
    int part = id / TRIS_PER_PART;
    rawTriangles3D tri = geometry[id];
    float scale = robotScale > 0.001f ? robotScale : 1.0f;
    float3 a = transformVertex(transforms[part], tri.vertex1, scale);
    float3 b = transformVertex(transforms[part], tri.vertex2, scale);
    float3 c = transformVertex(transforms[part], tri.vertex3, scale);
    InteropRawTriangle3D rendered = {};
    rendered.data[0] = a.x; rendered.data[1] = a.y; rendered.data[2] = a.z;
    rendered.data[3] = tri.r; rendered.data[4] = tri.g; rendered.data[5] = tri.b;
    rendered.data[6] = b.x; rendered.data[7] = b.y; rendered.data[8] = b.z;
    rendered.data[9] = tri.r; rendered.data[10] = tri.g; rendered.data[11] = tri.b;
    rendered.data[12] = c.x; rendered.data[13] = c.y; rendered.data[14] = c.z;
    rendered.data[15] = tri.r; rendered.data[16] = tri.g; rendered.data[17] = tri.b;
    output[id] = rendered;
}

void initrobot() {
    std::vector<rawTriangles3D> geometry;
    createRobotGeometry(geometry);
    cudaMalloc(&d_robot_geom, TOTAL_TRIS * sizeof(rawTriangles3D));
    cudaMalloc(&d_robot_transforms, sizeof(h_robot_transforms));
    cudaMemcpy(d_robot_geom, geometry.data(), TOTAL_TRIS * sizeof(rawTriangles3D), cudaMemcpyHostToDevice);

    // noRender owns the OpenGL VBO; create it first, then register that same
    // storage with CUDA so the render kernel can write directly into it.
    render.rawTriangleBatchInterop3D(TOTAL_TRIS, ROBOT_INTEROP_ID);
    cudaError_t interopError = cudaGraphicsGLRegisterBuffer(
        &g_robotInteropResource, rawTri3dBatchVBO[ROBOT_INTEROP_ID],
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (interopError != cudaSuccess) {
        std::fprintf(stderr, "robot interop registration failed: %s\n",
            cudaGetErrorString(interopError));
    } else {
        g_robotInteropReady = true;
    }

    cudaMemcpyToSymbol(robotX, &h_robotX, sizeof(float));
    cudaMemcpyToSymbol(robotY, &h_robotY, sizeof(float));
    cudaMemcpyToSymbol(robotZ, &h_robotZ, sizeof(float));
    cudaMemcpyToSymbol(leftShoulderJointSideways, &h_leftShoulderJointSideways, sizeof(float));
    cudaMemcpyToSymbol(rightShoulderJointSideways, &h_rightShoulderJointSideways, sizeof(float));
    cudaMemcpyToSymbol(leftShoulderJointTwist, &h_leftShoulderJointTwist, sizeof(float));
    cudaMemcpyToSymbol(rightShoulderJointTwist, &h_rightShoulderJointTwist, sizeof(float));
    cudaMemcpyToSymbol(leftHipJointSideways, &h_leftHipJointSideways, sizeof(float));
    cudaMemcpyToSymbol(rightHipJointSideways, &h_rightHipJointSideways, sizeof(float));
    cudaMemcpyToSymbol(leftHipJointTwist, &h_leftHipJointTwist, sizeof(float));
    cudaMemcpyToSymbol(rightHipJointTwist, &h_rightHipJointTwist, sizeof(float));
    cudaMemcpyToSymbol(robotMoveX, &h_robotMoveX, sizeof(float));
    cudaMemcpyToSymbol(robotMoveY, &h_robotMoveY, sizeof(float));
    cudaMemcpyToSymbol(robotMoveZ, &h_robotMoveZ, sizeof(float));
    cudaMemcpyToSymbol(robotScale, &h_robotScale, sizeof(float));
    cudaMemcpyToSymbol(gravity, &h_gravity, sizeof(float));
    cudaMemcpyToSymbol(friction, &h_friction, sizeof(float));
    cudaMemcpyToSymbol(drag, &h_drag, sizeof(float));
    cudaMemcpyToSymbol(bounce, &h_bounce, sizeof(float));
    cudaMemcpyToSymbol(d_floorY, &floorY, sizeof(float));
    createBulletWorld();
    g_shapeScale = safeScale(h_robotScale);
    updateRenderTransforms();
    writeBulletStateToDevice();
    writeContactFlagsToDevice();
    g_lastPhysicsUpdate = std::chrono::steady_clock::now();
    g_havePhysicsUpdateTime = true;
}

void updaterobot() {
    if (!g_world) return;
    pullJointTargetsFromDevice();
    updateBulletMaterial();
    updateBulletScale();
    const auto now = std::chrono::steady_clock::now();
    float frameDt = g_havePhysicsUpdateTime
        ? std::chrono::duration<float>(now - g_lastPhysicsUpdate).count()
        : 1.0f / 60.0f;
    g_lastPhysicsUpdate = now;
    g_havePhysicsUpdateTime = true;
    frameDt = clampf(frameDt, 1.0f / 240.0f, 0.05f);

    float fixedDt = 1.0f / 120.0f;
    cudaMemcpyFromSymbol(&fixedDt, deltaTime, sizeof(float));
    fixedDt = clampf(fixedDt, 0.0005f, 0.05f);
    if (h_ragdollTimer > 0.0f)
        h_ragdollTimer = std::max(0.0f, h_ragdollTimer - frameDt);
    driveBulletJoints(fixedDt);
    applyRobotMovement(frameDt);
    // Advance by wall-clock time while retaining a stable fixed substep.
    // This prevents a 60 Hz render loop from making a 120 Hz robot fall at
    // half speed.
    g_world->stepSimulation(frameDt, 8, fixedDt);
    writeBulletStateToDevice();
    writeContactFlagsToDevice();
    updateRenderTransforms();
    InteropRawTriangle3D* d_interopOutput = nullptr;
    if (mapRobotInteropBuffer(d_interopOutput)) {
        int blocks = (TOTAL_TRIS + 255) / 256;
        renderRobotKernel<<<blocks, 256>>>(d_robot_geom, d_robot_transforms, d_interopOutput);
        cudaDeviceSynchronize();
        unmapRobotInteropBuffer();
    }
}

void renderRobot() { render.rawTriangleBatchInterop3D(TOTAL_TRIS, ROBOT_INTEROP_ID); }

void readRobotRoot(float& x, float& y, float& z) {
    cudaMemcpyFromSymbol(&x, robotX, sizeof(float));
    cudaMemcpyFromSymbol(&y, robotY, sizeof(float));
    cudaMemcpyFromSymbol(&z, robotZ, sizeof(float));
}

void readRobotContacts(bool* outContacts, int count) {
    if (!outContacts || count <= 0) return;
    int n = std::min(count, ROBOT_PART_COUNT);
    cudaMemcpyFromSymbol(outContacts, robotPartTouchingGround, n * sizeof(bool), 0, cudaMemcpyDeviceToHost);
}

void syncvar(int id) {
    switch (id) {
    case 0:
        cudaMemcpyToSymbol(d_floorY, &floorY, sizeof(float));
        if (g_groundBody) {
            btTransform transform = g_groundBody->getWorldTransform();
            transform.setOrigin(btVector3(floorX, floorY - 0.5f, floorZ));
            g_groundBody->setWorldTransform(transform);
        }
        break;
    case 1:
        cudaMemcpyToSymbol(gravity, &h_gravity, sizeof(float));
        if (g_world) g_world->setGravity(btVector3(0, h_gravity, 0));
        break;
    case 2: cudaMemcpyToSymbol(friction, &h_friction, sizeof(float)); break;
    case 3: cudaMemcpyToSymbol(drag, &h_drag, sizeof(float)); break;
    case 4: cudaMemcpyToSymbol(bounce, &h_bounce, sizeof(float)); break;
    case 5:
        cudaMemcpyToSymbol(robotScale, &h_robotScale, sizeof(float));
        updateBulletScale();
        break;
    case 6: cudaMemcpyToSymbol(robotX, &h_robotX, sizeof(float)); teleportRobotToHostRoot(); break;
    case 7: cudaMemcpyToSymbol(robotY, &h_robotY, sizeof(float)); teleportRobotToHostRoot(); break;
    case 8: cudaMemcpyToSymbol(robotZ, &h_robotZ, sizeof(float)); teleportRobotToHostRoot(); break;
    case 9: cudaMemcpyToSymbol(leftShoulderJoint, &h_leftShoulderJoint, sizeof(float)); break;
    case 10: cudaMemcpyToSymbol(rightShoulderJoint, &h_rightShoulderJoint, sizeof(float)); break;
    case 11: cudaMemcpyToSymbol(leftElbowJoint, &h_leftElbowJoint, sizeof(float)); break;
    case 12: cudaMemcpyToSymbol(rightElbowJoint, &h_rightElbowJoint, sizeof(float)); break;
    case 13: cudaMemcpyToSymbol(hipjoints, &h_hipjoints, sizeof(float)); break;
    case 14: cudaMemcpyToSymbol(hipJointSideways, &h_hipJointSideways, sizeof(float)); break;
    case 15: cudaMemcpyToSymbol(leftUpperLegJoint, &h_leftUpperLegJoint, sizeof(float)); break;
    case 16: cudaMemcpyToSymbol(rightUpperLegJoint, &h_rightUpperLegJoint, sizeof(float)); break;
    case 17: cudaMemcpyToSymbol(leftKneeJoint, &h_leftKneeJoint, sizeof(float)); break;
    case 18: cudaMemcpyToSymbol(rightKneeJoint, &h_rightKneeJoint, sizeof(float)); break;
    case 19: cudaMemcpyToSymbol(g_ragdollTimer, &h_ragdollTimer, sizeof(float)); break;
    case 20: cudaMemcpyToSymbol(g_muscleScale, &h_muscleStrength, sizeof(float)); break;
    case 21: cudaMemcpyToSymbol(leftShoulderJointSideways, &h_leftShoulderJointSideways, sizeof(float)); break;
    case 22: cudaMemcpyToSymbol(rightShoulderJointSideways, &h_rightShoulderJointSideways, sizeof(float)); break;
    case 23: cudaMemcpyToSymbol(leftShoulderJointTwist, &h_leftShoulderJointTwist, sizeof(float)); break;
    case 24: cudaMemcpyToSymbol(rightShoulderJointTwist, &h_rightShoulderJointTwist, sizeof(float)); break;
    case 25: cudaMemcpyToSymbol(leftHipJointSideways, &h_leftHipJointSideways, sizeof(float)); break;
    case 26: cudaMemcpyToSymbol(rightHipJointSideways, &h_rightHipJointSideways, sizeof(float)); break;
    case 27: cudaMemcpyToSymbol(leftHipJointTwist, &h_leftHipJointTwist, sizeof(float)); break;
    case 28: cudaMemcpyToSymbol(rightHipJointTwist, &h_rightHipJointTwist, sizeof(float)); break;
    case 29: cudaMemcpyToSymbol(robotMoveX, &h_robotMoveX, sizeof(float)); break;
    case 30: cudaMemcpyToSymbol(robotMoveY, &h_robotMoveY, sizeof(float)); break;
    case 31: cudaMemcpyToSymbol(robotMoveZ, &h_robotMoveZ, sizeof(float)); break;
    }
}
