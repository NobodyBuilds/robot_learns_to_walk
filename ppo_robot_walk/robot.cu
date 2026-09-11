#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h>

#include "norender.h"
#include "renderdata.h"
#include "render.h"
#include "vars.h"
#include "network.h"

#define PI 3.14159265358979323846f
#define RAD(x) ((x) * PI / 180.0f)
#define NUM_PARTS 15
#define TRIS_PER_PART 12
#define TOTAL_TRIS (NUM_PARTS * TRIS_PER_PART)
#define ROBOT_INTEROP_ID 1

#define WAIST_MIN          -30.0f
#define WAIST_MAX           50.0f
#define HIP_MIN             -30.0f
#define HIP_MAX             100.0f
#define HIPROLL_MIN         -10.0f
#define HIPROLL_MAX          45.0f
#define KNEE_MIN              0.0f
#define KNEE_MAX            140.0f
#define SHOULDER_MIN        -60.0f
#define SHOULDER_MAX        170.0f
#define SHOULDER_SIDE_MIN   -90.0f
#define SHOULDER_SIDE_MAX    90.0f
#define SHOULDER_TWIST_MIN  -90.0f
#define SHOULDER_TWIST_MAX   90.0f
#define HIP_TWIST_MIN       -45.0f
#define HIP_TWIST_MAX        45.0f
#define ELBOW_MIN             0.0f
#define ELBOW_MAX           150.0f

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
    { -0.095f, 0.095f,  0.00f, 0.58f, -0.10f, 0.10f, A_LELBOW, A_LWRIST },
    { -0.085f, 0.085f,  0.60f, 0.88f, -0.08f, 0.08f, A_LELBOW, A_LWRIST },
    { -0.12f, 0.12f, -0.05f, 0.63f, -0.12f, 0.12f, A_RSHOULDER, A_RELBOW },
    { -0.095f, 0.095f,  0.00f, 0.58f, -0.10f, 0.10f, A_RELBOW, A_RWRIST },
    { -0.085f, 0.085f,  0.60f, 0.88f, -0.08f, 0.08f, A_RELBOW, A_RWRIST },
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

struct InteropRawTriangle3D {
    float data[18];
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

// Each entry owns an independent Bullet world. d_body stays a flat
// [robot_count] array, while contacts and constraints remain isolated.
struct RobotPhysics {
    btDiscreteDynamicsWorld* world = nullptr;
    btDefaultCollisionConfiguration* collisionConfig = nullptr;
    btCollisionDispatcher* dispatcher = nullptr;
    btBroadphaseInterface* broadphase = nullptr;
    btSequentialImpulseConstraintSolver* solver = nullptr;
    btRigidBody* groundBody = nullptr;
    btBoxShape* groundShape = nullptr;
    btGeneric6DofSpring2Constraint* waist = nullptr;
    BulletPart parts[NUM_PARTS] = {};
    std::vector<btTypedConstraint*> constraints;
    std::vector<HingeMotor> hingeMotors;
    std::vector<BallMotor> ballMotors;
    float shapeScale = 1.0f;
};

static std::vector<RobotPhysics> g_robots;
static std::vector<body> h_bodies;
static std::vector<unsigned char> g_pending_resets;
static int g_robot_count = 0;
static body* g_last_device_body = nullptr;
static int g_last_device_body_count = 0;
static bool g_device_body_initialized = false;

static cudaGraphicsResource* g_robotInteropResource = nullptr;
static bool g_robotInteropReady = false;
static int g_render_triangle_count = 0;
static rawTriangles3D* d_robot_geom = nullptr;
static std::vector<RobotRenderTransform> h_robot_transforms;
static RobotRenderTransform* d_robot_transforms = nullptr;
static std::chrono::steady_clock::time_point g_lastPhysicsUpdate;
static bool g_havePhysicsUpdateTime = false;

float h_ragdollTimer = 0.0f;
float h_muscleStrength = 1.0f;
__device__ float g_ragdollTimer = 0.0f;
__device__ float g_muscleScale = 1.0f;

static float safeScale(float scale) { return scale > 0.001f ? scale : 1.0f; }
static float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

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

static btTransform localFrame(btRigidBody* rigidBody, const btTransform& frame) {
    return rigidBody->getWorldTransform().inverse() * frame;
}

static void destroyRobotPhysics(RobotPhysics& robot) {
    if (robot.world) {
        for (btTypedConstraint* constraint : robot.constraints)
            robot.world->removeConstraint(constraint);
        for (btTypedConstraint* constraint : robot.constraints)
            delete constraint;
        robot.constraints.clear();
        robot.hingeMotors.clear();
        robot.ballMotors.clear();
        robot.waist = nullptr;

        for (BulletPart& part : robot.parts) {
            if (part.body) robot.world->removeRigidBody(part.body);
            delete part.body;
            delete part.shape;
            part = {};
        }
        if (robot.groundBody) robot.world->removeRigidBody(robot.groundBody);
        delete robot.groundBody;
        delete robot.groundShape;
    }

    delete robot.world;
    delete robot.solver;
    delete robot.broadphase;
    delete robot.dispatcher;
    delete robot.collisionConfig;
    robot = {};
}

static btRigidBody* makeRigidBody(RobotPhysics& robot, int part, float mass,
    const btVector3& center, const btMatrix3x3& basis, float scale) {
    const PartSpec& spec = PART_SPECS[part];
    btVector3 halfExtents((spec.x1 - spec.x0) * 0.5f,
        (spec.y1 - spec.y0) * 0.5f, (spec.z1 - spec.z0) * 0.5f);
    btBoxShape* shape = new btBoxShape(halfExtents);
    shape->setLocalScaling(btVector3(scale, scale, scale));
    shape->setMargin(0.015f);
    btVector3 inertia(0, 0, 0);
    shape->calculateLocalInertia(mass, inertia);
    btRigidBody::btRigidBodyConstructionInfo info(mass, nullptr, shape, inertia);
    btRigidBody* rigidBody = new btRigidBody(info);
    rigidBody->setWorldTransform(worldFrame(center, basis));
    rigidBody->setFriction(clampf(h_friction, 0.0f, 1.0f));
    rigidBody->setRestitution(clampf(h_bounce, 0.0f, 1.0f));
    rigidBody->setDamping(bulletLinearDamping(), 0.04f);
    rigidBody->setActivationState(DISABLE_DEACTIVATION);
    rigidBody->setUserIndex(part);
    robot.parts[part] = { rigidBody, shape };
    robot.world->addRigidBody(rigidBody);
    return rigidBody;
}

static void addFixedJoint(RobotPhysics& robot, btRigidBody* parent, btRigidBody* child,
    const btVector3& pivot) {
    btTransform frame = worldFrame(pivot, parent->getWorldTransform().getBasis());
    btFixedConstraint* joint = new btFixedConstraint(
        *parent, *child, localFrame(parent, frame), localFrame(child, frame));
    robot.world->addConstraint(joint, true);
    robot.constraints.push_back(joint);
}

static void addHingeJoint(RobotPhysics& robot, btRigidBody* parent, btRigidBody* child,
    const btVector3& pivot, const btVector3& axis, float low, float high, MotorInput input) {
    btTransform frame = worldFrame(pivot, basisAroundAxis(axis));
    btHingeConstraint* joint = new btHingeConstraint(
        *parent, *child, localFrame(parent, frame), localFrame(child, frame), true);
    joint->setLimit(RAD(low), RAD(high));
    joint->setMaxMotorImpulse(30.0f);
    joint->enableMotor(true);
    robot.world->addConstraint(joint, true);
    robot.constraints.push_back(joint);
    robot.hingeMotors.push_back({ joint, input, low, high });
}

static void addBallJoint(RobotPhysics& robot, btRigidBody* parent, btRigidBody* child,
    const btVector3& pivot, const btVector3& lower, const btVector3& upper,
    BallMotorInput input) {
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
    robot.world->addConstraint(joint, true);
    robot.constraints.push_back(joint);
    robot.ballMotors.push_back({ joint, input, lower, upper });
}

static void addWaistJoint(RobotPhysics& robot, btRigidBody* pelvis, btRigidBody* torso,
    const btVector3& pivot) {
    btTransform frame = worldFrame(pivot, btMatrix3x3::getIdentity());
    robot.waist = new btGeneric6DofSpring2Constraint(
        *pelvis, *torso, localFrame(pelvis, frame), localFrame(torso, frame));
    robot.waist->setLinearLowerLimit(btVector3(0, 0, 0));
    robot.waist->setLinearUpperLimit(btVector3(0, 0, 0));
    robot.waist->setAngularLowerLimit(btVector3(RAD(WAIST_MIN), 0.0f, RAD(HIPROLL_MIN)));
    robot.waist->setAngularUpperLimit(btVector3(RAD(WAIST_MAX), 0.0f, RAD(HIPROLL_MAX)));
    for (int axis = 3; axis <= 5; axis++) {
        robot.waist->enableMotor(axis, true);
        robot.waist->setServo(axis, true);
        robot.waist->setTargetVelocity(axis, 8.0f);
        robot.waist->setMaxMotorForce(axis, 40.0f);
    }
    robot.world->addConstraint(robot.waist, true);
    robot.constraints.push_back(robot.waist);
}

static void createRobotPhysics(RobotPhysics& robot, const btVector3& root) {
    destroyRobotPhysics(robot);
    robot.collisionConfig = new btDefaultCollisionConfiguration();
    robot.dispatcher = new btCollisionDispatcher(robot.collisionConfig);
    robot.broadphase = new btDbvtBroadphase();
    robot.solver = new btSequentialImpulseConstraintSolver();
    robot.world = new btDiscreteDynamicsWorld(robot.dispatcher, robot.broadphase,
        robot.solver, robot.collisionConfig);
    robot.world->setGravity(btVector3(0, h_gravity, 0));

    robot.groundShape = new btBoxShape(btVector3(floorwidth * 0.5f, 0.5f, floorheight * 0.5f));
    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(floorX, floorY - 0.5f, floorZ));
    btRigidBody::btRigidBodyConstructionInfo groundInfo(0.0f, nullptr, robot.groundShape);
    robot.groundBody = new btRigidBody(groundInfo);
    robot.groundBody->setWorldTransform(groundTransform);
    robot.groundBody->setFriction(clampf(h_friction, 0.0f, 1.0f));
    robot.groundBody->setRestitution(clampf(h_bounce, 0.0f, 1.0f));
    robot.groundBody->setUserIndex(-1);
    robot.world->addRigidBody(robot.groundBody);

    const float scale = safeScale(h_robotScale);
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
        makeRigidBody(robot, part, mass, center, basis, scale);
    }

    addWaistJoint(robot, robot.parts[PART_PELVIS].body, robot.parts[PART_TORSO].body,
        restPoint(A_PELVIS, root, scale));
    addFixedJoint(robot, robot.parts[PART_TORSO].body, robot.parts[PART_HEAD].body,
        restPoint(A_HEADBASE, root, scale));
    addBallJoint(robot, robot.parts[PART_PELVIS].body, robot.parts[PART_LUPPERLEG].body,
        restPoint(A_LHIP, root, scale),
        btVector3(RAD(HIP_MIN), RAD(HIPROLL_MIN), RAD(HIP_TWIST_MIN)),
        btVector3(RAD(HIP_MAX), RAD(HIPROLL_MAX), RAD(HIP_TWIST_MAX)), BALL_LEFT_HIP);
    addBallJoint(robot, robot.parts[PART_PELVIS].body, robot.parts[PART_RUPPERLEG].body,
        restPoint(A_RHIP, root, scale),
        btVector3(RAD(HIP_MIN), RAD(HIPROLL_MIN), RAD(HIP_TWIST_MIN)),
        btVector3(RAD(HIP_MAX), RAD(HIPROLL_MAX), RAD(HIP_TWIST_MAX)), BALL_RIGHT_HIP);
    addHingeJoint(robot, robot.parts[PART_LUPPERLEG].body, robot.parts[PART_LLOWERLEG].body,
        restPoint(A_LKNEE, root, scale), btVector3(1, 0, 0), KNEE_MIN, KNEE_MAX, MOTOR_LEFT_KNEE);
    addHingeJoint(robot, robot.parts[PART_RUPPERLEG].body, robot.parts[PART_RLOWERLEG].body,
        restPoint(A_RKNEE, root, scale), btVector3(1, 0, 0), KNEE_MIN, KNEE_MAX, MOTOR_RIGHT_KNEE);
    addFixedJoint(robot, robot.parts[PART_LLOWERLEG].body, robot.parts[PART_LFOOT].body,
        restPoint(A_LANKLE, root, scale));
    addFixedJoint(robot, robot.parts[PART_RLOWERLEG].body, robot.parts[PART_RFOOT].body,
        restPoint(A_RANKLE, root, scale));
    addBallJoint(robot, robot.parts[PART_TORSO].body, robot.parts[PART_LUPPERARM].body,
        restPoint(A_LSHOULDER, root, scale),
        btVector3(RAD(SHOULDER_MIN), RAD(SHOULDER_SIDE_MIN), RAD(SHOULDER_TWIST_MIN)),
        btVector3(RAD(SHOULDER_MAX), RAD(SHOULDER_SIDE_MAX), RAD(SHOULDER_TWIST_MAX)),
        BALL_LEFT_SHOULDER);
    addBallJoint(robot, robot.parts[PART_TORSO].body, robot.parts[PART_RUPPERARM].body,
        restPoint(A_RSHOULDER, root, scale),
        btVector3(RAD(SHOULDER_MIN), RAD(SHOULDER_SIDE_MIN), RAD(SHOULDER_TWIST_MIN)),
        btVector3(RAD(SHOULDER_MAX), RAD(SHOULDER_SIDE_MAX), RAD(SHOULDER_TWIST_MAX)),
        BALL_RIGHT_SHOULDER);
    addHingeJoint(robot, robot.parts[PART_LUPPERARM].body, robot.parts[PART_LFOREARM].body,
        restPoint(A_LELBOW, root, scale), btVector3(1, 0, 0), ELBOW_MIN, ELBOW_MAX,
        MOTOR_LEFT_ELBOW);
    addHingeJoint(robot, robot.parts[PART_RUPPERARM].body, robot.parts[PART_RFOREARM].body,
        restPoint(A_RELBOW, root, scale), btVector3(1, 0, 0), ELBOW_MIN, ELBOW_MAX,
        MOTOR_RIGHT_ELBOW);
    addFixedJoint(robot, robot.parts[PART_LFOREARM].body, robot.parts[PART_LHAND].body,
        restPoint(A_LWRIST, root, scale));
    addFixedJoint(robot, robot.parts[PART_RFOREARM].body, robot.parts[PART_RHAND].body,
        restPoint(A_RWRIST, root, scale));
    robot.shapeScale = scale;
}

static float motorTarget(const body& state, MotorInput input) {
    switch (input) {
    case MOTOR_LEFT_SHOULDER: return clampf(state.leftShoulderJoint, SHOULDER_MIN, SHOULDER_MAX);
    case MOTOR_RIGHT_SHOULDER: return clampf(state.rightShoulderJoint, SHOULDER_MIN, SHOULDER_MAX);
    case MOTOR_LEFT_ELBOW: return clampf(state.leftElbowJoint, ELBOW_MIN, ELBOW_MAX);
    case MOTOR_RIGHT_ELBOW: return clampf(state.rightElbowJoint, ELBOW_MIN, ELBOW_MAX);
    case MOTOR_LEFT_HIP: return clampf(state.leftUpperLegJoint, HIP_MIN, HIP_MAX);
    case MOTOR_RIGHT_HIP: return clampf(state.rightUpperLegJoint, HIP_MIN, HIP_MAX);
    case MOTOR_LEFT_KNEE: return clampf(state.leftKneeJoint, KNEE_MIN, KNEE_MAX);
    case MOTOR_RIGHT_KNEE: return clampf(state.rightKneeJoint, KNEE_MIN, KNEE_MAX);
    }
    return 0.0f;
}

static btVector3 ballMotorTarget(const body& state, BallMotorInput input) {
    switch (input) {
    case BALL_LEFT_SHOULDER:
        return btVector3(RAD(state.leftShoulderJoint), RAD(state.leftShoulderJointSideways),
            RAD(state.leftShoulderJointTwist));
    case BALL_RIGHT_SHOULDER:
        return btVector3(RAD(state.rightShoulderJoint), RAD(state.rightShoulderJointSideways),
            RAD(state.rightShoulderJointTwist));
    case BALL_LEFT_HIP:
        return btVector3(RAD(state.leftUpperLegJoint), RAD(state.leftHipJointSideways),
            RAD(state.leftHipJointTwist));
    case BALL_RIGHT_HIP:
        return btVector3(RAD(state.rightUpperLegJoint), RAD(state.rightHipJointSideways),
            RAD(state.rightHipJointTwist));
    }
    return btVector3(0, 0, 0);
}

static void driveBulletJoints(RobotPhysics& robot, const body& state, float dt) {
    bool motorsOn = h_muscleStrength > 0.001f && h_ragdollTimer <= 0.0f;
    for (const HingeMotor& motor : robot.hingeMotors) {
        motor.joint->enableMotor(motorsOn);
        if (motorsOn) {
            float target = clampf(motorTarget(state, motor.input), motor.low, motor.high);
            motor.joint->setMaxMotorImpulse(30.0f * clampf(h_muscleStrength, 0.0f, 2.0f));
            motor.joint->setMotorTarget(RAD(target), dt);
        }
    }

    for (const BallMotor& motor : robot.ballMotors) {
        for (int axis = 3; axis <= 5; axis++)
            motor.joint->enableMotor(axis, motorsOn);
        if (motorsOn) {
            btVector3 target = ballMotorTarget(state, motor.input);
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

    if (!robot.waist) return;
    robot.waist->enableMotor(3, motorsOn);
    robot.waist->enableMotor(5, motorsOn);
    if (motorsOn) {
        robot.waist->setServoTarget(3, RAD(clampf(state.hipjoints, WAIST_MIN, WAIST_MAX)));
        robot.waist->setServoTarget(5, RAD(clampf(state.hipJointSideways, HIPROLL_MIN, HIPROLL_MAX)));
        robot.waist->setMaxMotorForce(3, 40.0f * h_muscleStrength);
        robot.waist->setMaxMotorForce(5, 40.0f * h_muscleStrength);
    }
}

static void applyRobotMovement(RobotPhysics& robot, float frameDt) {
    btRigidBody* pelvis = robot.parts[PART_PELVIS].body;
    if (!pelvis || frameDt <= 0.0f) return;
    btVector3 localAcceleration(
        clampf(h_robotMoveX, -25.0f, 25.0f),
        clampf(h_robotMoveY, -25.0f, 25.0f),
        clampf(h_robotMoveZ, -25.0f, 25.0f));
    btVector3 worldAcceleration = pelvis->getWorldTransform().getBasis() * localAcceleration;
    const btScalar maxAcceleration = 25.0f;
    if (worldAcceleration.length2() > maxAcceleration * maxAcceleration)
        worldAcceleration = worldAcceleration.normalized() * maxAcceleration;
    pelvis->applyCentralImpulse(worldAcceleration * pelvis->getMass() * frameDt);
}

static void updateBulletMaterial(RobotPhysics& robot) {
    float frictionValue = clampf(h_friction, 0.0f, 1.0f);
    float restitutionValue = clampf(h_bounce, 0.0f, 1.0f);
    float damping = bulletLinearDamping();
    if (robot.groundBody) {
        robot.groundBody->setFriction(frictionValue);
        robot.groundBody->setRestitution(restitutionValue);
    }
    for (BulletPart& part : robot.parts) {
        if (!part.body) continue;
        part.body->setFriction(frictionValue);
        part.body->setRestitution(restitutionValue);
        part.body->setDamping(damping, 0.04f);
    }
}

static void updateBulletScale(RobotPhysics& robot) {
    float scale = safeScale(h_robotScale);
    if (std::fabs(scale - robot.shapeScale) < 1e-5f) return;
    for (BulletPart& part : robot.parts)
        if (part.shape) part.shape->setLocalScaling(btVector3(scale, scale, scale));
    robot.shapeScale = scale;
}

static btVector3 currentRobotRoot(const RobotPhysics& robot) {
    const btTransform& transform = robot.parts[PART_PELVIS].body->getWorldTransform();
    float scale = safeScale(h_robotScale);
    btVector3 localCenter(0,
        (PART_SPECS[PART_PELVIS].y0 + PART_SPECS[PART_PELVIS].y1) * 0.5f * scale,
        (PART_SPECS[PART_PELVIS].z0 + PART_SPECS[PART_PELVIS].z1) * 0.5f * scale);
    return transform * (-localCenter);
}

static void teleportRobotToRoot(RobotPhysics& robot, const btVector3& target) {
    if (!robot.world || !robot.parts[PART_PELVIS].body) return;
    btVector3 delta(target - currentRobotRoot(robot));
    for (BulletPart& part : robot.parts) {
        btTransform transform = part.body->getWorldTransform();
        transform.setOrigin(transform.getOrigin() + delta);
        part.body->setWorldTransform(transform);
        part.body->setLinearVelocity(btVector3(0, 0, 0));
        part.body->setAngularVelocity(btVector3(0, 0, 0));
        part.body->activate(true);
    }
}

static void setContactFlags(body& state, const bool contacts[NUM_PARTS]) {
    state.robotPelvisTouchingGround = contacts[PART_PELVIS];
    state.robotTorsoTouchingGround = contacts[PART_TORSO];
    state.robotHeadTouchingGround = contacts[PART_HEAD];
    state.robotLeftUpperLegTouchingGround = contacts[PART_LUPPERLEG];
    state.robotLeftLowerLegTouchingGround = contacts[PART_LLOWERLEG];
    state.robotLeftFootTouchingGround = contacts[PART_LFOOT];
    state.robotRightUpperLegTouchingGround = contacts[PART_RUPPERLEG];
    state.robotRightLowerLegTouchingGround = contacts[PART_RLOWERLEG];
    state.robotRightFootTouchingGround = contacts[PART_RFOOT];
    state.robotLeftUpperArmTouchingGround = contacts[PART_LUPPERARM];
    state.robotLeftForearmTouchingGround = contacts[PART_LFOREARM];
    state.robotLeftHandTouchingGround = contacts[PART_LHAND];
    state.robotRightUpperArmTouchingGround = contacts[PART_RUPPERARM];
    state.robotRightForearmTouchingGround = contacts[PART_RFOREARM];
    state.robotRightHandTouchingGround = contacts[PART_RHAND];
}

static void writeRobotStateToBody(const RobotPhysics& robot, body& state) {
    if (!robot.parts[PART_PELVIS].body) return;
    btVector3 root = currentRobotRoot(robot);
    state.positonX = float(root.x());
    state.positionY = float(root.y());
    state.positonZ = float(root.z());

    const btRigidBody* pelvis = robot.parts[PART_PELVIS].body;
    btVector3 linearVelocity = pelvis->getLinearVelocity();
    btVector3 angularVelocity = pelvis->getAngularVelocity();
    state.velX = float(linearVelocity.x());
    state.velY = float(linearVelocity.y());
    state.velZ = float(linearVelocity.z());
    state.angleVelX = float(angularVelocity.x());
    state.angleVelY = float(angularVelocity.y());
    state.angleVelZ = float(angularVelocity.z());

    bool contacts[NUM_PARTS] = {};
    btDispatcher* dispatcher = robot.world->getDispatcher();
    for (int i = 0; i < dispatcher->getNumManifolds(); i++) {
        btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(i);
        const btCollisionObject* objectA = manifold->getBody0();
        const btCollisionObject* objectB = manifold->getBody1();
        int part = -1;
        if (objectA->getUserIndex() >= 0 && objectA->getUserIndex() < NUM_PARTS &&
            objectB == robot.groundBody)
            part = objectA->getUserIndex();
        if (objectB->getUserIndex() >= 0 && objectB->getUserIndex() < NUM_PARTS &&
            objectA == robot.groundBody)
            part = objectB->getUserIndex();
        if (part < 0) continue;
        for (int p = 0; p < manifold->getNumContacts(); p++) {
            if (manifold->getContactPoint(p).getDistance() <= 0.02f) {
                contacts[part] = true;
                break;
            }
        }
    }
    setContactFlags(state, contacts);
}

static int activeRobotCount() {
    return std::max(1, robot_count);
}

static btVector3 initialRobotRoot(int index) {
    const float scale = safeScale(h_robotScale);
    const int columns = std::max(1, int(std::ceil(std::sqrt(float(g_robot_count)))));
    const float spacing = 6.0f * scale;
    return btVector3(h_robotX + float(index % columns) * spacing, h_robotY,
        h_robotZ + float(index / columns) * spacing);
}

static void fillInitialBody(body& state, int index) {
    btVector3 root = initialRobotRoot(index);
    state = {};
    state.positonX = float(root.x());
    state.positionY = float(root.y());
    state.positonZ = float(root.z());
    state.leftShoulderJoint = h_leftShoulderJoint;
    state.rightShoulderJoint = h_rightShoulderJoint;
    state.leftShoulderJointSideways = h_leftShoulderJointSideways;
    state.rightShoulderJointSideways = h_rightShoulderJointSideways;
    state.leftShoulderJointTwist = h_leftShoulderJointTwist;
    state.rightShoulderJointTwist = h_rightShoulderJointTwist;
    state.leftElbowJoint = h_leftElbowJoint;
    state.rightElbowJoint = h_rightElbowJoint;
    state.hipjoints = h_hipjoints;
    state.hipJointSideways = h_hipJointSideways;
    state.leftUpperLegJoint = h_leftUpperLegJoint;
    state.rightUpperLegJoint = h_rightUpperLegJoint;
    state.leftHipJointSideways = h_leftHipJointSideways;
    state.rightHipJointSideways = h_rightHipJointSideways;
    state.leftHipJointTwist = h_leftHipJointTwist;
    state.rightHipJointTwist = h_rightHipJointTwist;
    state.leftKneeJoint = h_leftKneeJoint;
    state.rightKneeJoint = h_rightKneeJoint;
}

static void mirrorFirstBodyToHost(const body& state) {
    h_leftShoulderJoint = state.leftShoulderJoint;
    h_rightShoulderJoint = state.rightShoulderJoint;
    h_leftShoulderJointSideways = state.leftShoulderJointSideways;
    h_rightShoulderJointSideways = state.rightShoulderJointSideways;
    h_leftShoulderJointTwist = state.leftShoulderJointTwist;
    h_rightShoulderJointTwist = state.rightShoulderJointTwist;
    h_leftElbowJoint = state.leftElbowJoint;
    h_rightElbowJoint = state.rightElbowJoint;
    h_hipjoints = state.hipjoints;
    h_hipJointSideways = state.hipJointSideways;
    h_leftUpperLegJoint = state.leftUpperLegJoint;
    h_rightUpperLegJoint = state.rightUpperLegJoint;
    h_leftHipJointSideways = state.leftHipJointSideways;
    h_rightHipJointSideways = state.rightHipJointSideways;
    h_leftHipJointTwist = state.leftHipJointTwist;
    h_rightHipJointTwist = state.rightHipJointTwist;
    h_leftKneeJoint = state.leftKneeJoint;
    h_rightKneeJoint = state.rightKneeJoint;
}

static bool syncBodiesFromDevice() {
    if (!d_body) return true;
    if (g_last_device_body != d_body || g_last_device_body_count != g_robot_count) {
        cudaError_t error = cudaMemcpy(d_body, h_bodies.data(),
            g_robot_count * sizeof(body), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "robot body initialization failed: %s\n",
                cudaGetErrorString(error));
            return false;
        }
        g_last_device_body = d_body;
        g_last_device_body_count = g_robot_count;
        g_device_body_initialized = true;
        return true;
    }

    cudaError_t error = cudaMemcpy(h_bodies.data(), d_body,
        g_robot_count * sizeof(body), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "robot body download failed: %s\n", cudaGetErrorString(error));
        return false;
    }
    g_device_body_initialized = true;
    if (!h_bodies.empty()) mirrorFirstBodyToHost(h_bodies[0]);
    return true;
}

static void syncBodiesToDevice() {
    if (!d_body || !g_device_body_initialized) return;
    cudaError_t error = cudaMemcpy(d_body, h_bodies.data(),
        g_robot_count * sizeof(body), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        std::fprintf(stderr, "robot body upload failed: %s\n", cudaGetErrorString(error));
}

static void writeBodyFieldToDevice(size_t offset, const void* value, size_t size) {
    if (!d_body || g_robot_count <= 0) return;
    cudaError_t error = cudaMemcpy(reinterpret_cast<char*>(d_body) + offset,
        value, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        std::fprintf(stderr, "robot body field upload failed: %s\n", cudaGetErrorString(error));
}

static void uploadResetBodies();

static void updateRenderTransforms() {
    if (h_robot_transforms.empty()) return;

    // d_body owns the rendered robot root.  The body struct does not contain
    // per-part orientations, so preserve Bullet's relative articulation while
    // translating the complete pose to the root stored in d_body.  This keeps
    // a CUDA reset visible without rebuilding an upright pose through a
    // fallen Bullet root.
    if (!syncBodiesFromDevice()) return;

    for (int i = 0; i < g_robot_count; i++) {
        const body& state = h_bodies[i];
        const RobotPhysics& robot = g_robots[i];
        if (!robot.parts[PART_PELVIS].body) continue;

        btVector3 bodyRoot(state.positonX, state.positionY, state.positonZ);
        btVector3 physicsRoot = currentRobotRoot(robot);
        btVector3 rootDelta = bodyRoot - physicsRoot;
        for (int part = 0; part < NUM_PARTS; part++) {
            const btTransform& transform = robot.parts[part].body->getWorldTransform();

            RobotRenderTransform& output = h_robot_transforms[i * NUM_PARTS + part];
            btVector3 center = transform.getOrigin() + rootDelta;
            output.origin = make_float3(float(center.x()), float(center.y()), float(center.z()));
            const btMatrix3x3& basis = transform.getBasis();
            btVector3 x = basis.getColumn(0), y = basis.getColumn(1), z = basis.getColumn(2);
            output.axisX = make_float3(float(x.x()), float(x.y()), float(x.z()));
            output.axisY = make_float3(float(y.x()), float(y.y()), float(y.z()));
            output.axisZ = make_float3(float(z.x()), float(z.y()), float(z.z()));
        }
    }
    cudaMemcpy(d_robot_transforms, h_robot_transforms.data(),
        h_robot_transforms.size() * sizeof(RobotRenderTransform), cudaMemcpyHostToDevice);
}

static void resetPendingRobots() {
    if (g_robot_count <= 0 || g_robots.empty() || h_bodies.empty() ||
        g_pending_resets.empty())
        return;

    bool resetNeeded = false;
    for (int i = 0; i < g_robot_count; i++) {
        if (!g_pending_resets[i]) continue;

        createRobotPhysics(g_robots[i], initialRobotRoot(i));
        fillInitialBody(h_bodies[i], i);
        writeRobotStateToBody(g_robots[i], h_bodies[i]);
        g_pending_resets[i] = 0;
        resetNeeded = true;
    }

    if (!resetNeeded) return;

    uploadResetBodies();
    updateRenderTransforms();
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

__device__ inline float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 scale3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 transformVertex(const RobotRenderTransform& transform, vertex3d v, float scale) {
    return add3(transform.origin, add3(scale3(transform.axisX, v.x * scale),
        add3(scale3(transform.axisY, v.y * scale), scale3(transform.axisZ, v.z * scale))));
}

__global__ void renderRobotKernel(const rawTriangles3D* geometry,
    const RobotRenderTransform* transforms, InteropRawTriangle3D* output, int robotCount) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = robotCount * TOTAL_TRIS;
    if (id >= total) return;
    int robotIndex = id / TOTAL_TRIS;
    int localId = id % TOTAL_TRIS;
    int part = localId / TRIS_PER_PART;
    rawTriangles3D tri = geometry[localId];
    float scale = robotScale > 0.001f ? robotScale : 1.0f;
    const RobotRenderTransform& transform = transforms[robotIndex * NUM_PARTS + part];
    float3 a = transformVertex(transform, tri.vertex1, scale);
    float3 b = transformVertex(transform, tri.vertex2, scale);
    float3 c = transformVertex(transform, tri.vertex3, scale);
    InteropRawTriangle3D rendered = {};
    rendered.data[0] = a.x; rendered.data[1] = a.y; rendered.data[2] = a.z;
    rendered.data[3] = tri.r; rendered.data[4] = tri.g; rendered.data[5] = tri.b;
    rendered.data[6] = b.x; rendered.data[7] = b.y; rendered.data[8] = b.z;
    rendered.data[9] = tri.r; rendered.data[10] = tri.g; rendered.data[11] = tri.b;
    rendered.data[12] = c.x; rendered.data[13] = c.y; rendered.data[14] = c.z;
    rendered.data[15] = tri.r; rendered.data[16] = tri.g; rendered.data[17] = tri.b;
    output[id] = rendered;
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
    if (error != cudaSuccess || !output || bytes < size_t(g_render_triangle_count) * sizeof(InteropRawTriangle3D)) {
        std::fprintf(stderr, "robot interop pointer failed: %s\n", cudaGetErrorString(error));
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

void initrobot() {
    g_robot_count = activeRobotCount();
    g_robots.resize(g_robot_count);
    h_bodies.resize(g_robot_count);
    g_pending_resets.assign(g_robot_count, 0);
    h_robot_transforms.resize(g_robot_count * NUM_PARTS);

    for (int i = 0; i < g_robot_count; i++) {
        fillInitialBody(h_bodies[i], i);
        createRobotPhysics(g_robots[i], initialRobotRoot(i));
    }

    std::vector<rawTriangles3D> geometry;
    createRobotGeometry(geometry);
    cudaMalloc(&d_robot_geom, TOTAL_TRIS * sizeof(rawTriangles3D));
    cudaMalloc(&d_robot_transforms,
        h_robot_transforms.size() * sizeof(RobotRenderTransform));
    cudaMemcpy(d_robot_geom, geometry.data(), TOTAL_TRIS * sizeof(rawTriangles3D),
        cudaMemcpyHostToDevice);

    g_render_triangle_count = g_robot_count * TOTAL_TRIS;
    render.rawTriangleBatchInterop3D(g_render_triangle_count, ROBOT_INTEROP_ID);
    cudaError_t interopError = cudaGraphicsGLRegisterBuffer(
        &g_robotInteropResource, rawTri3dBatchVBO[ROBOT_INTEROP_ID],
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (interopError != cudaSuccess) {
        std::fprintf(stderr, "robot interop registration failed: %s\n",
            cudaGetErrorString(interopError));
    } else {
        g_robotInteropReady = true;
    }

    cudaMemcpyToSymbol(robotScale, &h_robotScale, sizeof(float));
    cudaMemcpyToSymbol(gravity, &h_gravity, sizeof(float));
    cudaMemcpyToSymbol(friction, &h_friction, sizeof(float));
    cudaMemcpyToSymbol(drag, &h_drag, sizeof(float));
    cudaMemcpyToSymbol(bounce, &h_bounce, sizeof(float));
    cudaMemcpyToSymbol(d_floorY, &floorY, sizeof(float));
    updateRenderTransforms();
    for (int i = 0; i < g_robot_count; i++) writeRobotStateToBody(g_robots[i], h_bodies[i]);
    syncBodiesFromDevice();
    syncBodiesToDevice();
    g_lastPhysicsUpdate = std::chrono::steady_clock::now();
    g_havePhysicsUpdateTime = true;
}

void updaterobot() {
    if (g_robots.empty()) return;
    if (!syncBodiesFromDevice()) return;

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

    for (int i = 0; i < g_robot_count; i++) {
        RobotPhysics& robot = g_robots[i];
        updateBulletMaterial(robot);
        updateBulletScale(robot);
        driveBulletJoints(robot, h_bodies[i], fixedDt);
        applyRobotMovement(robot, frameDt);
        robot.world->stepSimulation(frameDt, 8, fixedDt);
        writeRobotStateToBody(robot, h_bodies[i]);
        if (h_bodies[i].robotPelvisTouchingGround ||
            h_bodies[i].robotHeadTouchingGround ||
            h_bodies[i].robotTorsoTouchingGround)
            h_bodies[i].alive = false;

        const float dx = h_bodies[i].positonX - targetx;
        const float dz = h_bodies[i].positonZ - targetz;
        const float targetRadius = std::max(0.0f, targetradious);
        if (dx * dx + dz * dz <= targetRadius * targetRadius)
            h_bodies[i].reached = true;

        if (!h_bodies[i].alive || h_bodies[i].reached)
            g_pending_resets[i] = 1;
    }

    syncBodiesToDevice();
    updateRenderTransforms();
    InteropRawTriangle3D* d_interopOutput = nullptr;
    if (mapRobotInteropBuffer(d_interopOutput)) {
        int blocks = (g_render_triangle_count + 255) / 256;
        renderRobotKernel<<<blocks, 256>>>(d_robot_geom, d_robot_transforms,
            d_interopOutput, g_robot_count);
        cudaDeviceSynchronize();
        unmapRobotInteropBuffer();
    }
}

static void uploadResetBodies() {
    if (!d_body || h_bodies.empty() || g_robot_count <= 0) return;
    cudaError_t error = cudaMemcpy(d_body, h_bodies.data(),
        g_robot_count * sizeof(body), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "robot reset upload failed: %s\n",
            cudaGetErrorString(error));
        return;
    }
    g_last_device_body = d_body;
    g_last_device_body_count = g_robot_count;
    g_device_body_initialized = true;
}

void resetrobot(int id) {
    if (id < 0 || id >= g_robot_count || id >= static_cast<int>(g_robots.size()) ||
        id >= static_cast<int>(h_bodies.size()))
        return;

    createRobotPhysics(g_robots[id], initialRobotRoot(id));
    fillInitialBody(h_bodies[id], id);
    writeRobotStateToBody(g_robots[id], h_bodies[id]);
    uploadResetBodies();
    updateRenderTransforms();
}

void resetrobots() {
    if (g_robot_count <= 0 || g_robots.empty() || h_bodies.empty()) return;

    const int count = std::min(g_robot_count,
        std::min(static_cast<int>(g_robots.size()), static_cast<int>(h_bodies.size())));
    for (int i = 0; i < count; i++) {
        createRobotPhysics(g_robots[i], initialRobotRoot(i));
        fillInitialBody(h_bodies[i], i);
        writeRobotStateToBody(g_robots[i], h_bodies[i]);
        g_pending_resets[i] = 0;
    }
    uploadResetBodies();
    updateRenderTransforms();
}

void renderRobot() {
    resetPendingRobots();
    updateRenderTransforms();
    render.rawTriangleBatchInterop3D(g_render_triangle_count, ROBOT_INTEROP_ID);
}

void readRobotRoot(float& x, float& y, float& z) {
    if (g_robot_count <= 0) return;
    syncBodiesFromDevice();
    x = h_bodies[0].positonX;
    y = h_bodies[0].positionY;
    z = h_bodies[0].positonZ;
}

void readRobotContacts(bool* outContacts, int count) {
    if (!outContacts || count <= 0 || g_robot_count <= 0) return;
    syncBodiesFromDevice();
    const body& state = h_bodies[0];
    bool contacts[NUM_PARTS] = {
        state.robotPelvisTouchingGround, state.robotTorsoTouchingGround,
        state.robotHeadTouchingGround, state.robotLeftUpperLegTouchingGround,
        state.robotLeftLowerLegTouchingGround, state.robotLeftFootTouchingGround,
        state.robotRightUpperLegTouchingGround, state.robotRightLowerLegTouchingGround,
        state.robotRightFootTouchingGround, state.robotLeftUpperArmTouchingGround,
        state.robotLeftForearmTouchingGround, state.robotLeftHandTouchingGround,
        state.robotRightUpperArmTouchingGround, state.robotRightForearmTouchingGround,
        state.robotRightHandTouchingGround
    };
    int n = std::min(count, NUM_PARTS);
    std::copy(contacts, contacts + n, outContacts);
}

static void updateFloorForAllRobots() {
    for (RobotPhysics& robot : g_robots) {
        if (!robot.groundBody) continue;
        btTransform transform = robot.groundBody->getWorldTransform();
        transform.setOrigin(btVector3(floorX, floorY - 0.5f, floorZ));
        robot.groundBody->setWorldTransform(transform);
    }
}

void syncvar(int id) {
    if (g_robot_count <= 0) return;
    switch (id) {
    case 0:
        cudaMemcpyToSymbol(d_floorY, &floorY, sizeof(float));
        updateFloorForAllRobots();
        break;
    case 1:
        cudaMemcpyToSymbol(gravity, &h_gravity, sizeof(float));
        for (RobotPhysics& robot : g_robots)
            if (robot.world) robot.world->setGravity(btVector3(0, h_gravity, 0));
        break;
    case 2: cudaMemcpyToSymbol(friction, &h_friction, sizeof(float)); break;
    case 3: cudaMemcpyToSymbol(drag, &h_drag, sizeof(float)); break;
    case 4: cudaMemcpyToSymbol(bounce, &h_bounce, sizeof(float)); break;
    case 5:
        cudaMemcpyToSymbol(robotScale, &h_robotScale, sizeof(float));
        for (RobotPhysics& robot : g_robots) updateBulletScale(robot);
        break;
    case 6:
        h_bodies[0].positonX = h_robotX;
        writeBodyFieldToDevice(offsetof(body, positonX), &h_bodies[0].positonX, sizeof(float));
        teleportRobotToRoot(g_robots[0], btVector3(h_robotX, h_bodies[0].positionY, h_bodies[0].positonZ));
        break;
    case 7:
        h_bodies[0].positionY = h_robotY;
        writeBodyFieldToDevice(offsetof(body, positionY), &h_bodies[0].positionY, sizeof(float));
        teleportRobotToRoot(g_robots[0], btVector3(h_bodies[0].positonX, h_robotY, h_bodies[0].positonZ));
        break;
    case 8:
        h_bodies[0].positonZ = h_robotZ;
        writeBodyFieldToDevice(offsetof(body, positonZ), &h_bodies[0].positonZ, sizeof(float));
        teleportRobotToRoot(g_robots[0], btVector3(h_bodies[0].positonX, h_bodies[0].positionY, h_robotZ));
        break;
    case 9:
        h_bodies[0].leftShoulderJoint = h_leftShoulderJoint;
        writeBodyFieldToDevice(offsetof(body, leftShoulderJoint), &h_bodies[0].leftShoulderJoint, sizeof(float));
        break;
    case 10:
        h_bodies[0].rightShoulderJoint = h_rightShoulderJoint;
        writeBodyFieldToDevice(offsetof(body, rightShoulderJoint), &h_bodies[0].rightShoulderJoint, sizeof(float));
        break;
    case 11:
        h_bodies[0].leftElbowJoint = h_leftElbowJoint;
        writeBodyFieldToDevice(offsetof(body, leftElbowJoint), &h_bodies[0].leftElbowJoint, sizeof(float));
        break;
    case 12:
        h_bodies[0].rightElbowJoint = h_rightElbowJoint;
        writeBodyFieldToDevice(offsetof(body, rightElbowJoint), &h_bodies[0].rightElbowJoint, sizeof(float));
        break;
    case 13:
        h_bodies[0].hipjoints = h_hipjoints;
        writeBodyFieldToDevice(offsetof(body, hipjoints), &h_bodies[0].hipjoints, sizeof(float));
        break;
    case 14:
        h_bodies[0].hipJointSideways = h_hipJointSideways;
        writeBodyFieldToDevice(offsetof(body, hipJointSideways), &h_bodies[0].hipJointSideways, sizeof(float));
        break;
    case 15:
        h_bodies[0].leftUpperLegJoint = h_leftUpperLegJoint;
        writeBodyFieldToDevice(offsetof(body, leftUpperLegJoint), &h_bodies[0].leftUpperLegJoint, sizeof(float));
        break;
    case 16:
        h_bodies[0].rightUpperLegJoint = h_rightUpperLegJoint;
        writeBodyFieldToDevice(offsetof(body, rightUpperLegJoint), &h_bodies[0].rightUpperLegJoint, sizeof(float));
        break;
    case 17:
        h_bodies[0].leftKneeJoint = h_leftKneeJoint;
        writeBodyFieldToDevice(offsetof(body, leftKneeJoint), &h_bodies[0].leftKneeJoint, sizeof(float));
        break;
    case 18:
        h_bodies[0].rightKneeJoint = h_rightKneeJoint;
        writeBodyFieldToDevice(offsetof(body, rightKneeJoint), &h_bodies[0].rightKneeJoint, sizeof(float));
        break;
    case 19: cudaMemcpyToSymbol(g_ragdollTimer, &h_ragdollTimer, sizeof(float)); break;
    case 20: cudaMemcpyToSymbol(g_muscleScale, &h_muscleStrength, sizeof(float)); break;
    case 21:
        h_bodies[0].leftShoulderJointSideways = h_leftShoulderJointSideways;
        writeBodyFieldToDevice(offsetof(body, leftShoulderJointSideways), &h_bodies[0].leftShoulderJointSideways, sizeof(float));
        break;
    case 22:
        h_bodies[0].rightShoulderJointSideways = h_rightShoulderJointSideways;
        writeBodyFieldToDevice(offsetof(body, rightShoulderJointSideways), &h_bodies[0].rightShoulderJointSideways, sizeof(float));
        break;
    case 23:
        h_bodies[0].leftShoulderJointTwist = h_leftShoulderJointTwist;
        writeBodyFieldToDevice(offsetof(body, leftShoulderJointTwist), &h_bodies[0].leftShoulderJointTwist, sizeof(float));
        break;
    case 24:
        h_bodies[0].rightShoulderJointTwist = h_rightShoulderJointTwist;
        writeBodyFieldToDevice(offsetof(body, rightShoulderJointTwist), &h_bodies[0].rightShoulderJointTwist, sizeof(float));
        break;
    case 25:
        h_bodies[0].leftHipJointSideways = h_leftHipJointSideways;
        writeBodyFieldToDevice(offsetof(body, leftHipJointSideways), &h_bodies[0].leftHipJointSideways, sizeof(float));
        break;
    case 26:
        h_bodies[0].rightHipJointSideways = h_rightHipJointSideways;
        writeBodyFieldToDevice(offsetof(body, rightHipJointSideways), &h_bodies[0].rightHipJointSideways, sizeof(float));
        break;
    case 27:
        h_bodies[0].leftHipJointTwist = h_leftHipJointTwist;
        writeBodyFieldToDevice(offsetof(body, leftHipJointTwist), &h_bodies[0].leftHipJointTwist, sizeof(float));
        break;
    case 28:
        h_bodies[0].rightHipJointTwist = h_rightHipJointTwist;
        writeBodyFieldToDevice(offsetof(body, rightHipJointTwist), &h_bodies[0].rightHipJointTwist, sizeof(float));
        break;
    case 29:
    case 30:
    case 31:
        break;
    }
}
