#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "render.h"
#include "norender.h"

//floor variables
inline std::vector<quadtexture2d> floorquads;
inline float floorwidth = 1000.0f;
inline float floorheight = 1000.0f;
inline float pixelWidth = 15.3f;
inline float pixelx = floorwidth / pixelWidth;
inline float pixely = floorheight / pixelWidth;
inline float floorY = 0.0f;
inline float floorZ = 0.0f;
inline float floorX = 0.0f;
inline float floorRotationX = 90.0f;
inline float floorRotationY = 0.0f;
/////
//robot variables
// Number of robot instances reserved for the multi-robot environment.
// The current renderer/physics path simulates one instance; keep this at 1
// until batched state and geometry buffers are added.
inline int robot_count = 1;
inline __device__ float robotX = 0.0f;
inline __device__ float robotY = 0.0f;
inline __device__ float robotZ = 0.0f;

// 3D locomotion command in the robot's local frame. Values are acceleration
// requests in m/s^2 and are applied through Bullet, so gravity and contacts
// still affect the resulting motion.
inline __device__ float robotMoveX = 0.0f;
inline __device__ float robotMoveY = 0.0f;
inline __device__ float robotMoveZ = 0.0f;
//
////joints values
inline __device__ float leftShoulderJoint = 0.0f;
inline __device__ float rightShoulderJoint = 0.0f;
inline __device__ float leftShoulderJointSideways = 0.0f;
inline __device__ float rightShoulderJointSideways = 0.0f;
inline __device__ float leftShoulderJointTwist = 0.0f;
inline __device__ float rightShoulderJointTwist = 0.0f;

inline __device__ float leftElbowJoint = 0.0f;
inline __device__ float rightElbowJoint = 0.0f;

inline __device__ float hipjoints = 0.0f;
inline __device__ float hipJointSideways = 0.0f;

inline __device__ float leftUpperLegJoint = 0.0f;
inline __device__ float rightUpperLegJoint = 0.0f;
inline __device__ float leftHipJointSideways = 0.0f;
inline __device__ float rightHipJointSideways = 0.0f;
inline __device__ float leftHipJointTwist = 0.0f;
inline __device__ float rightHipJointTwist = 0.0f;

inline __device__ float leftKneeJoint = 0.0f;
inline __device__ float rightKneeJoint = 0.0f;


inline constexpr int ROBOT_PART_COUNT = 15;
inline __device__ bool robotPartTouchingGround[ROBOT_PART_COUNT] = {};
inline __device__ bool robotAnyPartTouchingGround = false;


inline __device__ bool robotPelvisTouchingGround = false;
inline __device__ bool robotTorsoTouchingGround = false;
inline __device__ bool robotHeadTouchingGround = false;
inline __device__ bool robotLeftUpperLegTouchingGround = false;
inline __device__ bool robotLeftLowerLegTouchingGround = false;
inline __device__ bool robotLeftFootTouchingGround = false;
inline __device__ bool robotRightUpperLegTouchingGround = false;
inline __device__ bool robotRightLowerLegTouchingGround = false;
inline __device__ bool robotRightFootTouchingGround = false;
inline __device__ bool robotLeftUpperArmTouchingGround = false;
inline __device__ bool robotLeftForearmTouchingGround = false;
inline __device__ bool robotLeftHandTouchingGround = false;
inline __device__ bool robotRightUpperArmTouchingGround = false;
inline __device__ bool robotRightForearmTouchingGround = false;
inline __device__ bool robotRightHandTouchingGround = false;

//physics variables
inline __device__ float robotVelX = 0.0f;
inline __device__ float robotVelY = 0.0f;
inline __device__ float robotVelZ = 0.0f;

inline __device__ float robotRotX = 0.0f;
inline __device__ float robotRotY = 0.0f;
inline __device__ float robotRotZ = 0.0f;

inline __device__ float robotAngularVelX = 0.0f;
inline __device__ float robotAngularVelY = 0.0f;
inline __device__ float robotAngularVelZ = 0.0f;

inline __device__ float gravity = -9.8f;
inline __device__ float friction = 0.9f;
inline __device__ float drag = 0.98f;
inline __device__ float bounce = 0.0f;
inline __device__ float deltaTime = 1.0f/120.0f;

inline __device__ float robotScale = 1.0f;
inline __device__ float d_floorY = 0.0f;

// Host variables for UI
inline float h_robotX = 0.0f;
inline float h_robotY = 3.0f;
inline float h_robotZ = 0.0f;

inline float h_robotMoveX = 0.0f;
inline float h_robotMoveY = 0.0f;
inline float h_robotMoveZ = 0.0f;

inline float h_leftShoulderJoint = 0.0f;
inline float h_rightShoulderJoint = 0.0f;
inline float h_leftShoulderJointSideways = 0.0f;
inline float h_rightShoulderJointSideways = 0.0f;
inline float h_leftShoulderJointTwist = 0.0f;
inline float h_rightShoulderJointTwist = 0.0f;
inline float h_leftElbowJoint = 0.0f;
inline float h_rightElbowJoint = 0.0f;
inline float h_hipjoints = 0.0f;
inline float h_hipJointSideways = 0.0f;
inline float h_leftUpperLegJoint = 0.0f;
inline float h_rightUpperLegJoint = 0.0f;
inline float h_leftHipJointSideways = 0.0f;
inline float h_rightHipJointSideways = 0.0f;
inline float h_leftHipJointTwist = 0.0f;
inline float h_rightHipJointTwist = 0.0f;
inline float h_leftKneeJoint = 0.0f;
inline float h_rightKneeJoint = 0.0f;

inline float h_gravity = -9.8f;
inline float h_friction = 0.9f;
inline float h_drag = 0.98f;
inline float h_bounce = 0.0f;
inline float h_robotScale = 1.0f;

/////
