#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "render.h"
#include "norender.h"

int threads = 256;

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
// Number of independent robot instances in the environment.
inline int robot_count = 1;
struct body {
	float positonX = 0.0f;
	float positionY = 0.0f;
	float positonZ = 0.0f;

    float leftShoulderJoint = 0.0f;
    float leftShoulderJointSideways = 0.0f;
    float leftShoulderJointTwist = 0.0f;
    float leftElbowJoint = 0.0f;
	float rightShoulderJoint = 0.0f;
	float rightShoulderJointSideways = 0.0f;
	float rightShoulderJointTwist = 0.0f;
	float rightElbowJoint = 0.0f;
	float hipjoints = 0.0f;
	float rightHipJointSideways = 0.0f;
	float leftHipJointSideways = 0.0f;
	float leftHipJointTwist = 0.0f;


	float leftUpperLegJoint = 0.0f;
	float leftKneeJoint = 0.0f;
	float rightUpperLegJoint = 0.0f;
	float rightKneeJoint = 0.0f;


	float hipJointSideways = 0.0f;
	float rightHipJointTwist = 0.0f;

	bool robotPelvisTouchingGround = false;
	bool robotTorsoTouchingGround = false;
	bool robotHeadTouchingGround = false;
	bool robotLeftUpperLegTouchingGround = false;
	bool robotLeftLowerLegTouchingGround = false;
	bool robotLeftFootTouchingGround = false;
	bool robotRightUpperLegTouchingGround = false;
	bool robotRightLowerLegTouchingGround = false;
	bool robotRightFootTouchingGround = false;
	bool robotLeftUpperArmTouchingGround = false;
	bool robotLeftForearmTouchingGround = false;
	bool robotLeftHandTouchingGround = false;
	bool robotRightUpperArmTouchingGround = false;
	bool robotRightForearmTouchingGround = false;
	bool robotRightHandTouchingGround = false;

};

//inline __device__ float robotX = 0.0f;
//inline __device__ float robotY = 0.0f;
//inline __device__ float robotZ = 0.0f;
//// 3D locomotion command in the robot's local frame. Values are acceleration
//// requests in m/s^2 and are applied through Bullet, so gravity and contacts
//// still affect the resulting motion.
//
//inline __device__ float robotMoveX = 0.0f;
//inline __device__ float robotMoveY = 0.0f;
//inline __device__ float robotMoveZ = 0.0f;
//
////joints values
//inline __device__ float leftShoulderJoint = 0.0f;
//inline __device__ float leftShoulderJointSideways = 0.0f;
//inline __device__ float leftShoulderJointTwist = 0.0f;
//inline __device__ float leftElbowJoint = 0.0f;

//inline __device__ float rightShoulderJoint = 0.0f;
//inline __device__ float rightShoulderJointSideways = 0.0f;
//inline __device__ float rightShoulderJointTwist = 0.0f;
//inline __device__ float rightElbowJoint = 0.0f;


//inline __device__ float hipjoints = 0.0f;
//inline __device__ float rightHipJointSideways = 0.0f;
//inline __device__ float leftHipJointSideways = 0.0f;
//inline __device__ float leftHipJointTwist = 0.0f;
//
//
//inline __device__ float leftUpperLegJoint = 0.0f;
//inline __device__ float leftKneeJoint = 0.0f;
//inline __device__ float rightUpperLegJoint = 0.0f;
//inline __device__ float rightKneeJoint = 0.0f;
//
//
//
//inline __device__ float hipJointSideways = 0.0f;
//inline __device__ float rightHipJointTwist = 0.0f;
//
//
//inline constexpr int ROBOT_PART_COUNT = 15;
//inline __device__ bool robotPartTouchingGround[ROBOT_PART_COUNT] = {};
//inline __device__ bool robotAnyPartTouchingGround = false;
//
//
//inline __device__ bool robotPelvisTouchingGround = false;
//inline __device__ bool robotTorsoTouchingGround = false;
//inline __device__ bool robotHeadTouchingGround = false;
//inline __device__ bool robotLeftUpperLegTouchingGround = false;
//inline __device__ bool robotLeftLowerLegTouchingGround = false;
//inline __device__ bool robotLeftFootTouchingGround = false;
//inline __device__ bool robotRightUpperLegTouchingGround = false;
//inline __device__ bool robotRightLowerLegTouchingGround = false;
//inline __device__ bool robotRightFootTouchingGround = false;
//inline __device__ bool robotLeftUpperArmTouchingGround = false;
//inline __device__ bool robotLeftForearmTouchingGround = false;
//inline __device__ bool robotLeftHandTouchingGround = false;
//inline __device__ bool robotRightUpperArmTouchingGround = false;
//inline __device__ bool robotRightForearmTouchingGround = false;
//inline __device__ bool robotRightHandTouchingGround = false;



inline __device__ float gravity = -9.8f;
inline __device__ float friction = 0.9f;
inline __device__ float drag = 0.98f;
inline __device__ float bounce = 0.0f;
inline __device__ float deltaTime = 1.0f/120.0f;

inline __device__ float robotScale = 3.0f;
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
//env

float targetx = 500.0f;
float targetz = 250.0f;
float spawnx = 0.0f;
float spawnz = 0.0f;
float maxdisttotarget = 0.0f;
float mloss = 0.0f;
float oldmloss = 0.0f;
float rollout_time = 0.0f;

int inputs = 34;
int output = 18;
int actor_layers;
int critic_layers;
int rollout_epoch = 2;
int adam_step = 0;
bool training = true;

struct replaybuffer {
	float s1[34];

	float reward;
	float logprob;
	float old_logprob;
	float value;
	float rtg;
	float advantage;
	int action;

	bool done;
};
struct h_float2 {
	float x;
	float y;
};

struct Layer {
	int Nin;
	int Nout;
	int wIdx;
	int dIdx;
	int bIdx;

};


//ppo
int replaybuffersize = robot_count * 2048;
int gen = 0;
int step = 0;
int rolloutstep = 0;
int hbatchsize = 128;
inline float* d_actor_weights = nullptr;
inline float* d_critic_weights = nullptr;
inline float* d_actor_bias = nullptr;
inline float* d_critic_bias = nullptr;
inline body* d_body = nullptr;
inline replaybuffer* d_state = nullptr;
inline Layer* d_actlayer = nullptr;
inline Layer* d_critlayer = nullptr;
inline float* d_actor_nodvals = nullptr;
inline float* d_critic_nodvals = nullptr;
inline float* d_actor_preact = nullptr;
inline float* d_critic_preact = nullptr;
inline float* d_actor_delta = nullptr;
inline float* d_critic_delta = nullptr;
inline int* d_indices = nullptr;
inline float* d_antity_nodevals = nullptr;
inline float* actor_adam_weights = nullptr;
inline float* critic_adam_weights = nullptr;
inline float* actor_adam_bias = nullptr;
inline float* critic_adam_bias = nullptr;

inline std::vector<int> shuffled_indices;
