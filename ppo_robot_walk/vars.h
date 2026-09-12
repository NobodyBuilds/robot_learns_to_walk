#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "render.h"
#include "norender.h"

inline int threads = 256;

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
inline int sample_robot_count = 1;
struct body {
	float positonX = 0.0f;
	float positionY = 0.0f;
	float positonZ = 0.0f;

	float velX = 0.0f;
	float velY = 0.0f;
	float velZ = 0.0f;

	float angleVelX = 0.0f;
	float angleVelY = 0.0f;
	float angleVelZ = 0.0f;

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

	bool alive = true;
	bool reached = false;

};




inline __device__ float gravity = -9.8f;
inline __device__ float friction = 0.9f;
inline __device__ float drag = 0.98f;
inline __device__ float bounce = 0.0f;
inline __device__ float deltaTime = 1.0f/120.0f;

inline __device__ float robotScale = 3.0f;


// Host variables for UI
inline float h_robotX = 0.0f;
inline float h_robotY = 12.0f;
inline float h_robotZ = 0.0f;
inline float rotx = 0.0f;
inline float roty = 0.0f;

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
inline float h_hipjoints = -20.0f;
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

inline float targetx = 246;
inline float targetz = 250.0f;
inline float targetradious = 46.0f;
inline float trx = 90.0f;
inline float Try = 0.0f;
inline float spawnx = 0.0f;
inline float spawnz = 0.0f;
inline float maxdisttotarget = 0.0f;
inline float mloss = 0.0f;
inline float oldmloss = 0.0f;
inline float rollout_time = 0.0f;
inline float lr = 0.001f;
inline float gentime = 20.0f;
inline float timer = 0.0f;
inline float fpsTimer = 0.0f;
inline float fpsCount = 0;
inline float fps = 0;
inline float avgFps = 0;
inline float dt = 1.0f / 120.0f;
inline int inputs = 40;
inline int output = 18;
inline int actor_layers;
inline int critic_layers;
inline int rollout_epoch = 2;
inline int adam_step = 1;
inline bool training = false;
inline bool run_ai = false;

struct replaybuffer {
	float s1[40];

	float reward;
	float logprob;
	float old_logprob;
	float value;
	float rtg;
	float advantage;
	float action[18];
	float actionZ[18];

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
inline int replaybuffersize = robot_count * 2048;
inline int gen = 0;
inline int step = 0;
inline int rolloutstep = 0;
inline int hbatchsize = 128;
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
inline float2* actor_adam_weights = nullptr;//x== m,y==v
inline float2* actor_adam_bias = nullptr;//x== m,y==v
inline float2* critic_adam_weights = nullptr;
inline float2* critic_adam_bias = nullptr;


inline std::vector<int> shuffled_indices;


inline int blocks(int n) {
	return (n + threads - 1) / threads;
};
//rewards  
inline float alive = 1.0f;
inline float dead = -10.0f;
inline float win = 25.0f;
inline float reachtarget = 1.0f;
inline float feettouching = 1.0f;
inline float handtouching = -1.0f;
inline float headtouching = -0.0f;
inline float torsotouching = -0.0f;



inline float x1 = -10.0f;
inline float Y1 = 10.0f;
inline float z1 = 0.0f;

inline float x2 = 10.0f;
inline float y2 = 10.0f;
inline float z2 = 0.0f;

inline float x3 = 10.0f;
inline float y3 = -10.0f;
inline float z3 = 0.0f;

inline float x4 = -10.0f;
inline float y4 = -10.0f;
inline float z4 = 0.0f;
