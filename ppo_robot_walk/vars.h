#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "render.h"
#include "norender.h"
#define usecuda true
#define usecpu false
#define network true
#define PI 3.14159265359f
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
inline int robot_count =1;
inline int sample_robot_count = 1;





inline __device__ float clamp(float val, float min, float max) {
	return fminf(fmaxf(val, min), max);
}


inline __device__ float friction = 0.9f;
inline  float dt = 1.0f/120.0f;





/////
//env
#if network
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
inline float h_dt = 1.0f / 120.0f;
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
#endif


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





//robot.cu data

struct rigidbody {

    bool isstatic;
    float mass;
    float invmass;
    float3 position;
    float3 velocity;
    float3 angularvel;
    float3 force;
    float3 torque;
    float3 inertia;
    float3 invinertia;
    float3 halfsize;
    float4 quatrotation;
   
};



struct robotbody {
    rigidbody head;
    rigidbody torso;

    rigidbody leftupperarm;
    rigidbody righttupperarm;

    rigidbody leftforearm;
    rigidbody rightforearm;

    rigidbody leftthigh;
    rigidbody rightthigh;

    rigidbody leftshin;
    rigidbody rightshin;
};

struct joint {
    float minAngle;
    float maxAngle;

    float targetAngle;
    float motorStrength;
   

    float3 parentanchor;
    float3 childanchor;

    float3 axis;
    rigidbody* parent;
    rigidbody* child;
    

};

inline std::vector<robotbody> bodydata;
inline std::vector<quadVertex3d> renderdata;


__host__ __device__
inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__
inline float3 operator*(const float3& a, float s) {
    return make_float3(
        a.x * s,
        a.y * s,
        a.z * s
    );
}

__host__ __device__
inline float3 operator*(float s, const float3& a) {
    return make_float3(
        a.x * s,
        a.y * s,
        a.z * s
    );
}

__host__ __device__
inline float3 operator/(const float3& a, float s) {
    return make_float3(
        a.x / s,
        a.y / s,
        a.z / s
    );
}

__host__ __device__
inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__
inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__
inline float3& operator*=(float3& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

__host__ __device__
inline float3& operator/=(float3& a, float s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
    return a;
}

__host__ __device__
inline float3 operator+(const float3& a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__host__ __device__
inline float3 operator+(float a, const float3& b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

__host__ __device__
inline float3 operator-(const float3& a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__host__ __device__
inline float3 operator-(float a, const float3& b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}

