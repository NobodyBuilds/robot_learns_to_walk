#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "norender.h"
#define usecuda true
#define usecpu false
#define network true
#define DT 1.0f/120.0f
#define PI 3.14159265359f
#define startx 500.0f
#define degtorad PI/180.0f
#define radtodeg 180.0f/PI
#define maxjointstrength 2.0f
inline int threads = 256;
inline float dt = DT;
//floor variables
inline float floorx = 0.0f;
inline float floory = 50.0f;
inline float fsize = 200.0f;

//camera

inline float cx = 0.0f;
inline float cy = 0.0f;
inline float tx = 400.0f;

/////
//robot variables
// Number of independent robot instances in the environment.
struct part {
    float3 col;
    float angvel, mass, inertia, invmass, invinertia, torque;
    float angle;
    float2 pos;
    float2 size;
    float2 vel;
    float2 force;
    bool isstatic = false;
};

struct joint {
    part* parent;
    part* child;
    float2 parentanchor, childanchor;
    float targetangle, strength;
    float minangle, maxangle;
};
struct body {
    part torso;
    part leftthigh;
    part rightthigh;
    part leftshin;
    part rightshin;
    part leftfoot;
    part rightfoot;
    joint leftknee, rightknee;
    joint leftankle, rightankle;
    joint lefthip, righthip;
    bool torsotouchingground;
    bool leftfeettouching;
    bool rightfeettouching;
};
inline body* d_bodies = nullptr;
inline int robot_count =1;
inline int sample_robot_count = 1;
inline bool dummyQuad = false;
inline bool dummyCircle = false;





inline __device__ float clamp(float val, float min, float max) {
	return fminf(fmaxf(val, min), max);
}







/////
//env
#if network
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





inline std::vector<quadvertex2d> renderdata;



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
inline float3 operator/( float a, const float3& s) {
    return make_float3(
        a / s.x,
        a / s.y,
        a / s.z
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

