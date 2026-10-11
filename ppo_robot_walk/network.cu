#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdio> 
#include <cmath>
#include <math.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <unordered_set>
#include "vars.h"
#include "render.h"
#include "network.h"
#include "norender.h"


struct device_data {
	int n;
	int layers;
	float invscreenh;
};
__constant__ device_data d;
device_data h;
void setconst() {
	h.n = robot_count;
	h.layers = actor_layers;
	h.invscreenh = 1.0f/noRender.getScreenHeight();

	cudaMemcpyToSymbol(d, &h, sizeof(device_data), 0);
}
void geterror(const std::string& label, cudaError_t err) {
	static std::unordered_set<std::string> printed;

	if (printed.find(label) == printed.end())
	{
		if (err != cudaSuccess) {
			printf("%s error : %s \n", label.c_str(), cudaGetErrorString(err));
			printed.insert(label);
		}
	}
}

curandState* d_rngstate;
constexpr float ACTION_SIGMA = 0.6f;

__global__ void init_rng(curandState* state, int n, unsigned long seed) {
	for (int i = 0; i < n; i++) {
		curand_init(seed, i, 0, &state[i]);
	}
}
void allocate() {
	
	

	cudaMalloc(&d_rngstate, robot_count * sizeof(curandState));
	init_rng << <1, 1 >> > (d_rngstate, robot_count, 3476);
	printf("data allocated \n");
}


//weights and netrwork

int actor_weightbuffersize = 0;
int critic_weightbuffersize = 0;
int actor_biassize = 0;
int critic_biassize = 0;
int actor_nodedatasize = 0;
int critic_nodedatasize = 0;
int antitynodesize = 0;
std::vector<Layer> actor_layerdata;
std::vector<Layer> critic_layerdata;
void addlayer(int in, int out, bool isactor) {
	Layer l;
	l.Nin = in;
	l.Nout = out;
	l.wIdx = 0;
	l.dIdx = 0;
	l.bIdx = 0;
	if (isactor) {
		actor_layerdata.push_back(l);
		actor_layers++;
		actor_weightbuffersize += (in * out);
		actor_nodedatasize += out;
		antitynodesize += out * robot_count;
		actor_biassize += out;
	}
	else {
		critic_layerdata.push_back(l);
		critic_layers++;
		critic_weightbuffersize += (in * out);
		critic_nodedatasize += out;
		critic_biassize += out;
	}

}

void setoffsets(bool isactor) {

	auto& layerdata = isactor ? actor_layerdata : critic_layerdata;

	for (int i = 0; i < layerdata.size(); i++) {
		int k = i - 1;
		layerdata[i].wIdx = i == 0 ? 0 : layerdata[k].wIdx + layerdata[k].Nin * layerdata[k].Nout;
		layerdata[i].dIdx = i == 0 ? 0 : layerdata[k].dIdx + layerdata[k].Nout;
		layerdata[i].bIdx = i == 0 ? 0 : layerdata[k].bIdx + layerdata[k].Nout;

	}
}
void initlayers() {
	//easy configraton of layers
	//actor network
	addlayer(inputs, 128, true);
	addlayer(128, 64, true);
	addlayer(64, 32, true);
	addlayer(32, 32, true);
	addlayer(32, output, true);


	setoffsets(true);
	//critic network
	addlayer(inputs, 128, false);
	addlayer(128, 64, false);
	addlayer(64, 32, false);
	addlayer(32, 32, false);
	addlayer(32, 1, false);



	setoffsets(false);

	printf("model parameter size :%d \n", actor_weightbuffersize + actor_biassize);
}



__host__ __device__ __forceinline__ int idx(int offset, int k) {
	return offset + k;
}
__device__  int antityidx(int off, int k, int ci) {
	return(off + k) * d.n + ci;
}

void readfiles(const std::string& dir, std::vector<float>& container)
{
	container.clear();
	std::fstream file(dir, std::ios::in);
	if (!file.is_open())
	{
		std::cerr << "Error opening " << dir << " file" << std::endl;
		return;
	}

	std::string val;

	while (file >> val)
	{
		container.push_back(std::stof(val));

	}
}

std::vector<float> actor_weights;
std::vector<float> critic_weights;
std::vector<float> actor_bias;
std::vector<float> critic_bias;
void initWB(float min, float max) {




	bool weightsloaded = true;
	readfiles("modeldata/actorweights.txt", actor_weights);
	readfiles("modeldata/criticweights.txt", critic_weights);
	readfiles("modeldata/actorbias.txt", actor_bias);
	readfiles("modeldata/criticbias.txt", critic_bias);
	if (actor_weights.size() != actor_weightbuffersize) {
		printf("%zd / %d \n", actor_weights.size(), actor_weightbuffersize);
		weightsloaded = false;
	}
	if (critic_weights.size() != critic_weightbuffersize) {
		weightsloaded = false;
	}
	if (actor_bias.size() != actor_biassize) {
		weightsloaded = false;
	}
	if (critic_bias.size() != critic_biassize) {
		weightsloaded = false;
	}
	if (!weightsloaded) {
		printf("weights loading error ,generating new weights\n");
		actor_bias.clear();
		critic_bias.clear();
		actor_weights.clear();
		critic_weights.clear();
		actor_weights.resize(actor_weightbuffersize);
		actor_bias.resize(actor_biassize);

		critic_weights.resize(critic_weightbuffersize);
		critic_bias.resize(critic_biassize);
		float bound = sqrtf(60.0f / (32 + 64));
		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(-bound, bound);
		//actor network weights and bias
		for (int i = 0; i < actor_weightbuffersize; i++) actor_weights[i] = dist(rng);
		for (int i = 0; i < actor_biassize; i++) actor_bias[i] = dist(rng);
		//critic network weights and bias
		for (int i = 0; i < critic_weightbuffersize; i++) critic_weights[i] = dist(rng);
		for (int i = 0; i < critic_biassize; i++) critic_bias[i] = dist(rng);

	}
	else {
		printf("weights loaded from files\n");
	}
}
void write_file(const std::string& address, std::vector<float>& arr) {


	std::ofstream file(address, std::ios::trunc);
	if (!file.is_open())
	{
		std::cerr << "Error opening" << address << "for writing" << std::endl;
	}
	for (float v : arr)
		file << v << " ";
	file << "\n";

	file.close();

}
void save_weights() {


	cudaError_t err= cudaMemcpy(actor_weights.data(), d_actor_weights, actor_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	if (err !=cudaSuccess) {printf(" memcpy from device to save actor weights failed %s \n", cudaGetErrorString(err));}
	err=cudaMemcpy(critic_weights.data(), d_critic_weights, critic_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	if (err !=cudaSuccess) {printf(" memcpy from device to save critic weights failed %s \n", cudaGetErrorString(err));}
	err=cudaMemcpy(actor_bias.data(), d_actor_bias, actor_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	if (err !=cudaSuccess) {printf(" memcpy from device to save actor bias failed %s \n", cudaGetErrorString(err));}
	err=cudaMemcpy(critic_bias.data(), d_critic_bias, critic_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	if (err !=cudaSuccess) {printf(" memcpy from device to save critic bias failed %s \n", cudaGetErrorString(err));}
	
	std::remove("modeldata/actorweights.txt");
	std::remove("modeldata/criticweights.txt");
	std::remove("modeldata/actorbias.txt");
	std::remove("modeldata/criticbias.txt");

	write_file("modeldata/actorweights.txt", actor_weights);
	write_file("modeldata/criticweights.txt", critic_weights);
	write_file("modeldata/actorbias.txt", actor_bias);
	write_file("modeldata/criticbias.txt", critic_bias);
	printf("weights saved \n");
}

__device__ double MSE = 0.0f;

__device__ float leakyrelu(float x) {

	return (x > 0.f) ? x : 0.01f * x;
	//return tanhf(x);

}
__device__ __forceinline__ void firstlayer(int n, int ci, int w, int D, int Inputs, float* input, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals) {

	for (int i = 0; i < n; i++) {

		float x = bias[idx(0, i)];
		for (int k = 0; k < inputs; k++) {
			x += input[k] * weights[k * n + i];

		}

		nodevals[antityidx(D, i, ci)] = leakyrelu(x);
	}

}
__device__ void solvelayers(int n, int ci, int nin, int w, int d, int b, int p, bool isout, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData) {


	for (int i = 0; i < n; i++) {


		float val = dBias[idx(b, i)];

		for (int j = 0; j < nin; j++) {
			float v = dNodeData[antityidx(p, j, ci)];
			val += v * dWeights[idx(w + i * nin, j)];
		}

		dNodeData[antityidx(d, i, ci)] = isout ? val : leakyrelu(val);
	}

}
__device__ float dslope(float x) {
	return x > 0.f ? 1.f : 0.01f;
}

__device__ float get_lClip(replaybuffer* buffer, int s) {


	float diff = clamp(buffer[s].logprob - buffer[s].old_logprob, -4, 4);
	float r = expf(diff);

	float ep = 0.2f;
	float clipped_out = 0.f;
	float adv = buffer[s].advantage;
	if (adv >= 0) {
		if (r > 1 + ep) clipped_out = 0;
		else clipped_out = -adv * r;
	}

	if (adv <= 0) {
		if (r < 1 - ep) clipped_out = 0;
		else clipped_out = -adv * r;
	}
	return clipped_out;
}
__device__ float beta = 0.05f;

__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin, int l1d, bool outlayer, bool isactor,const float* __restrict__ nodedata, float* dDelta, const float* __restrict__ dWeights,const float* __restrict__ dpreact, replaybuffer*  buffer, int s,const int* __restrict__ indices, int nodedatasize, int gen,int batchsize) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)return;

	int bidx = indices[s * batchsize + b];

	if (outlayer) {
		if (isactor) {
			float coeff = get_lClip(buffer, bidx);
			float mean = nodedata[b * nodedatasize + d + i];
			float diff = buffer[bidx].actionZ[i] - mean;
			dDelta[b * nodedatasize + d + i] = coeff * diff / (ACTION_SIGMA * ACTION_SIGMA);

		}
		else {
	float target = buffer[bidx].rtg;
			dDelta[b * nodedatasize + d + i] = nodedata[b * nodedatasize + d + i] - target;
		}
	}
	else {
		float sum = 0.0f;
#pragma unroll 4
		for (int k = 0; k < l1_nout; k++) {
			sum += dWeights[l1w + k * l1_nin + i] * dDelta[b * nodedatasize + l1d + k];
		}
		dDelta[b * nodedatasize + d + i] = sum * dslope(dpreact[b * nodedatasize + d + i]);
	}



}
__device__ float getmoment(float moment, float gradient, float beta) {
	float x = 0.0f;

	x = beta * moment + (1.0f - beta) * gradient;

	return x;


}
__device__ float getv(float v, float gradient, float beta) {
	float x = 0.0f;

	x = beta * v + (1.0f - beta) * gradient * gradient;

	return x;

}
__device__ float getlr(float lr, float m, float v, float eps) {
	return lr * m / (sqrtf(v) + eps);
}

__device__ __forceinline__ void adam_update(float& w, float2& mv, const float g,
	const float lrate, const float b1, const float b2, const float bt1, const float bt2)
{
	const float m = b1 * mv.x + (1.0f - b1) * g;
	const float v = b2 * mv.y + (1.0f - b2) * g * g;
	mv.x = m; mv.y = v;
	w -= lrate * (m * bt1) / (sqrtf(v * bt2) + 1e-6f);      
}

__global__ void tuneweights(const Layer* __restrict__ L, const int curbatch, const int s, const int nds,
	const float* __restrict__ nodes, const float* __restrict__ delta,
	const replaybuffer* __restrict__ buffer, const int* __restrict__ indices,
	float* W, float* Bv, float2* aw, float2* ab,
	const float lrate, const float b1, const float b2, const float bt1, const float bt2,const int batchsize) {

	


	extern __shared__ float sh[];                      

	int row = blockIdx.x, l = 0;
	for (;; ++l) {
		const int rows = (l == 0) ? L[0].Nin : L[l].Nout;
		if (row < rows) break;
		row -= rows;
	}
	const int n = L[l].Nout, nin = L[l].Nin, lw = L[l].wIdx, d = L[l].dIdx, lb = L[l].bIdx;
	const float fb = (float)curbatch;

	if (l == 0) {
		const int k = row;
		for (int b = threadIdx.x; b < curbatch; b += blockDim.x)
			sh[b] = buffer[indices[s * batchsize + b]].s1[k];
		__syncthreads();
		for (int i = threadIdx.x; i < n; i += blockDim.x) {
			float g = 0.0f, bg = 0.0f;
#pragma unroll 8
			for (int b = 0; b < curbatch; ++b) {
				const float dv = delta[b * nds + d + i];       
				g = fmaf(dv, sh[b], g);
				bg += dv;
			}
			const int wi = lw + k * n + i;
			adam_update(W[wi], aw[wi], g / fb, lrate, b1, b2, bt1, bt2);
			if (k == 0) adam_update(Bv[lb + i], ab[lb + i], bg / fb, lrate, b1, b2, bt1, bt2);
		}
	}
	else {
		const int i = row, l1d = L[l - 1].dIdx;
		for (int b = threadIdx.x; b < curbatch; b += blockDim.x)
			sh[b] = delta[b * nds + d + i];
		__syncthreads();
		for (int k = threadIdx.x; k < nin; k += blockDim.x) {
			float g = 0.0f;
#pragma unroll 8
			for (int b = 0; b < curbatch; ++b)
				g = fmaf(sh[b], nodes[b * nds + l1d + k], g);   
			const int wi = lw + i * nin + k;
			adam_update(W[wi], aw[wi], g / fb, lrate, b1, b2, bt1, bt2);
		}
		if (threadIdx.x == 0) {
			float bg = 0.0f;
			for (int b = 0; b < curbatch; ++b) bg += sh[b];
			adam_update(Bv[lb + i], ab[lb + i], bg / fb, lrate, b1, b2, bt1, bt2);
		}
	}
}
__device__ float sigmoid(float x) {

	return  1.0f / (1.0f + expf(-x));

}


__device__ float actionTarget(int j, float action,body& p) {
	float lo = 0.0f;
	float hi = 0.0f;
	switch (j) {
	case 0: lo = degtorad* p.lefthip.minangle; hi = degtorad * p.lefthip.maxangle; break;
	case 1: lo = degtorad* p.righthip.minangle; hi = degtorad * p.righthip.maxangle; break;
	case 2: lo = degtorad* p.leftknee.minangle; hi = degtorad * p.leftknee.maxangle; break;
	case 3: lo = degtorad* p.rightknee.minangle; hi = degtorad * p.rightknee.maxangle; break;
	case 4: lo = degtorad* p.leftankle.minangle; hi = degtorad * p.leftankle.maxangle; break;
	case 5: lo = degtorad* p.rightankle.minangle; hi = degtorad * p.rightankle.maxangle; break;
	case 6: lo = 0.0f; hi = 1.0f; break;
	case 7: lo = 0.0f; hi = 1.0f; break;
	case 8: lo = 0.0f; hi = 1.0f; break;
	case 9: lo = 0.0f; hi = 1.0f; break;
	case 10: lo = 0.0f; hi = 1.0f; break;
	case 11: lo = 0.0f; hi = 1.0f; break;
	
	}
	return (lo) + 0.5f * (action + 1.0f) * ((hi) - (lo));
}
__device__ void setJointAction(body& b, int j, float value) {
	switch (j) {
	case 0: b.lefthip.targetangle = value *radtodeg; break;
	case 1: b.righthip.targetangle = value *radtodeg; break;
	case 2: b.leftknee.targetangle = value *radtodeg; break;
	case 3: b.rightknee.targetangle = value *radtodeg; break;
	case 4: b.leftankle.targetangle = value *radtodeg; break;
	case 5: b.rightankle.targetangle = value *radtodeg; break;
	case 6: b.lefthip.strength = value * maxjointstrength; break;
	case 7: b.righthip.strength = value * maxjointstrength; break;
	case 8: b.leftknee.strength = value * maxjointstrength; break;
	case 9: b.rightknee.strength = value * maxjointstrength; break;
	case 10: b.leftankle.strength = value * maxjointstrength; break;
	case 11: b.rightankle.strength = value * maxjointstrength; break;

	
	}
}
__device__ float tanhGaussianLogProb(float z, float action, float mean) {
	float diff = z - mean;
	return -0.5f * diff * diff / (ACTION_SIGMA * ACTION_SIGMA)
		- logf(ACTION_SIGMA)
		- 0.9189385332f
		- logf(fmaxf(1.0f - action * action, 1e-6f));
}
__device__ void getoutput(int D, float* nodevals, body* d_body, replaybuffer* buffer, int s, int ci, curandState* d_rngstate) {
	int bidx = s * d.n + ci;
	body b = d_body[ci];
	float oldLogprob = 0.0f;
	for (int j = 0; j < output; j++) {
		float mean = nodevals[antityidx(D, j, ci)];
		float z = mean + ACTION_SIGMA * curand_normal(&d_rngstate[ci]);
		float action = tanhf(z);
		buffer[bidx].action[j] = action;
		buffer[bidx].actionZ[j] = z;
		float value = actionTarget(j, action,b);
		setJointAction(b, j, value);

		oldLogprob += tanhGaussianLogProb(z, action, mean);
	}
	d_body[ci] = b;
	buffer[bidx].old_logprob = oldLogprob;
}
__global__ void getlog(int n, int d, int* indices, replaybuffer* buffer, int s, float* nodevals, int nodedatasize,int batchsize) {

	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b >= n)return;
	int off = b * nodedatasize;
	int bidx = indices[s * batchsize + b];
	float logprob = 0.0f;
	for (int j = 0; j < output; j++) {
		float mean = nodevals[off + d + j];
		logprob += tanhGaussianLogProb(buffer[bidx].actionZ[j], buffer[bidx].action[j], mean);
	}
	buffer[bidx].logprob = logprob;
}
__device__ float oldr = 0.0f;
__device__ int id = 0;

__device__ int logframe = 0;
__device__ float normalize(float x, float max, float min) {
	return ((x - min) / (max - min)) * 2.0f - 1.0f;
	//[-1,1] bounded for better normalization for values in a rnage
}

__device__ float steppingreward(body& b,float stepr) {
	if (!b.alive)
		return 0.0f;

	constexpr float minimumStep = 10.0f;
	constexpr float fullRewardStep = 80.0f;
	float reward = 0.0f;
	bool leftContact = b.leftfeettouching;
	bool rightContact = b.rightfeettouching;

	// Record lift-off only while the other foot is supporting the robot.
	if (b.leftfootprevcontact && !leftContact && rightContact) {
		b.leftfootswinging = true;
		b.leftfoottakeoffrelx = b.leftfoot.pos.x - b.torso.pos.x;
	}
	if (b.rightfootprevcontact && !rightContact && leftContact) {
		b.rightfootswinging = true;
		b.rightfoottakeoffrelx = b.rightfoot.pos.x - b.torso.pos.x;
	}

	// Pay once, on landing, for placing the swinging foot farther forward
	// relative to the torso while the opposite foot remains in contact.
	if (b.leftfootswinging && !b.leftfootprevcontact && leftContact) {
		if (rightContact) {
			float landingRelX = b.leftfoot.pos.x - b.torso.pos.x;
			float forwardStride = landingRelX - b.leftfoottakeoffrelx;
			float score = clamp((forwardStride - minimumStep) /
				(fullRewardStep - minimumStep), 0.0f, 1.0f);
			reward += stepr * score;
		}
		b.leftfootswinging = false;
	}
	if (b.rightfootswinging && !b.rightfootprevcontact && rightContact) {
		if (leftContact) {
			float landingRelX = b.rightfoot.pos.x - b.torso.pos.x;
			float forwardStride = landingRelX - b.rightfoottakeoffrelx;
			float score = clamp((forwardStride - minimumStep) /
				(fullRewardStep - minimumStep), 0.0f, 1.0f);
			reward += stepr * score;
		}
		b.rightfootswinging = false;
	}

	b.leftfootprevcontact = leftContact;
	b.rightfootprevcontact = rightContact;
	return reward;
}
__device__ float degval(part& p) {
	return expf(-0.5f * (p.angle / 15.0f) * (p.angle / 15.0f));
}
__device__ float kneeStraightness(part& thigh, part& shin) {
	float angleDiff = (shin.angle - thigh.angle) * degtorad;
	angleDiff = atan2f(sinf(angleDiff), cosf(angleDiff));
	float normalizedDiff = angleDiff / (15.0f * degtorad);
	return expf(-0.5f * normalizedDiff * normalizedDiff);
}

__global__ void reward_kernel(int n, int s, replaybuffer* buffer, body* d_body, float aliver, float deadr, float feetr, float distr, float yr, float reverse, float fr, float targetSpeed, float stepr, float slidingPenaltyWeight) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	int bidx = s * d.n + i;

	body& b = d_body[i];

	
	float dead = d_body[i].alive ? aliver : -deadr;

	float feet = (d_body[i].leftfeettouching || d_body[i].rightfeettouching) ? feetr : 0.0f;

	float olddist = d_body[i].olddist;
	float dist = d_body[i].torso.pos.x - startx;
	float dx = dist - olddist;
	float speed = dx / fmaxf(DT, 1e-4f);

	if (d_body[i].torso.pos.x > maxdist || d_body[i].torso.pos.x < -maxdist) {
		printf("WARNING: increase maxdist define or it will cause errors");
		//future warning
	}


	float val = clamp(d_body[i].torso.pos.y / 535.0f, 0.0f, 1.0f);
	
	float kneePosture = kneeStraightness(b.leftthigh, b.leftshin)
		+ kneeStraightness(b.rightthigh, b.rightshin);
	float upright = degval(b.torso) + kneePosture;
	bool hasSupport = d_body[i].leftfeettouching || d_body[i].rightfeettouching;
	float progress = clamp(speed / fmaxf(targetSpeed, 1.0f), -1.0f, 1.0f);
	progress *= progress < 0.0f ? reverse : fr;
	float movementReward = d_body[i].alive && hasSupport
		? distr * progress * upright
		: 0.0f;
	bool footSwinging = d_body[i].leftfootswinging || d_body[i].rightfootswinging;
	float uncoordinatedForwardSpeed = clamp(speed / fmaxf(targetSpeed, 1.0f), 0.0f, 1.0f);
	float slidingPenalty = d_body[i].alive && hasSupport && !footSwinging
		? slidingPenaltyWeight * uncoordinatedForwardSpeed
		: 0.0f;

	float straight;
	if (hasSupport) {
		// Keep a small balance signal, but don't let standing dominate sparse steps.
		straight = 0.1f * (upright + val * yr);
	}
	else {
		val = d_body[i].torso.pos.y / 535.0f;
		float limit = 1.2f;//give headroom for small jumps
		straight = val > limit ? -0.25f * (val - limit) * yr : 0.0f;
	}
		
	float stepreward = steppingreward(d_body[i], stepr);
		


	
	float sum = dead + feet + straight + movementReward + stepreward - slidingPenalty;
	float reward =sum;

	d_body[i].olddist = d_body[i].torso.pos.x - 500.0f;


	buffer[bidx].reward = reward;

	
	if (!d_body[i].alive) {
		resetrobotidxkernel(d_body[i]);
	}
	

}
__device__ void getQvalue(int s, int c, int D, float* nodevals, replaybuffer* buffer) {
	int bidx = s * d.n + c;
	buffer[bidx].value = nodevals[antityidx(D, 0, c)];
	if (!isfinite(buffer[bidx].value)) {
		buffer[bidx].value = 0.0f;
		printf("BAD PPO %d value=%f \n",
			s,
			buffer[bidx].value
		);
	}
}
__global__ void computevalskernel(int n, int ticks, replaybuffer* buffer) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= n) return;
	float y = 0.99f;
	float gae = 0;
	for (int t = ticks - 1; t >= 0; t--) {
		int i = t * n + c;
		int next = (t + 1) * n + c;

		float nextValue = 0.0f;

		if (t != ticks - 1)
			nextValue = buffer[next].value;


		float mask = 1.0f - buffer[i].done;


		float delta =
			buffer[i].reward
			+ y * nextValue * mask
			- buffer[i].value;


		gae =
			delta
			+ y * 0.95 * mask * gae;


		buffer[i].advantage = gae;
		buffer[i].rtg = gae + buffer[i].value;


		atomicAdd(&MSE, gae * gae);
	}
}
__global__ void compute_rtg(int n, replaybuffer* buffer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	buffer[i].rtg = buffer[i].advantage + buffer[i].value;
}
__device__ double d_adv_mean;
__device__ double d_adv_var;
__global__ void compute_adv_stats(int n, replaybuffer* buffer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	double a = buffer[i].advantage;
	atomicAdd(&d_adv_mean, a);
	atomicAdd(&d_adv_var, a * a);
}
__global__ void normalize_advantages(int n, replaybuffer* buffer, float mean, float std) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	buffer[i].advantage = (buffer[i].advantage - mean) / (std + 1e-8f);
}
__device__ float boolTofloat(bool x) {
	
	return x ? 1.0f : 0.0f;
}

__device__ void fill_partinput(int i, float* input, part& p) {
	input[i] = normalize(p.pos.x-500.0f,maxdist,-maxdist);
	input[i+1] = normalize(p.pos.y,1200.0f,-1200.0f);
	input[i+2] = normalize(p.vel.x,maxvel,-maxvel);
	input[i+3] = normalize(p.vel.y, maxvel, -maxvel);
	input[i+4] = p.angle /180.0f;
	input[i+5] = normalize(p.angvel,maxangvel,-maxangvel);
	input[i+6] = p.mass/ maxmass;
	input[i+7] = p.inertia/maxinertia;
	

	
}
__device__ void fill_jointinput(int i, float* input, joint& j) {
	input[i] = normalize(j.targetangle,j.maxangle,j.minangle);
	input[i + 1] = j.strength / maxjointstrength;
}


__global__ void netkernel(int n, body* d_body, const float* __restrict__ weights,
	const float* __restrict__ bias, float* nodevals, const Layer* layer, bool isactor, replaybuffer* buffer, int s, curandState* rng

) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;





	for (int l = 0; l < d.layers; l++) {

		if (l == 0) {

			body b = d_body[i];

			






			float input[inputs] = { 0 };
			input[0] = 0.125f;
			fill_partinput(1, input, b.torso);
			fill_partinput(9, input, b.leftthigh);
			fill_partinput(17, input, b.rightthigh);
			fill_partinput(25, input, b.leftshin);
			fill_partinput(33, input, b.rightshin);
			fill_partinput(41, input, b.leftfoot);
			fill_partinput(49, input, b.rightfoot);

			fill_jointinput(57, input, b.lefthip);
			fill_jointinput(59, input, b.righthip);
			fill_jointinput(61, input, b.leftknee);
			fill_jointinput(63, input, b.rightknee);
			fill_jointinput(65, input, b.leftankle);
			fill_jointinput(67, input, b.rightankle);
			input[69] = boolTofloat(b.leftfeettouching);
				input[70] = boolTofloat(b.rightfeettouching);
				input[71] = boolTofloat(b.torsotouchingground);
				input[72] = boolTofloat(b.leftshintouching);
				input[73] = boolTofloat(b.rightshintouching);
				input[74] = boolTofloat(b.rightthightouching);
				input[75] = boolTofloat(b.leftthightouching);


				
			
			int insize = inputs;
			if (isactor) {
				int bidx = s * d.n + i;
				for (int b = 0; b < insize; b++) {
					buffer[bidx].s1[b] = input[b];

				}
			}
			else {
				int bidx = s * d.n + i;
				for (int b = 0; b < insize; b++) {
					input[b] = buffer[bidx].s1[b];
				}
			}
			int didx = layer[0].dIdx;

			firstlayer(layer[0].Nout, i, 0, didx, inputs, input, weights, bias, nodevals);
		}
		else {


			int nin = layer[l - 1].Nout;
			int wl = layer[l].wIdx;
			int di = layer[l].dIdx;
			int b = layer[l].bIdx;
			int p = layer[l - 1].dIdx;

			bool isout = (l == d.layers - 1);


			solvelayers(layer[l].Nout, i, nin, wl, di, b, p, isout, weights, bias, nodevals);

		}





	}
	float outd = layer[d.layers - 1].dIdx;
	if (isactor) {
		getoutput(outd, nodevals, d_body, buffer, s, i, rng);
	}
	else {
		getQvalue(s, i, outd, nodevals, buffer);
	}



}
__global__ void getdone(int n, body* b, replaybuffer* buffer, int s,bool ld) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	int bidx = s * d.n + i;

	if (!ld) {
		if (b[i].torsotouchingground || b[i].torso.pos.x < -100.0f) {

			buffer[bidx].done = true;
			b[i].alive = false;
		}

		else {
			buffer[bidx].done = false;
			b[i].alive = true;
		}
	}
	else {

		if (b[i].torsotouchingground || b[i].leftshintouching || b[i].rightshintouching
			|| b[i].leftthightouching || b[i].rightthightouching || b[i].torso.pos.x < -100.0f) {

			buffer[bidx].done = true;
			b[i].alive = false;
		}
		else {
			buffer[bidx].done = false;
			b[i].alive = true;
		}
	}

	if (b[i].torso.pos.x > maxdist) {
		printf(" max dist reached \n");
		resetrobotidxkernel(b[i]);
	}

}
//learner net
__global__ void firstlayerfrozen(int n,const int s,const int* __restrict__ indices, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals, float* d_preact,const replaybuffer* __restrict__ buffer,const int nodedatasize,int batchsize)
{
	__shared__ float s_in[inputs];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int B = blockIdx.y;

	const replaybuffer* rb = buffer + indices[s * batchsize + B];
	for (int k = threadIdx.x; k < inputs; k += blockDim.x) s_in[k] = rb->s1[k];
	__syncthreads();

	if (i >= n)return;



	float x = bias[ i];

#pragma unroll 4
	for (int k = 0; k < inputs; k++) x = fmaf(s_in[k], weights[k * n + i], x);   

	const int off = B * nodedatasize + i;
	d_preact[off] = x;
	nodevals[off] = leakyrelu(x);

}
__global__ void solvefrozenlayers(int n,const int nin,const int w,const int D,const int b,const int p,const bool isout,const int* __restrict__ indices, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData, float* preact, int nodedatasize) {



	const int lane = threadIdx.x & 31;
	const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   
	const int B = blockIdx.y;
	if (i >= n) return;
	const float* in = dNodeData + B * nodedatasize + p;   
	const float* wr = dWeights + w + i * nin;


	float acc = 0.0f;
#pragma unroll 4
	for (int j = lane; j < nin; j += 32) acc = fmaf(in[j], wr[j], acc);

#pragma unroll
	for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);

	if (lane == 0) {
		const float val = acc + dBias[b + i];
		const int off = B * nodedatasize + D + i;
		preact[off] = val;
		dNodeData[off] = isout ? val : leakyrelu(val);
	}

}

void checkdone(int step) {
	getdone << <blocks(robot_count), threads >> > (robot_count, d_bodies, d_state, step,legalive);
}
void normalize_advantages() {

	double zero = 0.0;
	cudaMemcpyToSymbol(d_adv_mean, &zero, sizeof(double));
	cudaMemcpyToSymbol(d_adv_var, &zero, sizeof(double));

	int t = 256;
	int b = (replaybuffersize + t - 1) / t;
	compute_adv_stats << <b, t >> > (replaybuffersize, d_state);

	double mean = 0.0, var = 0.0;
	cudaMemcpyFromSymbol(&mean, d_adv_mean, sizeof(double));
	cudaMemcpyFromSymbol(&var, d_adv_var, sizeof(double));
	mean /= replaybuffersize;
	var = var / replaybuffersize - mean * mean;
	float std = sqrtf(fmaxf((float)var, 1e-8f));

	normalize_advantages << <b, t >> > (replaybuffersize, d_state, (float)mean, std);

	//compute_rtg << <blocks(settings.replaybuffersize), threads >> > (settings.replaybuffersize, d_state);
}
void runfrozennet(int s, int curbatch, bool isactor) {
	auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	int layers = (isactor) ? actor_layers : critic_layers;

	int l1blocks = (layerdata[0].Nout + 128 - 1) / 128;
	int w = layerdata[0].wIdx;
	dim3 grid(l1blocks, curbatch);

	if (isactor) {
		firstlayerfrozen << <grid, layerdata[0].Nout >> > (layerdata[0].Nout, s, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, d_state, actor_nodedatasize,hbatchsize);
	}
	else {
		firstlayerfrozen << <grid, layerdata[0].Nout >> > (layerdata[0].Nout, s, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, d_state, critic_nodedatasize,hbatchsize);

	}


	for (int l = 1; l < layers; l++)
	{

		int Blocks = (layerdata[l].Nout + threads - 1) / threads;
		int shared_bytes = layerdata[l].Nin * sizeof(float);

		int nin = layerdata[l - 1].Nout;
		int wl = layerdata[l].wIdx;
		int d = layerdata[l].dIdx;
		int b = layerdata[l].bIdx;
		int p = layerdata[l - 1].dIdx;

		bool isout = (l == layers - 1);
		int n = layerdata[l].Nout;
		int tpb = n <= 32 ? 32 : (n <= 64 ? 64 : 128);
		const int warpsPerBlock = tpb / 32;
		int blocks = (n + warpsPerBlock - 1) / warpsPerBlock;
		dim3 grid2(blocks, curbatch);
		if (isactor) {

			solvefrozenlayers << <grid2, tpb >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, actor_nodedatasize);
		}
		else {

			solvefrozenlayers << <grid2, tpb >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, critic_nodedatasize);
		}



	}
	if (isactor) {
		int outd = layerdata[actor_layers - 1].dIdx;

		getlog << <blocks(curbatch), threads >> > (curbatch, outd, d_indices, d_state, s, d_actor_nodvals, actor_nodedatasize,hbatchsize);
	}

}
void reward(int s) {
	reward_kernel << <blocks(robot_count), threads >> > (robot_count, s, d_state, d_bodies, alive, dead, feettouching, distr, yup, reversepenalty, forwardr, targetForwardSpeed, steppingRewardWeight, slidingPenaltyWeight);



}
void computevals() {

	computevalskernel << <blocks(robot_count), threads >> > (robot_count, replaybuffersize / robot_count, d_state);


}
void backpropogation(int s, int curBatch, bool isactor) {

	auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	int layers = (isactor) ? actor_layers : critic_layers;


	for (int l = layers - 1; l >= 0; l--) {


		int d = layerdata[l].dIdx;
		int n = layerdata[l].Nout;
		int l1out = (l < layers - 1) ? layerdata[l + 1].Nout : 0;
		int l1w = (l < layers - 1) ? layerdata[l + 1].wIdx : 0;
		int l1d = (l < layers - 1) ? layerdata[l + 1].dIdx : 0;
		int l1nin = (l < layers - 1) ? layerdata[l + 1].Nin : 0;



		bool outlayer = (l == layers - 1) ? 1 : 0;

		int tpb = n <= 32 ? 32 : (n <= 64 ? 64 : 128);
		int blocks = (n + tpb - 1) / tpb;
		dim3 grid(blocks, curBatch);
		if (isactor) {
			compute_delta << <grid, tpb >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_actor_nodvals, d_actor_delta, d_actor_weights, d_actor_preact, d_state, s, d_indices, actor_nodedatasize, rolloutstep,hbatchsize);
		}
		else {

			compute_delta << <grid, tpb >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_critic_nodvals, d_critic_delta, d_critic_weights, d_critic_preact, d_state, s, d_indices, critic_nodedatasize, rolloutstep,hbatchsize);
		}


	}
	const int t = isactor ? actor_adam_step++ : critic_adam_step++;

	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float betat1 = 1.0f / (1.0f - powf(beta1, t));
	float betat2 = 1.0f / (1.0f - powf(beta2, t));
		
		int rows = layerdata[0].Nin;                     
		for (int l = 1; l < layers; l++) rows += layerdata[l].Nout;   
		const size_t smem = curBatch * sizeof(float);
		if (isactor)
			tuneweights << <rows, 128, smem >> > (d_actlayer, curBatch, s, actor_nodedatasize,
				d_actor_nodvals, d_actor_delta, d_state, d_indices,
				d_actor_weights, d_actor_bias, actor_adam_weights, actor_adam_bias,
				lr, beta1, beta2, betat1, betat2,hbatchsize);
		else
			tuneweights << <rows, 128, smem >> > (d_critlayer, curBatch, s, critic_nodedatasize,
				d_critic_nodvals, d_critic_delta, d_state, d_indices,
				d_critic_weights, d_critic_bias, critic_adam_weights, critic_adam_bias,
				lr, beta1, beta2, betat1, betat2,hbatchsize);


}
void frozennet(bool isactor) {
	//auto& layerdata = (isactor) ? actor_layerdata : critic_layerdata;
	//int layers = (isactor) ? settings.actor_layers : settings.critic_layers;
	int numbatches = (replaybuffersize + hbatchsize - 1) / hbatchsize;
	for (int s = 0; s < numbatches; s++) {
		int curBatch = std::min(hbatchsize, replaybuffersize - s * hbatchsize);

		runfrozennet(s, curBatch, isactor);


		backpropogation(s, curBatch, isactor);


	}
}
void net(int s, bool isactor) {

	if (isactor) {
		netkernel << <blocks(robot_count), threads >> > (robot_count, d_bodies, d_actor_weights, d_actor_bias, d_antity_nodevals, d_actlayer, isactor, d_state, s, d_rngstate);
	}
	else {
		netkernel << <blocks(robot_count), threads >> > (robot_count, d_bodies, d_critic_weights, d_critic_bias, d_antity_nodevals, d_critlayer, isactor, d_state, s, d_rngstate);

	}

}

void run_network(float dt) {

	net(step, true);//actor forward pass
	net(step, false);//critic forward pass
	updateRobot(dt);
	checkdone(step);
	reward(step);
	float h_reward = 0.0f;
	float zero = 0.0f;









	step++;
	if (step >= replaybuffersize / robot_count && training) {
		rolloutstep++;
		auto start = std::chrono::high_resolution_clock::now();
		computevals();
		oldmloss = mloss;
		double l = 0.0f;
		cudaError_t err = cudaMemcpyFromSymbol(&l, MSE, sizeof(double));
		geterror("mse cpy from symbol", err);
		mloss = (l / (double)replaybuffersize);
		double zz = 0;
		cudaMemcpyToSymbol(MSE, &zz, sizeof(double));
		normalize_advantages(); /*if no learning sign re compute
		rtg after normilization tho it doesnt needed to as rtg is only used in critic backprop but good to try
		also the norm adv is used in lclip it could also be an issue,
		a better thing would be to compute rtg after normalized adv so critic and actor both reviced target based on norm adv
		--it didnt work.
		*/

		for (int i = 0; i < rollout_epoch; i++) {
			shuffleindices();
			frozennet(false);//critic backprop
			frozennet(true);//actor backprop



		
		}
		auto end = std::chrono::steady_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		rollout_time = elapsed.count();
	}
	if (step >= replaybuffersize / robot_count)	step = 0;

}

std::mt19937 shuffleRng(1234);
void shuffleindices() {
	shuffled_indices.resize(replaybuffersize);
	for (int i = 0; i < replaybuffersize; i++) {
		shuffled_indices[i] = i;
	}
	std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), shuffleRng);
	cudaError_t err =cudaMemcpy(d_indices, shuffled_indices.data(), replaybuffersize * sizeof(int), cudaMemcpyHostToDevice);
	
	geterror("indices copy", err);
}

void allocatenetmem() {
	cudaError_t err;
	err= cudaMalloc(&d_actor_weights, actor_weightbuffersize * sizeof(float));
	if (err) { printf("actor weights malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critic_weights, critic_weightbuffersize * sizeof(float));
	if (err) { printf("critic weights malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_actor_bias, actor_biassize * sizeof(float));
	if (err) { printf("actor bias malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critic_bias, critic_biassize * sizeof(float));
	if (err) { printf("critic bias malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_actor_nodvals, hbatchsize * actor_nodedatasize * sizeof(float));
	if (err) { printf("actor nodevals malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critic_nodvals, hbatchsize * critic_nodedatasize * sizeof(float));
	if (err) { printf("critic nodevals malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_actor_preact, hbatchsize * actor_nodedatasize * sizeof(float));
	if (err) { printf("actor pereact malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critic_preact, hbatchsize * critic_nodedatasize * sizeof(float));
	if (err) { printf("critic preact malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_actor_delta, hbatchsize * actor_nodedatasize * sizeof(float));
	if (err) { printf("actor delta malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critic_delta, hbatchsize * critic_nodedatasize * sizeof(float));
	if (err) { printf("critic delta malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_state, replaybuffersize * sizeof(replaybuffer));
	if (err) { printf("state  malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_indices, replaybuffersize * sizeof(int));
	if (err) { printf("indices malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_antity_nodevals, antitynodesize * sizeof(float));
	if (err) { printf("antity nodevals malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_actlayer, actor_layers * sizeof(Layer));
	if (err) { printf("actlayer malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&d_critlayer, critic_layers * sizeof(Layer));
	if (err) { printf("critic layer malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&actor_adam_weights, actor_weightbuffersize * sizeof(float2));
	if (err) { printf("actor adam weights malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&actor_adam_bias, actor_biassize * sizeof(float2));
	if (err) { printf("actor adam bias malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&critic_adam_weights, critic_weightbuffersize * sizeof(float2));
	if (err) { printf("critic adam weights malloc failed %s \n", cudaGetErrorString(err)); }
	err= cudaMalloc(&critic_adam_bias, critic_biassize * sizeof(float2));
	if (err) { printf("critic adam bias malloc failed %s \n", cudaGetErrorString(err)); }
	
	err= cudaMemset(actor_adam_weights, 0.0f, actor_weightbuffersize * sizeof(float2));
	if (err) { printf("actor adam weights memset failed %s \n", cudaGetErrorString(err)); }
	err= cudaMemset(actor_adam_bias, 0.0f, actor_biassize * sizeof(float2));
	if (err) { printf("actor adam bias memset failed %s \n", cudaGetErrorString(err)); }
	err= cudaMemset(critic_adam_weights, 0.0f, critic_weightbuffersize * sizeof(float2));
	if (err) { printf("critic adam weights memset failed %s \n", cudaGetErrorString(err)); }
	err= cudaMemset(critic_adam_bias, 0.0f, critic_biassize * sizeof(float2));
	if (err) { printf("critic adam bias memset failed %s \n", cudaGetErrorString(err)); }

	
		
	
	
}
void copywbtogpu() {
	cudaError_t err;

	err=cudaMemcpy(d_actor_weights, actor_weights.data(), actor_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	if (err) { printf("actor weights memcpy failed %s \n", cudaGetErrorString(err)); }
	err=cudaMemcpy(d_critic_weights, critic_weights.data(), critic_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	if (err) { printf("critic weights memcpy failed %s \n", cudaGetErrorString(err)); }
	err=cudaMemcpy(d_actor_bias, actor_bias.data(), actor_biassize * sizeof(float), cudaMemcpyHostToDevice);
	if (err) { printf("actor bias memcpy failed %s \n", cudaGetErrorString(err)); }
	err=cudaMemcpy(d_critic_bias, critic_bias.data(), critic_biassize * sizeof(float), cudaMemcpyHostToDevice);
	if (err) { printf("critic bias memcpy failed %s \n", cudaGetErrorString(err)); }
	err=cudaMemcpy(d_actlayer, actor_layerdata.data(), actor_layers * sizeof(Layer), cudaMemcpyHostToDevice);
	if (err) { printf("actlayer memcpy failed %s \n", cudaGetErrorString(err)); }
	err=cudaMemcpy(d_critlayer, critic_layerdata.data(), critic_layers * sizeof(Layer), cudaMemcpyHostToDevice);
	if (err) { printf("critlayer memcpy failed %s \n", cudaGetErrorString(err)); }

	

}
void restart() {
	save_weights();
	unregistervbo();
#if usecuda
	freedevmem();
#endif
	cudafree();
	printf("memfree on restart \n");
	robot_count = sample_robot_count;
	actor_weightbuffersize = 0;
	critic_weightbuffersize = 0;
	actor_biassize = 0;
	critic_biassize = 0;
	actor_nodedatasize = 0;
	rolloutstep = 0;
	actor_adam_step = 1;
	critic_adam_step = 1;
	step = 0;
	gen = 0;
	replaybuffersize = robot_count * 2048;
	critic_nodedatasize = 0;
	actor_layers = 0;
	critic_layers = 0;
	printf("variable reset \n");
	initrobot(robot_count);

	initnetwork();




}
void initnetwork() {
	replaybuffersize -= replaybuffersize % robot_count;
	initlayers();
	setconst();
	initWB(-0.5f, 0.5f);
	allocatenetmem();
	copywbtogpu();
	allocate();
	//setmaxdisttotarget();

	printf("network init complete \n==================\n");

}
void cudafree() {
	cudaFree(d_actor_nodvals);
	cudaFree(d_critic_nodvals);
	cudaFree(d_actor_weights);
	cudaFree(d_critic_weights);
	cudaFree(d_actor_bias);
	cudaFree(d_critic_bias);
	cudaFree(d_actor_preact);
	cudaFree(d_critic_preact);
	cudaFree(d_actor_delta);
	cudaFree(d_critic_delta);
	cudaFree(d_state);
	cudaFree(d_indices);
	cudaFree(d_antity_nodevals);
	cudaFree(d_actlayer);
	cudaFree(d_critlayer);


}
