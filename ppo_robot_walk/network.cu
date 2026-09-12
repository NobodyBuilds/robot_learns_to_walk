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



struct device_data {
	float targetx;
	float targetz;
	float maxdisttotarget;
	int n;
	int layers;

};
device_data h;
__constant__ device_data d;

void setconst() {
	h.n = robot_count;
	h.targetx = targetx;
	h.targetz = targetz;
	h.maxdisttotarget = maxdisttotarget;
	h.layers = actor_layers;
	cudaMemcpyToSymbol(d, &h, sizeof(device_data));
}

void geterror(const std::string& label, cudaError_t err) {
	static std::unordered_set<std::string> printed;

	if (printed.find(label) == printed.end())
	{
		if (err != cudaSuccess) {
			printf("%s error : %s \n", label.c_str(), cudaGetErrorString(err));
		}
		printed.insert(label);
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
	int n = robot_count;
	cudaMalloc(&d_body, n * sizeof(body));
	
	cudaMalloc(&d_rngstate, robot_count * sizeof(curandState));
	init_rng << <1, 1 >> > (d_rngstate, robot_count, 3476);
	printf("data allocated \n");
}
void setmaxdisttotarget() {
	float dx = 0.0f- targetx;
	float dz = 0.0f- targetz;


	maxdisttotarget = sqrtf(dx * dx + dz * dz);
	setconst();
	
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
		antitynodesize += out *robot_count;
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


	cudaMemcpy(actor_weights.data(), d_actor_weights, actor_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(critic_weights.data(), d_critic_weights, critic_weightbuffersize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(actor_bias.data(), d_actor_bias, actor_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(critic_bias.data(), d_critic_bias, critic_biassize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (err) {
		printf(" memcpy grom device to save weights failed %s \n", cudaGetErrorString(err));
	}
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


__device__  int batchsize = 128;



__device__ float leakyrelu(float x) {

	//return (x > 0.f) ? x : 0.01f * x;
	return tanhf(x);

}
__device__ __forceinline__ void firstlayer(int n, int ci, int w, int D, int inputs, float* input, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals) {

	for (int i = 0; i < n; i++) {

		float x = bias[idx(0, i)];
		for (int k = 0; k < inputs; k++) {
			x += input[k] * weights[idx(w + i * inputs, k)];

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
__device__ float clamp(float val, float min, float max) {
	return fminf(fmaxf(val, min), max);
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
__global__ void compute_delta(int n, int d, int l1_nout, int l1w, int l1_nin, int l1d, bool outlayer, bool isactor, float* nodedata, float* dDelta, float* dWeights, float* dpreact, replaybuffer* buffer, int s, int* indices, int nodedatasize, int gen) {


	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)return;

	int bidx = indices[s * batchsize + b];

	float target = buffer[bidx].rtg;
	if (outlayer) {
		if (isactor) {
			float coeff = get_lClip(buffer, bidx);
			float mean = nodedata[b * nodedatasize + d + i];
			float diff = buffer[bidx].actionZ[i] - mean;
			dDelta[b * nodedatasize + d + i] = coeff * diff / (ACTION_SIGMA * ACTION_SIGMA);

		}
		else {
			dDelta[b * nodedatasize + d + i] = nodedata[b * nodedatasize + d + i] - target;
		}
	}
	else {
		float sum = 0.0f;
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
__global__ void tuneweights(int n, int nin, int l, int l1d, int lw, int lsize, int d, int lb, float lr, float* dNodeData, float* dWeights, float* dDelta, float* dBias, const replaybuffer* __restrict__ buffer, int s, int curbatch, int nodedatasize, int* indices, bool isactor, float2* aweights, float2* abias, int t) {

	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float betat1 = 1.0f - powf(beta1, t);
	float betat2 = 1.0f - powf(beta2, t);

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	float biasgrad = 0.0f;
	for (int b = 0; b < curbatch; b++)
		biasgrad += dDelta[b * nodedatasize + d + i];
	float bm = abias[lb + i].x;
	float bv = abias[lb + i].y;
	bm = getmoment(bm, biasgrad / curbatch, beta1);
	abias[lb + i].x = bm;
	float bm_hat = bm / betat1;


	bv = getv(bv, biasgrad / curbatch, beta2);
	abias[lb + i].y = bv;
	float bv_hat = bv / betat2;

	float bupdate = getlr(lr, bm_hat, bv_hat, 1e-6f);
	//	bupdate= clamp(bupdate, -0.05f, 0.05f);
	dBias[lb + i] -= bupdate;
	for (int k = 0; k < nin; k++)
	{
		float wgrad = 0.0f;
		for (int b = 0; b < curbatch; b++)
		{
			int off = b * nodedatasize;
			float nodeVal = (l == 0) ? buffer[indices[s * batchsize + b]].s1[k] : dNodeData[off + l1d + k];
			wgrad += dDelta[off + d + i] * nodeVal;
		}

		float wm = aweights[lw + i * lsize + k].x;
		float wv = aweights[lw + i * lsize + k].y;
		wm = getmoment(wm, wgrad / curbatch, beta1);
		aweights[lw + i * lsize + k].x = wm;
		float wm_hat = wm / betat1;

		wv = getv(wv, wgrad / curbatch, beta2);
		aweights[lw + i * lsize + k].y = wv;
		float wv_hat = wv / betat2;
		float wupdate = getlr(lr, wm_hat, wv_hat, 1e-6f);
		//wupdate = clamp(wupdate, -0.05f, 0.05f);
		dWeights[lw + i * lsize + k] -= wupdate;
	}
}
__device__ float sigmoid(float x) {

	return  1.0f / (1.0f + expf(-x));

}
__device__ float actionTarget(int j, float action) {
	float lo = 0.0f;
	float hi = 0.0f;
	switch (j) {
	case 0: lo = -30.0f; hi = 50.0f; break;
	case 1: lo = -10.0f; hi = 45.0f; break;
	case 2: lo = 0.0f; hi = 150.0f; break;
	case 3: lo = 0.0f; hi = 150.0f; break;
	case 4: lo = -10.0f; hi = 45.0f; break;
	case 5: lo = -10.0f; hi = 45.0f; break;
	case 6: lo = -45.0f; hi = 45.0f; break;
	case 7: lo = -45.0f; hi = 45.0f; break;
	case 8: lo = 0.0f; hi = 140.0f; break;
	case 9: lo = 0.0f; hi = 140.0f; break;
	case 10: lo = -60.0f; hi = 170.0f; break;
	case 11: lo = -60.0f; hi = 170.0f; break;
	case 12: lo = -90.0f; hi = 90.0f; break;
	case 13: lo = -90.0f; hi = 90.0f; break;
	case 14: lo = -90.0f; hi = 90.0f; break;
	case 15: lo = -90.0f; hi = 90.0f; break;
	case 16: lo = -30.0f; hi = 100.0f; break;
	case 17: lo = -30.0f; hi = 100.0f; break;
	}
	return lo + 0.5f * (action + 1.0f) * (hi - lo);
}
__device__ void setJointAction(body& b, int j, float value) {
	switch (j) {
	case 0: b.hipjoints = value; break;
	case 1: b.hipJointSideways = value; break;
	case 2: b.leftElbowJoint = value; break;
	case 3: b.rightElbowJoint = value; break;
	case 4: b.leftHipJointSideways = value; break;
	case 5: b.rightHipJointSideways = value; break;
	case 6: b.leftHipJointTwist = value; break;
	case 7: b.rightHipJointTwist = value; break;
	case 8: b.leftKneeJoint = value; break;
	case 9: b.rightKneeJoint = value; break;
	case 10: b.leftShoulderJoint = value; break;
	case 11: b.rightShoulderJoint = value; break;
	case 12: b.leftShoulderJointSideways = value; break;
	case 13: b.rightShoulderJointSideways = value; break;
	case 14: b.leftShoulderJointTwist = value; break;
	case 15: b.rightShoulderJointTwist = value; break;
	case 16: b.leftUpperLegJoint = value; break;
	case 17: b.rightUpperLegJoint = value; break;
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
	for (int j = 0; j < 18; j++) {
		float mean = nodevals[antityidx(D, j, ci)];
		float z = mean + ACTION_SIGMA * curand_normal(&d_rngstate[ci]);
		float action = tanhf(z);
		buffer[bidx].action[j] = action;
		buffer[bidx].actionZ[j] = z;
		setJointAction(b, j, actionTarget(j, action));
		oldLogprob += tanhGaussianLogProb(z, action, mean);
	}
	d_body[ci] = b;
	buffer[bidx].old_logprob = oldLogprob;
}
__global__ void getlog(int n, int d, int* indices, replaybuffer* buffer, int s, float* nodevals, int nodedatasize) {

	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b >= n)return;
	int off = b * nodedatasize;
	int bidx = indices[s * batchsize + b];
	float logprob = 0.0f;
	for (int j = 0; j < 18; j++) {
		float mean = nodevals[off + d + j];
		logprob += tanhGaussianLogProb(buffer[bidx].actionZ[j], buffer[bidx].action[j], mean);
	}
	buffer[bidx].logprob = logprob;
}
__device__ float d_reward = 0.0f;
__device__ float oldr = 0.0f;
__device__ int id = 0;
__device__ float DT = 1.0f / 120.0f;
__device__ int logframe = 0;
__device__ __forceinline__ void resetBody(body* bodies, int id)
{
	if (bodies == nullptr || id < 0 || id >= d.n)
		return;

	body reset{};
	reset.positionY = 20.0f;
	reset.positonX = 0.0f;
	reset.positonZ = 0.0f;
	reset.alive = true;
	reset.reached = false;

	bodies[id]=reset;
}
__global__ void reward_kernel(int n, int s, replaybuffer* buffer,body* d_body ,float aliver, float deadr, float winr, float feetr,float handr,float headr,float torsor,float reachr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	int bidx = s * d.n + i;

	body c = d_body[i];
	float alivereward = aliver;
	if (!c.alive)alivereward = deadr;
	float dx = c.positonX - d.targetx;
	float dy = c.positonZ - d.targetz;
	float curdist = sqrtf(dx * dx + dy * dy);
	float reached = c.reached ? winr : 0.0f;

	
	float progressreward =reachr*( 1.0f-( curdist / d.maxdisttotarget)) ;
	
	float a = 0.0f;
	if (c.robotLeftFootTouchingGround || c.robotRightFootTouchingGround) {
		a = feetr;
	}
	float b = 0.0f;
	if (c.robotHeadTouchingGround ) {
		b = headr;
	}
	float C = 0.0f;
	if (c.robotTorsoTouchingGround ) {
		C = torsor;
	}
	float d = 0.0f;
		if (c.robotLeftHandTouchingGround||c.robotRightHandTouchingGround) {
			d = torsor;
		}
		float knee = 0.0f;
		if (c.robotLeftLowerLegTouchingGround || c.robotRightLowerLegTouchingGround) {
			knee = -1.0f;
		}

	float reward = progressreward+ alivereward+ reached +a+b+C+d+knee + c.positionY ;


	buffer[bidx].reward = reward;
	
	atomicAdd(&d_reward, reward);
	if (!c.alive || c.reached) {
		
		resetBody(d_body, i);
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
__global__ void firstlayerfrozen(int n, int s, int* indices, const float* __restrict__ weights, const float* __restrict__ bias, float* nodevals, float* d_preact, replaybuffer* buffer, int nodedatasize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y;
	if (i >= n)return;



	float x = bias[idx(0, i)];
	for (int k = 0; k < 40; k++) {
		x += buffer[indices[s * batchsize + b]].s1[k] * weights[idx(0 + i * 40, k)];

	}
	int off = b * nodedatasize + i;
	d_preact[off] = x;
	nodevals[off] = leakyrelu(x);

}
__global__ void solvefrozenlayers(int n, int nin, int w, int D, int b, int p, bool isout, int* indices, const float* __restrict__ dWeights, const float* __restrict__ dBias, float* dNodeData, float* preact, int nodedatasize) {



	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int B = blockIdx.y;
	if (i >= n)return;


	float val = dBias[idx(b, i)];

	for (int j = 0; j < nin; j++) {
		float v = dNodeData[B * nodedatasize + p + j];
		val += v * dWeights[w + i * nin + j];
	}
	int off = B * nodedatasize + D + i;
	preact[off] = val;
	dNodeData[off] = isout ? val : leakyrelu(val);


}
__device__ float normalize(float x,float max,float min){
	return ((x - min) / (max - min)) * 2.0f - 1.0f;
	//[-1,1] bounded for better normalization for values in a rnage
}
__device__ float boolTofloat(bool x) {
	float a = 0.0f;
	if (x)a = 1.0f;
	if (!x)a = 0.0f;
	return a;
}
__global__ void netkernel(int n,body* d_body, const float* __restrict__ weights,
	const float* __restrict__ bias, float* nodevals, const Layer* layer, bool isactor, replaybuffer* buffer, int s, curandState* rng

) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

	

	

		for (int l = 0; l < d.layers; l++) {

			if (l == 0) {

				body b = d_body[i];

				float dx = d.targetx - b.positonX;
				float dy = d.targetz - b.positonZ;
				float dist = sqrtf(dx * dx + dy * dy);

				

			


				float input[40] = { dist / d.maxdisttotarget,
					normalize(b.hipjoints,-30,50),


					normalize(b.hipJointSideways,-10,45),
					normalize(b.leftHipJointSideways,-10,45),
					normalize(b.rightHipJointSideways,-10,45),
					normalize(b.leftHipJointTwist,-45,45),
					normalize(b.rightHipJointTwist,-45,45),

					normalize(b.leftShoulderJoint,-60,170),
					normalize(b.rightShoulderJoint,-60,170),

					normalize(b.leftShoulderJointSideways,-90,90),
					normalize(b.rightShoulderJointSideways,-90,90),

					normalize(b.leftShoulderJointTwist,-90,90),
					normalize(b.rightShoulderJointTwist,-90,90),

					normalize(b.leftElbowJoint,0,150),
					normalize(b.rightElbowJoint,0,150),

					normalize(b.leftUpperLegJoint,-30,100),
					normalize(b.rightUpperLegJoint,-30,100),

					normalize(b.leftKneeJoint,0,140),
					normalize(b.rightKneeJoint,0,140),

					boolTofloat(b.robotHeadTouchingGround),

					boolTofloat(b.robotLeftFootTouchingGround) ,
					boolTofloat(b.robotRightFootTouchingGround) ,

					boolTofloat(b.robotLeftForearmTouchingGround) ,
					boolTofloat(b.robotRightForearmTouchingGround) ,

					boolTofloat(b.robotLeftHandTouchingGround) ,
					boolTofloat(b.robotRightHandTouchingGround) ,

					boolTofloat(b.robotLeftLowerLegTouchingGround) ,
					boolTofloat(b.robotRightLowerLegTouchingGround) ,

					boolTofloat(b.robotLeftUpperArmTouchingGround) ,
					boolTofloat(b.robotRightUpperArmTouchingGround) ,

					boolTofloat(b.robotLeftUpperLegTouchingGround) ,
					boolTofloat(b.robotRightUpperLegTouchingGround) ,

					boolTofloat(b.robotPelvisTouchingGround) ,
					boolTofloat(b.robotTorsoTouchingGround) ,
					tanhf(b.velX/25.0f),
					tanhf(b.velY/25.0f),
					tanhf(b.velZ/25.0f),
					tanhf(b.angleVelX/15.0f),
					tanhf(b.angleVelY/15.0f),
					tanhf(b.angleVelZ/15.0f),



				};
				int insize = 40;
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

				firstlayer(layer[0].Nout, i, 0, didx, 40, input, weights, bias, nodevals);
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
__global__ void getdone(int n, body* b,replaybuffer* buffer,int s) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	int bidx = s * d.n + i;
	if (!b[i].alive || b[i].reached) {
		
		buffer[bidx].done = true;
	}
	else {
		buffer[bidx].done = false;
	}
	
}


void checkdone(int step) {
	getdone << <blocks(robot_count), threads >> > (robot_count, d_body, d_state, step);
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

	int l1blocks = (layerdata[0].Nout + threads - 1) / threads;
	int w = layerdata[0].wIdx;
	dim3 grid(l1blocks, curbatch);

	if (isactor) {
		firstlayerfrozen << <grid, threads >> > (layerdata[0].Nout, s, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, d_state, actor_nodedatasize);
	}
	else {
		firstlayerfrozen << <grid, threads >> > (layerdata[0].Nout, s, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, d_state, critic_nodedatasize);

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
		dim3 grid2(Blocks, curbatch);
		if (isactor) {

			solvefrozenlayers << <grid2, threads >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_actor_weights, d_actor_bias, d_actor_nodvals, d_actor_preact, actor_nodedatasize);
		}
		else {

			solvefrozenlayers << <grid2, threads >> > (layerdata[l].Nout, nin, wl, d, b, p, isout, d_indices, d_critic_weights, d_critic_bias, d_critic_nodvals, d_critic_preact, critic_nodedatasize);
		}



	}
	if (isactor) {
		int outd = layerdata[actor_layers - 1].dIdx;

		getlog << <blocks(curbatch), threads >> > (curbatch, outd, d_indices, d_state, s, d_actor_nodvals, actor_nodedatasize);
	}

}
void reward(int s) {
	reward_kernel << <blocks(robot_count), threads >> > (robot_count, s, d_state,d_body,alive,dead,win,feettouching,handtouching,headtouching,torsotouching,reachtarget );
	


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

		int blocks = (n + threads - 1) / threads;
		dim3 grid(blocks, curBatch);
		if (isactor) {
			compute_delta << <grid, threads >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_actor_nodvals, d_actor_delta, d_actor_weights, d_actor_preact, d_state, s, d_indices, actor_nodedatasize, rolloutstep);
		}
		else {

			compute_delta << <grid, threads >> > (n, d, l1out, l1w, l1nin, l1d, outlayer, isactor,
				d_critic_nodvals, d_critic_delta, d_critic_weights, d_critic_preact, d_state, s, d_indices, critic_nodedatasize, rolloutstep);
		}


	}

	for (int l = layers - 1; l >= 0; l--) {
		int blocks = (layerdata[l].Nout + threads - 1) / threads;
		int nin = layerdata[l].Nin;
		int l1d = (l == 0) ? 0 : layerdata[l - 1].dIdx;
		int lw = layerdata[l].wIdx;
		int lsize = (l == 0) ? inputs : layerdata[l].Nin;
		int d = layerdata[l].dIdx;
		int lb = layerdata[l].bIdx;

		if (isactor) {
			tuneweights << <blocks, threads >> > (layerdata[l].Nout, nin, l, l1d, lw, lsize, d, lb, lr,
				d_actor_nodvals, d_actor_weights, d_actor_delta, d_actor_bias, d_state, s, curBatch, actor_nodedatasize, d_indices, true, actor_adam_weights, actor_adam_bias, adam_step);
		}
		else {
			tuneweights << <blocks, threads >> > (layerdata[l].Nout, nin, l, l1d, lw, lsize, d, lb, lr,
				d_critic_nodvals, d_critic_weights, d_critic_delta, d_critic_bias, d_state, s, curBatch, critic_nodedatasize, d_indices, false, critic_adam_weights, critic_adam_bias, adam_step);
		}



	}


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
		netkernel << <blocks(robot_count), threads >> > (robot_count,d_body,  d_actor_weights, d_actor_bias, d_antity_nodevals, d_actlayer, isactor, d_state, s, d_rngstate);
	}
	else {
		netkernel << <blocks(robot_count), threads >> > (robot_count,d_body, d_critic_weights, d_critic_bias, d_antity_nodevals, d_critlayer, isactor, d_state, s, d_rngstate);

	}

}

void run_network() {
	
	net(step, true);//actor forward pass
	net(step, false);//critic forward pass
	//updaterobot();
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
			adam_step++;



			err = cudaMemcpyFromSymbol(&h_reward, d_reward, sizeof(float));
			cudaMemcpyToSymbol(d_reward, &zero, sizeof(float));
			geterror("reward cpy from symbol", err);
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
	cudaMemcpy(d_indices, shuffled_indices.data(), replaybuffersize * sizeof(int), cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
	geterror("indices copy", err);
}

void allocatenetmem() {
	cudaMalloc(&d_actor_weights, actor_weightbuffersize * sizeof(float));
	cudaMalloc(&d_critic_weights, critic_weightbuffersize * sizeof(float));
	cudaMalloc(&d_actor_bias, actor_biassize * sizeof(float));
	cudaMalloc(&d_critic_bias, critic_biassize * sizeof(float));
	cudaMalloc(&d_actor_nodvals, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_nodvals, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_preact, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_preact, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_actor_delta, hbatchsize * actor_nodedatasize * sizeof(float));
	cudaMalloc(&d_critic_delta, hbatchsize * critic_nodedatasize * sizeof(float));
	cudaMalloc(&d_state, replaybuffersize * sizeof(replaybuffer));
	cudaMalloc(&d_indices, replaybuffersize * sizeof(int));
	cudaMalloc(&d_antity_nodevals, antitynodesize * sizeof(float));
	cudaMalloc(&d_actlayer, actor_layers * sizeof(Layer));
	cudaMalloc(&d_critlayer, critic_layers * sizeof(Layer));
	cudaMalloc(&actor_adam_weights, actor_weightbuffersize * sizeof(float2));
	cudaMalloc(&actor_adam_bias, actor_biassize * sizeof(float2));
	cudaMalloc(&critic_adam_weights, critic_weightbuffersize * sizeof(float2));
	cudaMalloc(&critic_adam_bias, critic_biassize * sizeof(float2));

	cudaMemset(actor_adam_weights, 0.0f, actor_weightbuffersize * sizeof(float2));
	cudaMemset(actor_adam_bias, 0.0f, actor_biassize * sizeof(float2));
	cudaMemset(critic_adam_weights, 0.0f, critic_weightbuffersize * sizeof(float2));
	cudaMemset(critic_adam_bias, 0.0f, critic_biassize * sizeof(float2));

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network malloc failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network malloc done \n");
	}
}
void copywbtogpu() {
	cudaMemcpy(d_actor_weights, actor_weights.data(), actor_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critic_weights, critic_weights.data(), critic_weightbuffersize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_actor_bias, actor_bias.data(), actor_biassize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critic_bias, critic_bias.data(), critic_biassize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_actlayer, actor_layerdata.data(), actor_layers * sizeof(Layer), cudaMemcpyHostToDevice);
	cudaMemcpy(d_critlayer, critic_layerdata.data(), critic_layers * sizeof(Layer), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err) {
		printf("network memcpy failed %s \n", cudaGetErrorString(err));
	}
	else {
		printf("network memcpy done \n");
	}

}
void restart() {
	save_weights();
	
	cudafree();

	printf("memfree on restart \n");
	robot_count = sample_robot_count;
	actor_weightbuffersize = 0;
	critic_weightbuffersize = 0;
	actor_biassize = 0;
	critic_biassize = 0;
	actor_nodedatasize = 0;
	rolloutstep = 0;
	adam_step = 1;
	step = 0;
	gen = 0;
	replaybuffersize = robot_count * 2048;
	critic_nodedatasize = 0;
	actor_layers = 0;
	critic_layers = 0;

	initrobot();

	initnetwork();
	printf("network initialized \n");
	



}
void initnetwork() {
	replaybuffersize -= replaybuffersize % robot_count;
	initlayers();

	initWB(-0.5f, 0.5f);
	allocatenetmem();
	copywbtogpu();
	allocate();
	setmaxdisttotarget();



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
	cudaFree(d_body);


}
