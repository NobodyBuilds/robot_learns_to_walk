#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "norender.h"
#include "vars.h"
#include "render.h"
#include  "network.h"
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;


cudaError_t err;




#if usecpu

std::vector<quadVertex3d>renderdata;
struct robot {
	vector<float> x, y, z;
	vector<float> torsoRX, torsoRY,headRX,headRY;
	vector<float> leftshoulderRX, leftshoulderRY, rightshoulderRX, rightshoulderRY;
	vector<float> leftelbowRX, leftelbowRY, rightelbowRX, rightelbowRY;
	vector<float> leftquadRX, leftquadRY, rightquadRX, rightquadRY;
	vector<float> leftkneeRX, leftkneeRY, rightkneeRX, rightkneeRY;



};
robot robodata;



void rot(float& x, float& y, float& z,
	float rotationX, float rotationY)
{
	float rx = rotationX * 3.14159265f / 180.0f;
	float ry = rotationY * 3.14159265f / 180.0f;

	
	float y1 = y * cos(rx) - z * sin(rx);
	float z1 = y * sin(rx) + z * cos(rx);

	y = y1;
	z = z1;

	
	float x1 = x * cos(ry) + z * sin(ry);
	float z2 = -x * sin(ry) + z * cos(ry);

	x = x1;
	z = z2;
}
void rotate(quadVertex3d& face, float rotx,float roty) {

	rot(face.x1, face.y1, face.z1, rotx, roty);
	rot(face.x2, face.y2, face.z2, rotx, roty);
	rot(face.x3, face.y3, face.z3, rotx, roty);
	rot(face.x4, face.y4, face.z4, rotx, roty);

}
void addpos(quadVertex3d& face, float x, float y, float z) {
	face.x1 += x;
	face.y1 += y;
	face.z1 += z;

	face.x2 += x;
	face.y2 += y;
	face.z2 += z;

	face.x3 += x;
	face.y3 += y;
	face.z3 += z;

	face.x4 += x;
	face.y4 += y;
	face.z4 += z;
}
void resetrobot(int n,robot& r)
{
	r.x.resize(n, 0.0f);
	r.y.resize(n, 30.0f);
	r.z.resize(n, 0.0f);

	r.torsoRX.resize(n, 0.0f);
	r.torsoRY.resize(n, 0.0f);
	r.headRX.resize(n, 0.0f);
	r.headRY.resize(n, 0.0f);

	r.leftshoulderRX.resize(n, 0.0f);
	r.leftshoulderRY.resize(n, 0.0f);
	r.rightshoulderRX.resize(n, 0.0f);
	r.rightshoulderRY.resize(n, 0.0f);

	r.leftelbowRX.resize(n, 0.0f);
	r.leftelbowRY.resize(n, 0.0f);
	r.rightelbowRX.resize(n, 0.0f);
	r.rightelbowRY.resize(n, 0.0f);

	r.leftquadRX.resize(n, 0.0f);
	r.leftquadRY.resize(n, 0.0f);
	r.rightquadRX.resize(n, 0.0f);
	r.rightquadRY.resize(n, 0.0f);

	r.leftkneeRX.resize(n, 0.0f);
	r.leftkneeRY.resize(n, 0.0f);
	r.rightkneeRX.resize(n, 0.0f);
	r.rightkneeRY.resize(n, 0.0f);
}
float3 getangpos(float rotx, float roty, float length, float3 pos) {
	float rx = rotx * 3.14159265f / 180.0f;
	float ry = roty * 3.14159265f / 180.0f;

	float dx = 0.0f;
	float dy = -1.0f;
	float dz = 0.0f;

	float y1 = dy * cosf(rx) - dz * sinf(rx);
	float z1 = dy * sinf(rx) + dz * cosf(rx);
	dy = y1;
	dz = z1;

	float x1 = dx * cosf(ry) + dz * sinf(ry);
	float z2 = -dx * sinf(ry) + dz * cosf(ry);
	dx = x1;
	dz = z2;

	float endx = pos.x + dx * length;
	float endy = pos.y + dy * length;
	float endz = pos.z + dz * length;
	return make_float3(endx, endy, endz);
}
void getcube(vector<quadVertex3d> &Data,int i,float x,float y,float z, float sizex,float sizey,float sizez,float r,float g,float b,float rotx,float roty,bool sidepivot) {
	
	


		quadVertex3d f1;
		quadVertex3d f2;
		quadVertex3d f3;
		quadVertex3d f5;
		quadVertex3d f6;
		quadVertex3d f4;


        sizex = sizex / 2;
        sizey = sizey / 2;
        sizez = sizez / 2;

        if (sidepivot) {

			f1.x1 = -sizex; f1.y1 = 0;         f1.z1 = 0;
			f1.x2 = -sizex; f1.y2 = -2 * sizey; f1.z2 = 0;
			f1.x3 = sizex; f1.y3 = -2 * sizey; f1.z3 = 0;
			f1.x4 = sizex; f1.y4 = 0;         f1.z4 = 0;

			f1.r = r * 0.9f;
			f1.g = g * 0.9f;
			f1.b = b * 0.9f;


			// Back
			f2.x1 = sizex; f2.y1 = 0;         f2.z1 = -sizez;
			f2.x2 = sizex; f2.y2 = -2 * sizey;  f2.z2 = -sizez;
			f2.x3 = -sizex; f2.y3 = -2 * sizey;  f2.z3 = -sizez;
			f2.x4 = -sizex; f2.y4 = 0;         f2.z4 = -sizez;

			f2.r = r * 0.5f;
			f2.g = g * 0.5f;
			f2.b = b * 0.5f;


			// Left
			f3.x1 = -sizex; f3.y1 = 0;         f3.z1 = -sizez;
			f3.x2 = -sizex; f3.y2 = -2 * sizey;  f3.z2 = -sizez;
			f3.x3 = -sizex; f3.y3 = -2 * sizey;  f3.z3 = 0;
			f3.x4 = -sizex; f3.y4 = 0;         f3.z4 = 0;

			f3.r = r * 0.65f;
			f3.g = g * 0.65f;
			f3.b = b * 0.65f;


			// Right
			f4.x1 = sizex; f4.y1 = 0;         f4.z1 = 0;
			f4.x2 = sizex; f4.y2 = -2 * sizey;  f4.z2 = 0;
			f4.x3 = sizex; f4.y3 = -2 * sizey;  f4.z3 = -sizez;
			f4.x4 = sizex; f4.y4 = 0;         f4.z4 = -sizez;

			f4.r = r * 0.8f;
			f4.g = g * 0.8f;
			f4.b = b * 0.8f;


			// Top
			f5.x1 = -sizex; f5.y1 = 0;         f5.z1 = -sizez;
			f5.x2 = -sizex; f5.y2 = 0;         f5.z2 = 0;
			f5.x3 = sizex; f5.y3 = 0;         f5.z3 = 0;
			f5.x4 = sizex; f5.y4 = 0;         f5.z4 = -sizez;

			f5.r = r;
			f5.g = g;
			f5.b = b;


			// Bottom
			f6.x1 = -sizex; f6.y1 = -2 * sizey; f6.z1 = 0;
			f6.x2 = -sizex; f6.y2 = -2 * sizey; f6.z2 = -sizez;
			f6.x3 = sizex; f6.y3 = -2 * sizey; f6.z3 = -sizez;
			f6.x4 = sizex; f6.y4 = -2 * sizey; f6.z4 = 0;

			f6.r = r * 0.4f;
			f6.g = g * 0.4f;
			f6.b = b * 0.4f;
        }
        else {

            // Front
            f1.x1 = -sizex; f1.y1 = sizey; f1.z1 = sizez;
            f1.x2 = -sizex; f1.y2 = -sizey; f1.z2 = sizez;
            f1.x3 = sizex; f1.y3 = -sizey; f1.z3 = sizez;
            f1.x4 = sizex; f1.y4 = sizey; f1.z4 = sizez;

            f1.r = r * 0.9f;
            f1.g = g * 0.9f;
            f1.b = b * 0.9f;


            // Back
            f2.x1 = sizex; f2.y1 = sizey; f2.z1 = -sizez;
            f2.x2 = sizex; f2.y2 = -sizey; f2.z2 = -sizez;
            f2.x3 = -sizex; f2.y3 = -sizey; f2.z3 = -sizez;
            f2.x4 = -sizex; f2.y4 = sizey; f2.z4 = -sizez;

            f2.r = r * 0.5f;
            f2.g = g * 0.5f;
            f2.b = b * 0.5f;


            // Left
            f3.x1 = -sizex; f3.y1 = sizey; f3.z1 = -sizez;
            f3.x2 = -sizex; f3.y2 = -sizey; f3.z2 = -sizez;
            f3.x3 = -sizex; f3.y3 = -sizey; f3.z3 = sizez;
            f3.x4 = -sizex; f3.y4 = sizey; f3.z4 = sizez;

            f3.r = r * 0.65f;
            f3.g = g * 0.65f;
            f3.b = b * 0.65f;


            // Right
            f4.x1 = sizex; f4.y1 = sizey; f4.z1 = sizez;
            f4.x2 = sizex; f4.y2 = -sizey; f4.z2 = sizez;
            f4.x3 = sizex; f4.y3 = -sizey; f4.z3 = -sizez;
            f4.x4 = sizex; f4.y4 = sizey; f4.z4 = -sizez;

            f4.r = r * 0.8f;
            f4.g = g * 0.8f;
            f4.b = b * 0.8f;


            // Top
            f5.x1 = -sizex; f5.y1 = sizey; f5.z1 = -sizez;
            f5.x2 = -sizex; f5.y2 = sizey; f5.z2 = sizez;
            f5.x3 = sizex; f5.y3 = sizey; f5.z3 = sizez;
            f5.x4 = sizex; f5.y4 = sizey; f5.z4 = -sizez;

            f5.r = r;
            f5.g = g;
            f5.b = b;


            // Bottom
            f6.x1 = -sizex; f6.y1 = -sizey; f6.z1 = sizez;
            f6.x2 = -sizex; f6.y2 = -sizey; f6.z2 = -sizez;
            f6.x3 = sizex; f6.y3 = -sizey; f6.z3 = -sizez;
            f6.x4 = sizex; f6.y4 = -sizey; f6.z4 = sizez;

            f6.r = r * 0.4f;
            f6.g = g * 0.4f;
            f6.b = b * 0.4f;
        }


	rotate(f1, rotx, roty);
	rotate(f2, rotx, roty);
	rotate(f3, rotx, roty);
	rotate(f4, rotx, roty);
	rotate(f5, rotx, roty);
	rotate(f6, rotx, roty);



	addpos(f1, x, y, z);
	addpos(f2, x, y, z);
	addpos(f3, x, y, z);
	addpos(f4, x, y, z);
	addpos(f5, x, y, z);
	addpos(f6, x, y, z);


	Data[i]=f1;
	Data[i+1]=f2;
	Data[i+2]=f3;
	Data[i+3]=f4;
	Data[i+4]=f5;
	Data[i+5]=f6;
}
void getrobot(int n,vector<quadVertex3d>& Data,robot& robodata) {
	
	for (int k=0; k < n; k++) {


		float x = robodata.x[k];
		float y = robodata.y[k];
		float z = robodata.z[k];
		

		float sizex = 10.0f*scale;
		float sizey = 20.0f*scale;
		float sizez = 5.0f*scale;
		

		robot h = robodata;
	
		int i = k*60;
		//torso
		getcube(Data,i ,x, y, z, sizex, sizey, sizez, r, g, b, robodata.torsoRX[k], robodata.torsoRY[k], false);
		//head
		float head_size_x = sizex * 0.5f;
		float head_size_y = sizey * 0.25f;
		float head_size_z = sizez;

		float head_off_x = 0.0f;
		float head_off_y = sizey * 0.625f;
		float head_off_z = 0.0f;
		rot(head_off_x, head_off_y, head_off_z, h.torsoRX[k], h.torsoRY[k]);
		float head_x = x + head_off_x;
		float head_y = y + head_off_y;
		float head_z = z + head_off_z;

		getcube(Data,i+6, head_x, head_y, head_z, head_size_x, head_size_y, head_size_z, r, g, b, h.headRX[k] + h.torsoRX[k], h.headRY[k] + h.torsoRY[k], false);
		//shoulder
		float shoulder_size_x = sizex * 0.2f;
		float shoulder_size_y = sizey * 0.5f;
		float shoulder_size_z = sizez * 0.8f;

		// Shoulder offsets relative to torso center
		float lshoulder_off_x = -(sizex * 0.6f);
		float lshoulder_off_y = sizey * 0.5f;
		float lshoulder_off_z = sizez * 0.2f;
		rot(lshoulder_off_x, lshoulder_off_y, lshoulder_off_z, h.torsoRX[k], h.torsoRY[k]);

		float rshoulder_off_x = sizex * 0.6f;
		float rshoulder_off_y = sizey * 0.5f;
		float rshoulder_off_z = sizez * 0.2f;
		rot(rshoulder_off_x, rshoulder_off_y, rshoulder_off_z, h.torsoRX[k], h.torsoRY[k]);

		float lshoulder_x = x + lshoulder_off_x;
		float lshoulder_y = y + lshoulder_off_y;
		float lshoulder_z = z + lshoulder_off_z;

		float rshoulder_x = x + rshoulder_off_x;
		float rshoulder_y = y + rshoulder_off_y;
		float rshoulder_z = z + rshoulder_off_z;

		//left
		getcube(Data,i+12, lshoulder_x, lshoulder_y, lshoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, r, g, b, h.leftshoulderRX[k] + h.torsoRX[k], h.leftshoulderRY[k] + h.torsoRY[k], true);
		//right
		getcube(Data,i+18, rshoulder_x, rshoulder_y, rshoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, r, g, b, h.rightshoulderRX[k] + h.torsoRX[k], h.rightshoulderRY[k] + h.torsoRY[k], true);
		//elbow
		float elbow_size_x = sizex * 0.2f;
		float elbow_size_y = sizey * 0.5f;
		float elbow_size_z = sizez * 0.8f;

		float3 left_spos = make_float3(lshoulder_x, lshoulder_y, lshoulder_z);
		float3 left_elbowpos = getangpos(h.leftshoulderRX[k] + h.torsoRX[k], h.leftshoulderRY[k] + h.torsoRY[k], shoulder_size_y, left_spos);

		float3 right_spos = make_float3(rshoulder_x, rshoulder_y, rshoulder_z);
		float3 right_elbowpos = getangpos(h.rightshoulderRX[k] + h.torsoRX[k], h.rightshoulderRY[k] + h.torsoRY[k], shoulder_size_y, right_spos);

		//left
		getcube(Data,i+24, left_elbowpos.x, left_elbowpos.y, left_elbowpos.z, elbow_size_x, elbow_size_y, elbow_size_z, r, g, b, h.leftelbowRX[k] + h.leftshoulderRX[k] + h.torsoRX[k], h.leftelbowRY[k] + h.leftshoulderRY[k] + h.torsoRY[k], true);
		//right
		getcube(Data,i+30, right_elbowpos.x, right_elbowpos.y, right_elbowpos.z, elbow_size_x, elbow_size_y, elbow_size_z, r, g, b, h.rightelbowRX[k] + h.rightshoulderRX[k] + h.torsoRX[k], h.rightelbowRY[k] + h.rightshoulderRY[k] + h.torsoRY[k], true);
		//upperlegs
		float legs_size_x = sizex * 0.2f;
		float legs_size_y = sizey * 0.5f;
		float legs_size_z = sizez * 0.8f;

		// Leg offsets relative to torso center
		float llegs_off_x = -(sizex * 0.3f);
		float llegs_off_y = -(sizey * 0.5f);
		float llegs_off_z = sizez * 0.2f;
		rot(llegs_off_x, llegs_off_y, llegs_off_z, h.torsoRX[k], h.torsoRY[k]);

		float rlegs_off_x = sizex * 0.3f;
		float rlegs_off_y = -(sizey * 0.5f);
		float rlegs_off_z = sizez * 0.2f;
		rot(rlegs_off_x, rlegs_off_y, rlegs_off_z, h.torsoRX[k], h.torsoRY[k]);

		float llegs_x = x + llegs_off_x;
		float llegs_y = y + llegs_off_y;
		float llegs_z = z + llegs_off_z;

		float rlegs_x = x + rlegs_off_x;
		float rlegs_y = y + rlegs_off_y;
		float rlegs_z = z + rlegs_off_z;

		//left
		getcube(Data,i+36, llegs_x, llegs_y, llegs_z, legs_size_x, legs_size_y, legs_size_z, r, g, b, h.leftquadRX[k] + h.torsoRX[k], h.leftquadRY[k] + h.torsoRY[k], true);
		//right
		getcube(Data,i+42, rlegs_x, rlegs_y, rlegs_z, legs_size_x, legs_size_y, legs_size_z, r, g, b, h.rightquadRX[k] + h.torsoRX[k], h.rightquadRY[k] + h.torsoRY[k], true);
		//lowerlegs

		float lowerlegs_size_x = sizex * 0.2f;
		float lowerlegs_size_y = sizey * 0.5f;
		float lowerlegs_size_z = sizez * 0.8f;

		float3 left_legpos = make_float3(llegs_x, llegs_y, llegs_z);
		float3 left_kneepos = getangpos(h.leftquadRX[k] + h.torsoRX[k], h.leftquadRY[k] + h.torsoRY[k], legs_size_y, left_legpos);

		float3 right_legpos = make_float3(rlegs_x, rlegs_y, rlegs_z);
		float3 right_kneepos = getangpos(h.rightquadRX[k] + h.torsoRX[k], h.rightquadRY[k] + h.torsoRY[k], legs_size_y, right_legpos);

		//left
		getcube(Data,i+48, left_kneepos.x, left_kneepos.y, left_kneepos.z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, r, g, b, h.leftkneeRX[k] + h.leftquadRX[k] + h.torsoRX[k], h.leftkneeRY[k] + h.leftquadRY[k] + h.torsoRY[k], true);
		//right
		getcube(Data,i+54, right_kneepos.x, right_kneepos.y, right_kneepos.z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, r, g, b, h.rightkneeRX[k] + h.rightquadRX[k] + h.torsoRX[k], h.rightkneeRY[k] + h.rightquadRY[k] + h.torsoRY[k], true);
	}


}
#endif

#if usecuda

struct d_robot {
	float3* pos;//x,y,z
	float3* col;//r,b,g
	float4* torsohead;//tx,ty,hx,hy
	float4* shoulders;//lsx,lsy,rsx,rsy
	float4* elbow;//lex,ley,rex,rey
	float4* quad;//lqx,lqy,rqx,rqy
	float4* knee;//lkx,lky,rkx,rky
};

d_robot d_robodata;
quadVertex3d* d_renderdata = nullptr;

static cudaGraphicsResource_t robotvbo = nullptr;
void registervbo(int n) {
	unsigned int vboid = vbo_id.quad3d_instanced_vbo(n*60, 1);
	if (vboid == 0) {
		printf(" vbo id is unintitalized \n");
		return;
	}
	err=cudaGraphicsGLRegisterBuffer(
		&robotvbo,
		vboid,
		cudaGraphicsRegisterFlagsWriteDiscard);
	geterror("glregister", err);
};


void allocatedevmem(int n) {

	cudaMalloc(&d_robodata.pos, n * sizeof(float3));
	cudaMalloc(&d_robodata.col, n * sizeof(float3));
	
	cudaMalloc(&d_robodata.shoulders, n * sizeof(float4));
	cudaMalloc(&d_robodata.elbow, n * sizeof(float4));
	cudaMalloc(&d_robodata.quad, n * sizeof(float4));
	cudaMalloc(&d_robodata.knee, n * sizeof(float4));
	cudaMalloc(&d_robodata.torsohead, n * sizeof(float4));
	cudaMalloc(&d_renderdata, n * 60 * sizeof(quadVertex3d));
	err = cudaGetLastError();
	geterror("mem allocation", err);
	

}
void freedevmem() {
	cudaFree(d_robodata.pos);
	cudaFree(d_robodata.col);

	cudaFree(d_robodata.shoulders);
	cudaFree(d_robodata.elbow);
	cudaFree(d_robodata.quad);
	cudaFree(d_robodata.knee);
	cudaFree(d_robodata.torsohead);

	cudaFree(d_renderdata);
}

__global__ void resetrobodata(int n,d_robot data){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;
	data.pos[i] = {0.0f,15.0f,0.0f};
	data.col[i] = {0.3f,0.4f,0.6f};
	data.shoulders[i] = { 0 };
	data.elbow[i] = { 0 };
	data.knee[i] = { 0 };
	data.quad[i] = { 0 };
	data.torsohead[i] = { 0 };
}
__device__ void d_rot(float& x, float& y, float& z, float rotx, float roty) {
	float rx = rotx * 3.14159265f / 180.0f;
	float ry = roty * 3.14159265f / 180.0f;


	float y1 = y * cos(rx) - z * sin(rx);
	float z1 = y * sin(rx) + z * cos(rx);

	y = y1;
	z = z1;


	float x1 = x * cos(ry) + z * sin(ry);
	float z2 = -x * sin(ry) + z * cos(ry);

	x = x1;
	z = z2;
}
__device__ void d_rotate(quadVertex3d& face, float rotx, float roty) {
	d_rot(face.x1, face.y1, face.z1, rotx, roty);
	d_rot(face.x2, face.y2, face.z2, rotx, roty);
	d_rot(face.x3, face.y3, face.z3, rotx, roty);
	d_rot(face.x4, face.y4, face.z4, rotx, roty);
}
__device__ void d_addpos(quadVertex3d& face, float x, float y, float z) {
	face.x1 += x;
	face.y1 += y;
	face.z1 += z;

	face.x2 += x;
	face.y2 += y;
	face.z2 += z;

	face.x3 += x;
	face.y3 += y;
	face.z3 += z;

	face.x4 += x;
	face.y4 += y;
	face.z4 += z;
}
__device__ float radclamp(float rad, float min, float max) {
	float minr = min * (PI / 180.0f);
	float maxr = max * (PI / 180.0f);
	return  clamp(rad, minr, maxr);
	
}
__device__ float3 d_getangpos(float rotx, float roty, float length, float3 pos) {
	float rx = rotx ;
	float ry = roty;

	float dx = 0.0f;
	float dy = -1.0f;
	float dz = 0.0f;

	float y1 = dy * cosf(rx) - dz * sinf(rx);
	float z1 = dy * sinf(rx) + dz * cosf(rx);
	dy = y1;
	dz = z1;

	float x1 = dx * cosf(ry) + dz * sinf(ry);
	float z2 = -dx * sinf(ry) + dz * cosf(ry);
	dx = x1;
	dz = z2;

	float endx = pos.x + dx * length;
	float endy = pos.y + dy * length;
	float endz = pos.z + dz * length;
	return make_float3(endx, endy, endz);
}
__device__ void d_getcube(quadVertex3d* Data,int i,float x,float y,float z, float sizex,float sizey,float sizez,float r,float g,float b,float rotx,float roty,bool sidepivot) {
	
	


		quadVertex3d f1;
		quadVertex3d f2;
		quadVertex3d f3;
		quadVertex3d f5;
		quadVertex3d f6;
		quadVertex3d f4;


        sizex = sizex / 2;
        sizey = sizey / 2;
        sizez = sizez / 2;

        if (sidepivot) {

			f1.x1 = -sizex; f1.y1 = 0;         f1.z1 = 0;
			f1.x2 = -sizex; f1.y2 = -2 * sizey; f1.z2 = 0;
			f1.x3 = sizex; f1.y3 = -2 * sizey; f1.z3 = 0;
			f1.x4 = sizex; f1.y4 = 0;         f1.z4 = 0;

			f1.r = r * 0.9f;
			f1.g = g * 0.9f;
			f1.b = b * 0.9f;


			// Back
			f2.x1 = sizex; f2.y1 = 0;         f2.z1 = -sizez;
			f2.x2 = sizex; f2.y2 = -2 * sizey;  f2.z2 = -sizez;
			f2.x3 = -sizex; f2.y3 = -2 * sizey;  f2.z3 = -sizez;
			f2.x4 = -sizex; f2.y4 = 0;         f2.z4 = -sizez;

			f2.r = r * 0.5f;
			f2.g = g * 0.5f;
			f2.b = b * 0.5f;


			// Left
			f3.x1 = -sizex; f3.y1 = 0;         f3.z1 = -sizez;
			f3.x2 = -sizex; f3.y2 = -2 * sizey;  f3.z2 = -sizez;
			f3.x3 = -sizex; f3.y3 = -2 * sizey;  f3.z3 = 0;
			f3.x4 = -sizex; f3.y4 = 0;         f3.z4 = 0;

			f3.r = r * 0.65f;
			f3.g = g * 0.65f;
			f3.b = b * 0.65f;


			// Right
			f4.x1 = sizex; f4.y1 = 0;         f4.z1 = 0;
			f4.x2 = sizex; f4.y2 = -2 * sizey;  f4.z2 = 0;
			f4.x3 = sizex; f4.y3 = -2 * sizey;  f4.z3 = -sizez;
			f4.x4 = sizex; f4.y4 = 0;         f4.z4 = -sizez;

			f4.r = r * 0.8f;
			f4.g = g * 0.8f;
			f4.b = b * 0.8f;


			// Top
			f5.x1 = -sizex; f5.y1 = 0;         f5.z1 = -sizez;
			f5.x2 = -sizex; f5.y2 = 0;         f5.z2 = 0;
			f5.x3 = sizex; f5.y3 = 0;         f5.z3 = 0;
			f5.x4 = sizex; f5.y4 = 0;         f5.z4 = -sizez;

			f5.r = r;
			f5.g = g;
			f5.b = b;


			// Bottom
			f6.x1 = -sizex; f6.y1 = -2 * sizey; f6.z1 = 0;
			f6.x2 = -sizex; f6.y2 = -2 * sizey; f6.z2 = -sizez;
			f6.x3 = sizex; f6.y3 = -2 * sizey; f6.z3 = -sizez;
			f6.x4 = sizex; f6.y4 = -2 * sizey; f6.z4 = 0;

			f6.r = r * 0.4f;
			f6.g = g * 0.4f;
			f6.b = b * 0.4f;
        }
        else {

            // Front
            f1.x1 = -sizex; f1.y1 = sizey; f1.z1 = sizez;
            f1.x2 = -sizex; f1.y2 = -sizey; f1.z2 = sizez;
            f1.x3 = sizex; f1.y3 = -sizey; f1.z3 = sizez;
            f1.x4 = sizex; f1.y4 = sizey; f1.z4 = sizez;

            f1.r = r * 0.9f;
            f1.g = g * 0.9f;
            f1.b = b * 0.9f;


            // Back
            f2.x1 = sizex; f2.y1 = sizey; f2.z1 = -sizez;
            f2.x2 = sizex; f2.y2 = -sizey; f2.z2 = -sizez;
            f2.x3 = -sizex; f2.y3 = -sizey; f2.z3 = -sizez;
            f2.x4 = -sizex; f2.y4 = sizey; f2.z4 = -sizez;

            f2.r = r * 0.5f;
            f2.g = g * 0.5f;
            f2.b = b * 0.5f;


            // Left
            f3.x1 = -sizex; f3.y1 = sizey; f3.z1 = -sizez;
            f3.x2 = -sizex; f3.y2 = -sizey; f3.z2 = -sizez;
            f3.x3 = -sizex; f3.y3 = -sizey; f3.z3 = sizez;
            f3.x4 = -sizex; f3.y4 = sizey; f3.z4 = sizez;

            f3.r = r * 0.65f;
            f3.g = g * 0.65f;
            f3.b = b * 0.65f;


            // Right
            f4.x1 = sizex; f4.y1 = sizey; f4.z1 = sizez;
            f4.x2 = sizex; f4.y2 = -sizey; f4.z2 = sizez;
            f4.x3 = sizex; f4.y3 = -sizey; f4.z3 = -sizez;
            f4.x4 = sizex; f4.y4 = sizey; f4.z4 = -sizez;

            f4.r = r * 0.8f;
            f4.g = g * 0.8f;
            f4.b = b * 0.8f;


            // Top
            f5.x1 = -sizex; f5.y1 = sizey; f5.z1 = -sizez;
            f5.x2 = -sizex; f5.y2 = sizey; f5.z2 = sizez;
            f5.x3 = sizex; f5.y3 = sizey; f5.z3 = sizez;
            f5.x4 = sizex; f5.y4 = sizey; f5.z4 = -sizez;

            f5.r = r;
            f5.g = g;
            f5.b = b;


            // Bottom
            f6.x1 = -sizex; f6.y1 = -sizey; f6.z1 = sizez;
            f6.x2 = -sizex; f6.y2 = -sizey; f6.z2 = -sizez;
            f6.x3 = sizex; f6.y3 = -sizey; f6.z3 = -sizez;
            f6.x4 = sizex; f6.y4 = -sizey; f6.z4 = sizez;

            f6.r = r * 0.4f;
            f6.g = g * 0.4f;
            f6.b = b * 0.4f;
        }
		rotx = rotx * (180.0f / PI);
		roty = roty * (180.0f / PI);

	d_rotate(f1, rotx, roty);
	d_rotate(f2, rotx, roty);
	d_rotate(f3, rotx, roty);
	d_rotate(f4, rotx, roty);
	d_rotate(f5, rotx, roty);
	d_rotate(f6, rotx, roty);



	d_addpos(f1, x, y, z);
	d_addpos(f2, x, y, z);
	d_addpos(f3, x, y, z);
	d_addpos(f4, x, y, z);
	d_addpos(f5, x, y, z);
	d_addpos(f6, x, y, z);


	Data[i]=f1;
	Data[i+1]=f2;
	Data[i+2]=f3;
	Data[i+3]=f4;
	Data[i+4]=f5;
	Data[i+5]=f6;
}
__global__ void d_getrobot(int n, quadVertex3d* Data, d_robot robodata,bool init) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= n)return;
	float x, y, z;
	if(init) {
		float spacing = 20.0f;
		int j = ceilf((sqrtf((float)k + 1.0f) - 1.0f) / 2.0f);

		int t = 2 * j + 1;
		int m = t * t;

		int px, py;

		int d = m - k;

		if (d < t)
		{
			px = j - d;
			py = j;
		}
		else if (d < 2 * t)
		{
			px = -j;
			py = j - (d - t);
		}
		else if (d < 3 * t)
		{
			px = -j + (d - 2 * t);
			py = -j;
		}
		else
		{
			px = j;
			py = -j + (d - 3 * t);
		}

		x = px * spacing;
		z = py * spacing;
		y = 15.0f;
		robodata.pos[k].x=x;
		robodata.pos[k].y=y;
		robodata.pos[k].z=z;
	}
	else {

		x = robodata.pos[k].x;
		y = robodata.pos[k].y;
		z = robodata.pos[k].z;
	}




	float sizex = 10.0f * 0.5f;
	float sizey = 20.0f * 0.5f;
	float sizez = 5.0f * 0.5f;

	// Body part color palette (left and right parts share the same colors)
	float torso_r = d_r,   torso_g = d_g,   torso_b = d_b;   // Torso (customizable via UI)
	float head_r  = 0.95f, head_g  = 0.75f, head_b  = 0.15f; // Head (bright amber/gold)

	// Arms (identical color for both Left and Right arms)
	float arm_upper_r = 0.95f, arm_upper_g = 0.45f, arm_upper_b = 0.20f; // Shoulders / upper arms (coral)
	float arm_lower_r = 0.85f, arm_lower_g = 0.35f, arm_lower_b = 0.15f; // Forearms / elbows (terracotta)

	// Legs (identical color for both Left and Right legs)
	float leg_upper_r = 0.20f, leg_upper_g = 0.75f, leg_upper_b = 0.40f; // Thighs / upper legs (emerald green)
	float leg_lower_r = 0.12f, leg_lower_g = 0.60f, leg_lower_b = 0.35f; // Shins / lower legs (forest jade)

	d_robot h = robodata;
	
	h.torsohead[k].z = radclamp(h.torsohead[k].z, -20.0f, 15.0f);
	h.torsohead[k].w = radclamp(h.torsohead[k].w, -75.0f, 75.0f);

	h.shoulders[k].x = radclamp(h.shoulders[k].x, -180.0f, 30.0f);
	h.shoulders[k].y = radclamp(h.shoulders[k].y, -95.0f, 60.0f);
	h.shoulders[k].z = radclamp(h.shoulders[k].z, -180.0f, 30.0f);
	h.shoulders[k].w = radclamp(h.shoulders[k].w, -95.0f, 60.0f);

	h.elbow[k].x = radclamp(h.elbow[k].x, -160.0f, 0.0f);
	h.elbow[k].y = radclamp(h.elbow[k].y, -30.0f, 40.0f);
	h.elbow[k].z = radclamp(h.elbow[k].z, -160.0f, 0.0f);
	h.elbow[k].w = radclamp(h.elbow[k].w, -30.0f, 40.0f);

	h.quad[k].x = radclamp(h.quad[k].x, -95.0f, 60.0f);
	h.quad[k].y = radclamp(h.quad[k].y, -20.0f, 50.0f);
	h.quad[k].z = radclamp(h.quad[k].z, -95.0f, 60.0f);
	h.quad[k].w = radclamp(h.quad[k].w, -20.0f, 50.0f);

	h.knee[k].x = radclamp(h.knee[k].x, 0.0f, 120.0f);
	h.knee[k].y = 0.0f;
	h.knee[k].z = radclamp(h.knee[k].z, 0.0f, 120.0f);
	h.knee[k].w = 0.0f;




	int i = k * 60;
	//torso
	d_getcube(Data, i, x, y, z, sizex, sizey, sizez, torso_r, torso_g, torso_b, robodata.torsohead[k].x, robodata.torsohead[k].y, false);
	//head
	float head_size_x = sizex * 0.5f;
	float head_size_y = sizey * 0.25f;
	float head_size_z = sizez;

	float head_off_x = 0.0f;
	float head_off_y = sizey * 0.625f;
	float head_off_z = 0.0f;
	d_rot(head_off_x, head_off_y, head_off_z, h.torsohead[k].x, h.torsohead[k].y);
	float head_x = x + head_off_x;
	float head_y = y + head_off_y;
	float head_z = z + head_off_z;
	float headrx = h.torsohead[k].z + h.torsohead[k].x;
	float headry = h.torsohead[k].w + h.torsohead[k].y;	

	d_getcube(Data, i + 6, head_x, head_y, head_z, head_size_x, head_size_y, head_size_z, head_r, head_g, head_b,headrx ,headry , false);
	//shoulder
	float shoulder_size_x = sizex * 0.2f;
	float shoulder_size_y = sizey * 0.5f;
	float shoulder_size_z = sizez * 0.8f;

	// Shoulder offsets relative to torso center
	float lshoulder_off_x = -(sizex * 0.6f);
	float lshoulder_off_y = sizey * 0.5f;
	float lshoulder_off_z = sizez * 0.2f;
	d_rot(lshoulder_off_x, lshoulder_off_y, lshoulder_off_z, h.torsohead[k].x, h.torsohead[k].y);

	float rshoulder_off_x = sizex * 0.6f;
	float rshoulder_off_y = sizey * 0.5f;
	float rshoulder_off_z = sizez * 0.2f;
	d_rot(rshoulder_off_x, rshoulder_off_y, rshoulder_off_z, h.torsohead[k].x, h.torsohead[k].y);

	float lshoulder_x = x + lshoulder_off_x;
	float lshoulder_y = y + lshoulder_off_y;
	float lshoulder_z = z + lshoulder_off_z;

	float rshoulder_x = x + rshoulder_off_x;
	float rshoulder_y = y + rshoulder_off_y;
	float rshoulder_z = z + rshoulder_off_z;

	//left shoulder
	d_getcube(Data, i + 12, lshoulder_x, lshoulder_y, lshoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, arm_upper_r, arm_upper_g, arm_upper_b, h.shoulders[k].x + h.torsohead[k].x, h.shoulders[k].y + h.torsohead[k].y, true);
	//right shoulder
	d_getcube(Data, i + 18, rshoulder_x, rshoulder_y, rshoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, arm_upper_r, arm_upper_g, arm_upper_b, h.shoulders[k].z + h.torsohead[k].x, h.shoulders[k].w + h.torsohead[k].y, true);
	//elbow
	float elbow_size_x = sizex * 0.2f;
	float elbow_size_y = sizey * 0.5f;
	float elbow_size_z = sizez * 0.8f;

	float3 left_spos = make_float3(lshoulder_x, lshoulder_y, lshoulder_z);
	float3 left_elbowpos = d_getangpos(h.shoulders[k].x + h.torsohead[k].x, h.shoulders[k].y + h.torsohead[k].y, shoulder_size_y, left_spos);

	float3 right_spos = make_float3(rshoulder_x, rshoulder_y, rshoulder_z);
	float3 right_elbowpos = d_getangpos(h.shoulders[k].z + h.torsohead[k].x, h.shoulders[k].w + h.torsohead[k].y, shoulder_size_y, right_spos);

	//left elbow
	d_getcube(Data, i + 24, left_elbowpos.x, left_elbowpos.y, left_elbowpos.z, elbow_size_x, elbow_size_y, elbow_size_z, arm_lower_r, arm_lower_g, arm_lower_b, h.elbow[k].x + h.shoulders[k].x + h.torsohead[k].x, h.elbow[k].y + h.shoulders[k].y + h.torsohead[k].y, true);
	//right elbow
	d_getcube(Data, i + 30, right_elbowpos.x, right_elbowpos.y, right_elbowpos.z, elbow_size_x, elbow_size_y, elbow_size_z, arm_lower_r, arm_lower_g, arm_lower_b, h.elbow[k].z + h.shoulders[k].z + h.torsohead[k].x, h.elbow[k].w + h.shoulders[k].w + h.torsohead[k].y, true);
	//upperlegs
	float legs_size_x = sizex * 0.2f;
	float legs_size_y = sizey * 0.5f;
	float legs_size_z = sizez * 0.8f;

	// Leg offsets relative to torso center
	float llegs_off_x = -(sizex * 0.3f);
	float llegs_off_y = -(sizey * 0.5f);
	float llegs_off_z = sizez * 0.2f;
	d_rot(llegs_off_x, llegs_off_y, llegs_off_z, h.torsohead[k].x, h.torsohead[k].y);

	float rlegs_off_x = sizex * 0.3f;
	float rlegs_off_y = -(sizey * 0.5f);
	float rlegs_off_z = sizez * 0.2f;
	d_rot(rlegs_off_x, rlegs_off_y, rlegs_off_z, h.torsohead[k].x, h.torsohead[k].y);

	float llegs_x = x + llegs_off_x;
	float llegs_y = y + llegs_off_y;
	float llegs_z = z + llegs_off_z;

	float rlegs_x = x + rlegs_off_x;
	float rlegs_y = y + rlegs_off_y;
	float rlegs_z = z + rlegs_off_z;

	//left thigh
	d_getcube(Data, i + 36, llegs_x, llegs_y, llegs_z, legs_size_x, legs_size_y, legs_size_z, leg_upper_r, leg_upper_g, leg_upper_b, h.quad[k].x + h.torsohead[k].x, h.quad[k].y + h.torsohead[k].y, true);
	//right thigh
	d_getcube(Data, i + 42, rlegs_x, rlegs_y, rlegs_z, legs_size_x, legs_size_y, legs_size_z, leg_upper_r, leg_upper_g, leg_upper_b, h.quad[k].z + h.torsohead[k].x, h.quad[k].w + h.torsohead[k].y, true);
	//lowerlegs

	float lowerlegs_size_x = sizex * 0.2f;
	float lowerlegs_size_y = sizey * 0.5f;
	float lowerlegs_size_z = sizez * 0.8f;

	float3 left_legpos = make_float3(llegs_x, llegs_y, llegs_z);
	float3 left_kneepos = d_getangpos(h.quad[k].x + h.torsohead[k].x, h.quad[k].y + h.torsohead[k].y, legs_size_y, left_legpos);

	float3 right_legpos = make_float3(rlegs_x, rlegs_y, rlegs_z);
	float3 right_kneepos = d_getangpos(h.quad[k].z + h.torsohead[k].x, h.quad[k].w + h.torsohead[k].y, legs_size_y, right_legpos);

	//left shin/knee
	d_getcube(Data, i + 48, left_kneepos.x, left_kneepos.y, left_kneepos.z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, leg_lower_r, leg_lower_g, leg_lower_b, h.knee[k].x + h.quad[k].x + h.torsohead[k].x, h.knee[k].y + h.quad[k].y + h.torsohead[k].y, true);
	//right shin/knee
	d_getcube(Data, i + 54, right_kneepos.x, right_kneepos.y, right_kneepos.z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, leg_lower_r, leg_lower_g, leg_lower_b, h.knee[k].z + h.quad[k].z + h.torsohead[k].x, h.knee[k].w + h.quad[k].w + h.torsohead[k].y, true);
	


}
__global__ void getinputs(int n,d_robot data)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)return;

	d_robot d = data;
	d.col[i].x = d_r;
	d.col[i].y = d_g;
	d.col[i].z = d_b;
	//d.pos[i].x = d_x;
	//d.pos[i].y = d_y;
	//d.pos[i].z = d_z;

	d.torsohead[i].x = d_tosorx;
	d.torsohead[i].y = d_tosory;
	d.torsohead[i].z = d_headx;
	d.torsohead[i].w = d_heady;

	d.shoulders[i].x = d_leftshoulx;
	d.shoulders[i].y = d_leftshouly;
	d.shoulders[i].z = d_rightshoulx;
	d.shoulders[i].w = d_rightshouly;

	d.elbow[i].x = d_leftelbowx;
	d.elbow[i].y = d_leftelbowy;
	d.elbow[i].z = d_rightelbowx;
	d.elbow[i].w = d_rightelbowy;

	d.quad[i].x = d_leftuplegx;
	d.quad[i].y = d_leftuplegy;
	d.quad[i].z = d_rightuplegx;
	d.quad[i].w = d_rightuplegy;

	d.knee[i].x = d_leftkneex;
	d.knee[i].y = d_leftkneey;
	d.knee[i].z = d_rightkneex;
	d.knee[i].w = d_rightkneey;
}


void uploaddata(int n) {


	err =cudaGraphicsMapResources(1, &robotvbo, 0);
	geterror("map res", err);
	size_t bytes = 0;

	err=cudaGraphicsResourceGetMappedPointer((void**)&d_renderdata, &bytes, robotvbo);
	geterror("mapping", err);
	getinputs<<<blocks(n), threads >> >(n,d_robodata);
	d_getrobot << <blocks(n), threads >> > (n, d_renderdata, d_robodata,false);
	err=cudaGraphicsUnmapResources(1, &robotvbo, 0);
	geterror("unmapping", err);

	render.quad3DBatchInterop3D(n*60, 1);
}
#endif

void initrobot() {

#if usecpu 
	resetrobot(1,robodata);
	renderdata.resize(100);
	getrobot(1,renderdata,robodata);
#endif
#if usecuda
	allocatedevmem(robot_count);
	registervbo(robot_count);
	resetrobodata << <blocks(robot_count), threads >> > (robot_count, d_robodata);
	d_getrobot << <blocks(robot_count), threads >> > (robot_count, d_renderdata, d_robodata,true);
#endif
	printf("robot init complete \n");
}



void renderRobot() {
#if usecpu 
	robodata.x[0] = x;
	robodata.y[0] = y;
	robodata.z[0] = z;

	robodata.torsoRX[0] = h_tosorx;
	robodata.torsoRY[0] = h_tosory;

	robodata.headRX[0] = h_headx;
	robodata.headRY[0] = h_heady;

	robodata.leftshoulderRX[0] = h_leftshoulx;
	robodata.leftshoulderRY[0] = h_leftshouly;

	robodata.rightshoulderRX[0] = h_rightshoulx;
	robodata.rightshoulderRY[0] = h_rightshouly;

	robodata.leftelbowRX[0] = h_leftelbowx;
	robodata.leftelbowRY[0] = h_leftelbowy;

	robodata.rightelbowRX[0] = h_rightelbowx;
	robodata.rightelbowRY[0] = h_rightelbowy;

	robodata.leftquadRX[0] = h_leftuplegx;
	robodata.leftquadRY[0] = h_leftuplegy;

	robodata.rightquadRX[0] = h_rightuplegx;
	robodata.rightquadRY[0] = h_rightuplegy;

	robodata.leftkneeRX[0] = h_leftkneex;
	robodata.leftkneeRY[0] = h_leftkneey;

	robodata.rightkneeRX[0] = h_rightkneex;
	robodata.rightkneeRY[0] = h_rightkneey;
		
	getrobot(1,renderdata, robodata);
	render.quad3DBatch(renderdata);
#endif
#if usecuda
	
	cudaMemcpyToSymbol(d_x, &x, sizeof(float));
	cudaMemcpyToSymbol(d_y, &y, sizeof(float));
	cudaMemcpyToSymbol(d_z, &z, sizeof(float));
	cudaMemcpyToSymbol(d_scale, &scale, sizeof(float));
	cudaMemcpyToSymbol(d_r, &r, sizeof(float));
	cudaMemcpyToSymbol(d_g, &g, sizeof(float));
	cudaMemcpyToSymbol(d_b, &b, sizeof(float));

	cudaMemcpyToSymbol(d_tosorx, &h_tosorx, sizeof(float));
	cudaMemcpyToSymbol(d_tosory, &h_tosory, sizeof(float));

	cudaMemcpyToSymbol(d_headx, &h_headx, sizeof(float));
	cudaMemcpyToSymbol(d_heady, &h_heady, sizeof(float));

	cudaMemcpyToSymbol(d_leftshoulx, &h_leftshoulx, sizeof(float));
	cudaMemcpyToSymbol(d_leftshouly, &h_leftshouly, sizeof(float));

	cudaMemcpyToSymbol(d_rightshoulx, &h_rightshoulx, sizeof(float));
	cudaMemcpyToSymbol(d_rightshouly, &h_rightshouly, sizeof(float));

	cudaMemcpyToSymbol(d_leftelbowx, &h_leftelbowx, sizeof(float));
	cudaMemcpyToSymbol(d_leftelbowy, &h_leftelbowy, sizeof(float));

	cudaMemcpyToSymbol(d_rightelbowx, &h_rightelbowx, sizeof(float));
	cudaMemcpyToSymbol(d_rightelbowy, &h_rightelbowy, sizeof(float));

	cudaMemcpyToSymbol(d_leftuplegx, &h_leftuplegx, sizeof(float));
	cudaMemcpyToSymbol(d_leftuplegy, &h_leftuplegy, sizeof(float));

	cudaMemcpyToSymbol(d_rightuplegx, &h_rightuplegx, sizeof(float));
	cudaMemcpyToSymbol(d_rightuplegy, &h_rightuplegy, sizeof(float));

	cudaMemcpyToSymbol(d_leftkneex, &h_leftkneex, sizeof(float));
	cudaMemcpyToSymbol(d_leftkneey, &h_leftkneey, sizeof(float));

	cudaMemcpyToSymbol(d_rightkneex, &h_rightkneex, sizeof(float));
	cudaMemcpyToSymbol(d_rightkneey, &h_rightkneey, sizeof(float));
	uploaddata(robot_count);
#endif
}