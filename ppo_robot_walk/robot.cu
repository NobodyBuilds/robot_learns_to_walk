#include <cuda.h>
#include  <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vars.h"
#include "render.h"
#include "network.h"
#include "norender.h"
#include "physics.h"



void rot(
	float& x, float& y, float& z,
	 const float4& q_in)
{
	float invLen = sqrtf(q_in.x * q_in.x + q_in.y * q_in.y + q_in.z * q_in.z + q_in.w * q_in.w);
	invLen = 1.0f / invLen;
	float4 q = make_float4(q_in.x * invLen, q_in.y * invLen, q_in.z * invLen, q_in.w * invLen);

	float tx = 2.0f * (q.y * z - q.z * y);
	float ty = 2.0f * (q.z * x - q.x * z);
	float tz = 2.0f * (q.x * y - q.y * x);

	float x1 = x + q.w * tx + (q.y * tz - q.z * ty);
	float y1 = y + q.w * ty + (q.z * tx - q.x * tz);
	float z1 = z + q.w * tz + (q.x * ty - q.y * tx);

	x = x1; y = y1; z = z1;
}
void rotate(quadVertex3d& face,const  float4& q)
{
	rot(face.x1, face.y1, face.z1, q);
	rot(face.x2, face.y2, face.z2, q);
	rot(face.x3, face.y3, face.z3, q);
	rot(face.x4, face.y4, face.z4, q);
}
void addpos(quadVertex3d& face, float3 pos) {

	float x, y, z;
	x = pos.x;
	y = pos.y;
	z = pos.z;

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

void getcube(int i,rigidbody part,std::vector<quadVertex3d>& Data,float3 col) {




	quadVertex3d f1;
	quadVertex3d f2;
	quadVertex3d f3;
	quadVertex3d f4;
	quadVertex3d f5;
	quadVertex3d f6;

	
	float sizex = part.halfsize.x ;
	float sizey = part.halfsize.y ;
	float sizez = part.halfsize.z ;

	float r = col.x;
	float g = col.y;
	float b = col.z;
	
	float4 quat = part.quatrotation;
	float3 pos = part.position;

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
	


	rotate(f1, quat);
	rotate(f2, quat);
	rotate(f3, quat);
	rotate(f4, quat);
	rotate(f5, quat);
	rotate(f6, quat);



	addpos(f1, pos);
	addpos(f2, pos);
	addpos(f3, pos);
	addpos(f4, pos);
	addpos(f5, pos);
	addpos(f6, pos);


	Data[i] = f1;
	Data[i + 1] = f2;
	Data[i + 2] = f3;
	Data[i + 3] = f4;
	Data[i + 4] = f5;
	Data[i + 5] = f6;
}

void asemblerobot(int n) {
	for (int i = 0; i < n; i++) {

		robotbody body =bodydata[i];
		int k = i * 60;
		getcube(k, body.torso, renderdata, { 0.3, 0.4, 0.6 });

		getcube(k + 6, body.head, renderdata, { 0.8, 0.7, 0.2 });

		getcube(k + 12, body.leftupperarm, renderdata, { 0.8, 0.2, 0.2 });
		getcube(k + 18, body.leftforearm, renderdata, { 0.8, 0.2, 0.2 });

		getcube(k + 24, body.righttupperarm, renderdata, { 0.8, 0.2, 0.2 });
		getcube(k + 30, body.rightforearm, renderdata, { 0.8, 0.2, 0.2 });

		getcube(k + 36, body.leftthigh, renderdata, { 0.2, 0.7, 0.3 });
		getcube(k + 42, body.leftshin, renderdata, { 0.2, 0.7, 0.3 });

		getcube(k + 48, body.rightthigh, renderdata, { 0.2, 0.7, 0.3 });
		getcube(k + 54, body.rightshin, renderdata, { 0.2, 0.7, 0.3 });

	}
}

void renderRobot() {
	
	updatephysics(robot_count);
	
	asemblerobot(robot_count);
	render.quad3DBatch(renderdata);
}


void freedevmem() {}