#include <iostream>
#include <cuda.h>
#include "vars.h"
#include "render.h"
#include "network.h"
#include "physics.h"

joint leftShoulder;
float3 rotatepoint(float4 quat, float3 point) {
	float3 result;
	float4 q = quat;
	float3 v = point;

	float x = q.x;
	float y = q.y;
	float z = q.z;
	float w = q.w;

	result.x =
		(1 - 2 * y * y - 2 * z * z) * v.x +
		(2 * x * y - 2 * z * w) * v.y +
		(2 * x * z + 2 * y * w) * v.z;

	result.y =
		(2 * x * y + 2 * z * w) * v.x +
		(1 - 2 * x * x - 2 * z * z) * v.y +
		(2 * y * z - 2 * x * w) * v.z;

	result.z =
		(2 * x * z - 2 * y * w) * v.x +
		(2 * y * z + 2 * x * w) * v.y +
		(1 - 2 * x * x - 2 * y * y) * v.z;

	return result;
}

float3 getworldanchor(rigidbody& part,float3 localanchor){

	float3 rotated = rotatepoint(part.quatrotation, localanchor);

	return part.position + rotated;

}

void initpart(rigidbody& part, float3 pos, float3 halfsize, float mass) {
	part = {};
	part.position = (float3)pos;
	part.halfsize = (float3)halfsize;
	part.mass = (float)mass;
	part.invmass = 1.0f / (float)mass;
	part.quatrotation = { 0,0,0,1 };
}

void initrobot(int n) {
	renderdata.resize(n*60);
	bodydata.resize(n);
	for (int i = 0; i < n; i++) {
		robotbody body = bodydata[i];
		initpart(body.torso, { 0, 35, 0 }, { 2, 4, 1 }, 10.0f);
		initpart(body.head, { 0, 41, 0 }, { 1.5f, 1.5f, 1.5f }, 3.0f);

		initpart(body.leftupperarm, { -3, 35, 0 }, { 1, 2, 1 }, 3.0f);
		initpart(body.leftforearm, { -3, 30.5f, 0 }, { 1, 2, 1 }, 2.0f);

		initpart(body.righttupperarm, { 3, 35, 0 }, { 1, 2, 1 }, 3.0f);
		initpart(body.rightforearm, { 3, 30.5f, 0 }, { 1, 2, 1 }, 2.0f);

		initpart(body.leftthigh, { -1.2f, 28, 0 }, { 1, 3, 1 }, 5.0f);
		initpart(body.leftshin, { -1.2f, 21.5f, 0 }, { 1, 3, 1 }, 4.0f);

		initpart(body.rightthigh, { 1.2f, 28, 0 }, { 1, 3, 1 }, 5.0f);
		initpart(body.rightshin, { 1.2f, 21.5f, 0 }, { 1, 3, 1 }, 4.0f);
		bodydata[i] = body;

		leftShoulder.parent = &bodydata[i].torso;
		leftShoulder.child = &bodydata[i].leftupperarm;

		leftShoulder.parentanchor = {
			-2, 4, 1
		};

		leftShoulder.childanchor = {
			1, 2, 1
		};
	}
	
}
float3 gravity = make_float3(
	0.0f,
	-9.81f,
	0.0f
);
float3 getjointerror(joint& Joint) {
	  return getworldanchor(*Joint.parent, Joint.parentanchor) - getworldanchor(*Joint.child, Joint.childanchor);
}
void solvejointpositon(joint& Joint) {
	
	float3 error = getjointerror(Joint);

	float totalinvmass = Joint.parent->invmass + Joint.child->invmass;
	if (totalinvmass <= 0.0f) {
		printf("total inverese mass is less than zero \n");
		return;
	}

	float parentweight = Joint.parent->invmass / totalinvmass;
	float childweight = Joint.child->invmass / totalinvmass;

	if (!Joint.parent->isstatic) {
		Joint.parent->position += error * parentweight;
	}

	if (!Joint.child->isstatic) {
		Joint.child->position += error * childweight;
	}
}

void applygravity(rigidbody& part) {
	if (part.isstatic)
		return;
	part.velocity += gravity * dt;
	part.position += part.velocity * dt;
	if (part.position.y <= part.halfsize.y) {
		part.position.y = part.halfsize.y;
		part.velocity.y = 0.0f;

	}

}
void updatephysics(int n) {
	
	for (int i = 0; i < n; i++) {
		
		applygravity(bodydata[i].torso);
		applygravity(bodydata[i].head);
		applygravity(bodydata[i].leftupperarm);
		applygravity(bodydata[i].righttupperarm);
		applygravity(bodydata[i].leftforearm);
		applygravity(bodydata[i].rightforearm);
		applygravity(bodydata[i].leftthigh);
		applygravity(bodydata[i].rightthigh);
		applygravity(bodydata[i].leftshin);
		applygravity(bodydata[i].rightshin);
		
		//solvejointpositon(leftShoulder);
	}

	

}