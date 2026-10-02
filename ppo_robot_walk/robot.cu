#include <cuda.h>
#include  <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "vars.h"
#include "render.h"
#include "network.h"
#include "norender.h"
#include "ui.h"

#define maxjointstrength 2.0f

std::vector<circlevertex2d> jointdata;
std::vector<quadvertex2d> dummyquad;
std::vector<circlevertex2d> dummycircle;
std::vector<circlevertex2d> jointrenderdata;

struct part {
	float3 col;
	float angvel,mass,inertia,invmass,invinertia,torque;
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
};
std::vector<body> robotdata;
float dx = 0.0f;
float dy = 0.0f;

float dr, dg, db;
float dw, dh;
float drot = 0.0f;


void initpart(float2 pos, float2 size, float3 col,float mass, part& p) {
	p.pos = pos;
	p.size = size;
	p.col = col;
	p.vel = { 0.0f,0.0f };
	p.angle = 0.0f;
	p.mass = mass;
	p.inertia = (1.0f / 12.0f) * mass * (size.x * size.x + size.y * size.y);
	p.invmass = 1.0f / mass;
	p.invinertia = 1.0f / p.inertia;
	p.torque = 0.0f;
	p.angvel = 0.0f;
}
void initjoint(part* parent, part* child, float2 parentanchor, float2 childanchor,float2 minmax,  joint& j) {
	j.parent = parent;
	j.child = child;
	j.parentanchor = parentanchor;
	j.childanchor = childanchor;
	j.targetangle = 0.0f;
	j.strength = 0.0f;
	j.minangle = minmax.x	;
	j.maxangle = minmax.y;
}
void setrobotdata(int n) {
	for (int i = 0; i < n; i++) {
		

		initpart({ 500.0f,520.0f }, { 200,70 }, { 0,1,0 }, 100.0f, robotdata[i].torso);
		initpart({ 500.0f,410.0f }, { 40,150 }, { 1,0,0 }, 50.0f, robotdata[i].leftthigh);
		initpart({ 500.0f,410.0f }, { 40,150 }, { 1,0,0 }, 50.0f, robotdata[i].rightthigh);
		initpart({ 500.0f,260.0f }, { 40,150 }, { 0,0,1 }, 30.0f, robotdata[i].leftshin);
		initpart({ 500.0f,260.0f }, { 40,150 }, { 0,0,1 }, 30.0f, robotdata[i].rightshin);
		initpart({ 500.0f,185.0f }, { 80,40 }, { 1,1,0 }, 20.0f, robotdata[i].leftfoot);
		initpart({ 500.0f,185.0f }, { 80,40 }, { 1,1,0 }, 20.0f, robotdata[i].rightfoot);
		initjoint(&robotdata[i].torso, &robotdata[i].leftthigh, { 0.0f,-35.0f }, { 0.0f,75.0f }, {-60,60}, robotdata[i].lefthip);
		initjoint(&robotdata[i].torso, &robotdata[i].rightthigh, { 0.0f,-35.0f }, { 0.0f,75.0f }, {-60,60}, robotdata[i].righthip);
		initjoint(&robotdata[i].leftthigh, &robotdata[i].leftshin, { 0.0f,-75.0f }, { 0.0f,75.0f }, {-90,90}, robotdata[i].leftknee);
		initjoint(&robotdata[i].rightthigh, &robotdata[i].rightshin, { 0.0f,-75.0f }, { 0.0f,75.0f }, {-90,90}, robotdata[i].rightknee);
		initjoint(&robotdata[i].leftshin, &robotdata[i].leftfoot, { 0.0f,-75.0f }, { 0.0f,0.0f }, {-10,10}, robotdata[i].leftankle);
		initjoint(&robotdata[i].rightshin, &robotdata[i].rightfoot, { 0.0f,-75.0f }, { 0.0f,0.0f }, {-10,10}, robotdata[i].rightankle);
	}
}
void initrobot(int n) {
	renderdata.resize(n*7);
	jointrenderdata.resize(n* 10);
	robotdata.resize(n);
	setrobotdata(n);
	dragfloat("left hip strength", &robotdata[0].lefthip.strength, 0.1f);
	dragfloat("left hip angle", &robotdata[0].lefthip.targetangle, 0.1f);
	dragfloat("right hip strength", &robotdata[0].righthip.strength, 0.1f);
	dragfloat("right hip angle", &robotdata[0].righthip.targetangle, 0.1f);
	dragfloat("left knee strength", &robotdata[0].leftknee.strength, 0.1f);
	dragfloat("left knee angle", &robotdata[0].leftknee.targetangle, 0.1f);
	dragfloat("right knee strength", &robotdata[0].rightknee.strength, 0.1f);
	dragfloat("right knee angle", &robotdata[0].rightknee.targetangle, 0.1f);
	dragfloat("left ankle strength", &robotdata[0].leftankle.strength, 0.1f);
	dragfloat("left ankle angle", &robotdata[0].leftankle.targetangle, 0.1f);
	dragfloat("right ankle strength", &robotdata[0].rightankle.strength, 0.1f);
	dragfloat("right ankle angle", &robotdata[0].rightankle.targetangle, 0.1f);
	dummyquad.clear();
	dummycircle.clear();
	
}

void drawdummyquad(quadvertex2d& quad, float2 pos, float2 size, float3 col, float rot) {
	quad.x = pos.x;
	quad.y = pos.y;
	quad.width = size.x;
	quad.height = size.y;
	quad.r = col.x;
	quad.g = col.y;
	quad.b = col.z;
	quad.rotation = rot;
}

void getcube( quadvertex2d& cube,float2 pos,float2 size,float rot,float3 col) {
	cube.x = pos.x;
	cube.y = pos.y;

	cube.width = size.x;
	cube.height = size.y;

	cube.rotation = rot ;
	cube.r = col.x;
	cube.g = col.y;
	cube.b = col.z;
}
void getcircle(circlevertex2d& circle, float2 pos, float size, float3 col) {
	circle.x = pos.x;
	circle.y = pos.y;
	circle.size = size;
	circle.r = col.x;
	circle.g = col.y;
	circle.b = col.z;
}
void asemblerobot(int n) {
	for (int i = 0; i < n; i++) {
		int k = i * 7;
		int c = i * 10;
		body& b = robotdata[i];

		
		getcube(renderdata[k], { b.torso.pos.x,b.torso.pos.y }, { b.torso.size.x,b.torso.size.y }, b.torso.angle, { b.torso.col.x,b.torso.col.y,b.torso.col.z });
		getcube(renderdata[k + 1], { b.leftthigh.pos.x ,b.leftthigh.pos.y }, { b.leftthigh.size.x,b.leftthigh.size.y }, b.leftthigh.angle, { b.leftthigh.col.x,b.leftthigh.col.y,b.leftthigh.col.z });
		getcube(renderdata[k + 2], { b.rightthigh.pos.x ,b.rightthigh.pos.y }, { b.rightthigh.size.x,b.rightthigh.size.y }, b.rightthigh.angle, { b.rightthigh.col.x,b.rightthigh.col.y,b.rightthigh.col.z });
		getcube(renderdata[k + 3], { b.leftshin.pos.x  ,b.leftshin.pos.y }, { b.leftshin.size.x,b.leftshin.size.y }, b.leftshin.angle, { b.leftshin.col.x,b.leftshin.col.y,b.leftshin.col.z });
		getcube(renderdata[k + 4], { b.rightshin.pos.x  ,b.rightshin.pos.y }, { b.rightshin.size.x,b.rightshin.size.y }, b.rightshin.angle, { b.rightshin.col.x,b.rightshin.col.y,b.rightshin.col.z });
		getcube(renderdata[k + 5], { b.leftfoot.pos.x  ,b.leftfoot.pos.y }, { b.leftfoot.size.x,b.leftfoot.size.y }, b.leftfoot.angle, { b.leftfoot.col.x,b.leftfoot.col.y,b.leftfoot.col.z });
		getcube(renderdata[k + 6], { b.rightfoot.pos.x  ,b.rightfoot.pos.y }, { b.rightfoot.size.x,b.rightfoot.size.y }, b.rightfoot.angle, { b.rightfoot.col.x,b.rightfoot.col.y,b.rightfoot.col.z });
		
	}
}
void regdummyquad() {
	if (dummyQuad) {
		quadvertex2d a = { 0 };
		a.x = dx;
		a.y = dy;
		a.width = dw;
		a.height = dh;
		a.r = dr;
		a.g = dg;
		a.b = db;
		a.rotation = drot;


		dummyquad.push_back(a);
		
	}
	if(dummyCircle) {
		circlevertex2d c;
		c.x = dx;
		c.y = dy;
		c.r = dr;
		c.g = dg;
		c.b = db;
		c.size = dw;
		dummycircle.push_back(c);
	}
}

float normalizeangle(float angle)
{
	while (angle > 180.0f)
		angle -= 360.0f;

	while (angle < -180.0f)
		angle += 360.0f;

	return angle;
}

void intigratepart(part& p,float dt=1/120.0f){
	p.force.y =   -9.81f * p.mass;
	p.vel.x += (p.force.x * p.invmass) * dt;
	p.vel.y += (p.force.y * p.invmass) * dt;
	p.pos.x += p.vel.x * dt;
	p.pos.y += p.vel.y * dt;
	p.angvel += (p.torque * p.invinertia) * dt;
	p.angvel *= 0.95f;
	p.angle += p.angvel * dt;
	p.angle = normalizeangle(p.angle);
	
}
void floorcolisionpart(part& p,float floorY=150.0f) {
	if (p.pos.y - p.size.y*0.5f < floorY) {
		p.pos.y = floorY + p.size.y*0.5f ;
		p.vel.y = 0.0f;
	}
}
float2 getworldanchor(part& p, float2 localanchor) {

	float radians = p.angle * (PI / 180.0f);
	float c = cosf(radians);
	float s = sinf(radians);

	float2 world;
	world.x = localanchor.x * c - localanchor.y * s;
	world.y = localanchor.x * s + localanchor.y * c;

	world.x += p.pos.x;
	world.y += p.pos.y;

	return world;
}
float2 getjointerror(joint& j)
{
	float2 parentworld =
		getworldanchor(*j.parent, j.parentanchor);

	float2 childworld =
		getworldanchor(*j.child, j.childanchor);

	return {
		parentworld.x - childworld.x,
		parentworld.y - childworld.y
	};
}
float getjointangle(joint& j)
{
	return normalizeangle(j.child->angle - j.parent->angle);
}
float getjointangleerror(joint& j)
{
	float angle = getjointangle(j);
	return normalizeangle( j.targetangle - angle);
}

void solvejointmotor(joint& j) {
	float error = getjointangleerror(j);

	float desiredangvel = error * (j.strength>maxjointstrength?maxjointstrength:j.strength);

	float relangvel = j.child->angvel - j.parent->angvel;

	float correction = desiredangvel - relangvel;

	float totalinvinertia = j.parent->invinertia + j.child->invinertia;

	float parentweight = j.parent->invinertia / totalinvinertia;
	float childweight = j.child->invinertia / totalinvinertia;

	if (!j.parent->isstatic)
		j.parent->angvel -= correction * parentweight;

	if (!j.child->isstatic)
		j.child->angvel += correction * childweight;

}
void solvejointposition(joint& j)
{
	float2 error = getjointerror(j);

	float totalinvmass =
		j.parent->invmass +
		j.child->invmass;

	if (totalinvmass == 0.0f)
		return;

	float parentweight =
		j.parent->invmass / totalinvmass;

	float childweight =
		j.child->invmass / totalinvmass;

	if (!j.parent->isstatic)
	{
		j.parent->pos.x -= error.x * parentweight;
		j.parent->pos.y -= error.y * parentweight;
		j.parent->vel.x -= error.x * parentweight;
		j.parent->vel.y -= error.y * parentweight;
	}

	if (!j.child->isstatic)
	{
		j.child->pos.x += error.x * childweight;
		j.child->pos.y += error.y * childweight;
		j.child->vel.x += error.x * childweight;
		j.child->vel.y += error.y * childweight;
	}
}
void solvejointlimits(joint& j)
{
	float angle = getjointangle(j);
	float target = angle;
	if (angle < j.minangle) {
		target = j.minangle;
	}
	else if (angle > j.maxangle) {
		target = j.maxangle;
	}
	float correction = target - angle;	
	if (correction == 0.0f) return;

	float totalinvmass = j.parent->invmass + j.child->invmass;
	if (totalinvmass == 0.0f) return;

	float parentweight = j.parent->invmass / totalinvmass;
	float childweight = j.child->invmass / totalinvmass;

	if (!j.parent->isstatic)
		j.parent->angle -= correction * parentweight;

	if (!j.child->isstatic)
		j.child->angle += correction * childweight;
}

void solvejoints(int i) {
	for (int k = 0; k < 1; k++){
		body& b = robotdata[i];
	solvejointposition(b.lefthip);
	solvejointposition(b.righthip);
	solvejointposition(b.leftknee);
	solvejointposition(b.rightknee);
	solvejointposition(b.leftankle);
	solvejointposition(b.rightankle);

	solvejointlimits(b.lefthip);
	solvejointlimits(b.righthip);
	solvejointlimits(b.leftknee);
	solvejointlimits(b.rightknee);
	solvejointlimits(b.leftankle);
	solvejointlimits(b.rightankle);

	solvejointmotor(b.lefthip);
	solvejointmotor(b.righthip);
	solvejointmotor(b.leftknee);
	solvejointmotor(b.rightknee);
	solvejointmotor(b.leftankle);
	solvejointmotor(b.rightankle);

}
}
void floorcolision(int i) {
	body& b = robotdata[i];
	floorcolisionpart(b.torso);
	floorcolisionpart(b.leftthigh);
	floorcolisionpart(b.rightthigh);
	floorcolisionpart(b.leftshin);
	floorcolisionpart(b.rightshin);
	floorcolisionpart(b.leftfoot);
	floorcolisionpart(b.rightfoot);
}
void intigrate(int i) {
	body& b = robotdata[i];
	intigratepart(b.torso);
	intigratepart(b.leftthigh);
	intigratepart(b.rightthigh);
	intigratepart(b.leftshin);
	intigratepart(b.rightshin);
	intigratepart(b.leftfoot);
	intigratepart(b.rightfoot);
}


void updateRobot() {
	for (int i = 0; i < robot_count; i++) {
		intigrate(i);
		floorcolision(i);
		solvejoints(i);
	}
}

void renderRobot() {
	updateRobot();
	//render.quadBatch(dummyquad);
	//render.circleBatch(dummycircle);
	
	//if (dummyQuad) {
	//	render.quad(dx, dy, dr, dg, db, dw, dh, drot);
	//}
	//if (dummyCircle) {
	//	render.circle(dx, dy, dr, dg, db, dw);
	//}

	asemblerobot(robot_count);
	render.quadBatch(renderdata);
	//render.circleBatch(jointrenderdata);
}


void freedevmem() {}
