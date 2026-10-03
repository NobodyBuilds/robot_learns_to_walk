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
	p.force = { 0.0f,0.0f };
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
		
		//joints
		getcircle(jointrenderdata[c], getworldanchor(*b.lefthip.parent, b.lefthip.parentanchor), 50.0f, { 1,1,1 });
		getcircle(jointrenderdata[c+1], getworldanchor(*b.lefthip.parent, b.lefthip.parentanchor), 35.0f, { 0,0,0 });

		getcircle(jointrenderdata[c + 2], getworldanchor(*b.leftknee.parent, b.leftknee.parentanchor), 50.0f, { 1,1,1 });
		getcircle(jointrenderdata[c + 3], getworldanchor(*b.leftknee.parent, b.leftknee.parentanchor), 35.0f, { 0,0,0 });

		getcircle(jointrenderdata[c + 4], getworldanchor(*b.rightknee.parent, b.rightknee.parentanchor), 50.0f, { 1,1,1 });
		getcircle(jointrenderdata[c + 5], getworldanchor(*b.rightknee.parent, b.rightknee.parentanchor), 35.0f, { 0,0,0 });

		getcircle(jointrenderdata[c + 6], getworldanchor(*b.leftankle.parent, b.leftankle.parentanchor), 30.0f, { 1,1,1 });
		getcircle(jointrenderdata[c + 7], getworldanchor(*b.leftankle.parent, b.leftankle.parentanchor), 20.0f, { 0,0,0 });

		getcircle(jointrenderdata[c + 8], getworldanchor(*b.rightankle.parent, b.rightankle.parentanchor), 30.0f, { 1,1,1 });
		getcircle(jointrenderdata[c + 9], getworldanchor(*b.rightankle.parent, b.rightankle.parentanchor), 20.0f, { 0,0,0 });

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

void intigratepart(part& p, float dt){
	p.force.y =   -9.81f * p.mass;
	p.vel.x += (p.force.x * p.invmass) * dt;
	p.vel.y += (p.force.y * p.invmass) * dt;
	p.pos.x += p.vel.x * dt;
	p.pos.y += p.vel.y * dt;
	p.angvel += (p.torque * p.invinertia) * dt;
	p.angvel *= 0.999f;
	p.angle += p.angvel * dt * (180.0f / PI);
	p.angle = normalizeangle(p.angle);
	
}

void getcorners(part& p, float2 corners[4]) {

	float hx = p.size.x * 0.5f;
	float hy = p.size.y * 0.5f;

	float angle = p.angle * (PI / 180.0f);

	float c = cosf(angle);
	float s = sinf(angle);

	float2 local[4] = {
		{-hx, -hy},
		{ hx, -hy},
		{ hx,  hy},
		{-hx,  hy}
	};

	for (int i = 0; i < 4; i++)
	{
		corners[i].x =
			p.pos.x + local[i].x * c - local[i].y * s;

		corners[i].y =
			p.pos.y + local[i].x * s + local[i].y * c;
	}

}
float getlowestpoint(part& p) {

	float2 corners[4];

	getcorners(p, corners);

	float lowest = corners[0].y;

	for (int i = 1; i < 4; i++) {
				if (corners[i].y < lowest)
					lowest = corners[i].y;
	}
	return lowest;
}
int getlowestcorner(part& p) {
	float2 corners[4];
	getcorners(p, corners);

	int lowestIndex = 0;

	for(int i=1; i < 4; i++) {
		if(corners[i].y < corners[lowestIndex].y)
			lowestIndex = i;
	}
	return lowestIndex;
}
float2 getfloorcontact(part& p) {
	float2 corners[4];
	getcorners(p, corners);
	int lowestIndex = getlowestcorner(p);
	return corners[lowestIndex];	
}
float2 getcontactoffset(part& p) {

	float2 contact = getfloorcontact(p);

	return {
		contact.x - p.pos.x,
		contact.y - p.pos.y};
}
float2 getcontactvel(part& p,float2 r) {
	float2 v;

	v.x = p.vel.x - p.angvel * r.y;
	v.y = p.vel.y + p.angvel * r.x;
	return v;
}
float getclampedtargetangle(joint& j)
{
	if (j.targetangle < j.minangle)
		return j.minangle;

	if (j.targetangle > j.maxangle)
		return j.maxangle;

	return j.targetangle;
}
int getfloorcontacts(part& p,float2 contact[2],float floorY = 150.0f) {
	float2 corners[4];
	getcorners(p, corners);
	int count = 0;

	for(int i=0; i < 4; i++) {
		if (corners[i].y <= floorY + 0.5f)
		{
			if (count < 2)
			{
				contact[count] = corners[i];
				count++;
			}
		}
	}
	return count;
}
void impulse(part& p,float2 impulse, float2 r) {
	p.vel.x += impulse.x * p.invmass;
	p.vel.y += impulse.y * p.invmass;
	float torqueImpulse = r.x * impulse.y - r.y * impulse.x;

	p.angvel += torqueImpulse * p.invinertia;
}
float solvefloorcontact(part& p, float2 contact) {
	float2 r={
		contact.x - p.pos.x,
		contact.y - p.pos.y
	};
	float2 contactVel = getcontactvel(p, r);
	if (contactVel.y >= 0.0f) return 0.0f;

	float effectiveMass =
		p.invmass +
		(r.x * r.x) * p.invinertia;

	if (effectiveMass == 0.0f)
		return 0.0f;

	float impulseMagnitude =
		-contactVel.y / effectiveMass;

	float2 impulseValue = {
		0.0f,
		impulseMagnitude
	};

	impulse(p, impulseValue, r);

	return impulseMagnitude;
}
void solvefloorfriction(part& p,float2 contact, float friction,float normalimpulse) {
	
	if (normalimpulse <= 0.0f) return;
	float2 r = {
		contact.x - p.pos.x,
		contact.y - p.pos.y
	};
	float2 contactVel = getcontactvel(p, r);
	float tangvel = contactVel.x;

	if (fabsf(tangvel) < 0.0001f) return;

	float effectivemass = p.invmass + (r.y * r.y) * p.invinertia;
	float implse = -tangvel / effectivemass;
	float maxfrciton = friction*normalimpulse;

	implse = fmaxf(-maxfrciton, fminf(implse, maxfrciton));

	float2 imp = { implse, 0.0f };

	impulse(p, imp, r);
}
void floorcolisionpart(part& p,float floorY=150.0f) {
	

	float lowest = getlowestpoint(p);

	if(lowest < floorY) {
		float correction = floorY - lowest;
		p.pos.y += correction;
		
	}

	
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
	float target = getclampedtargetangle(j);
	return normalizeangle( target - angle);
}

void solvecontacts(part& p) {
	float2 contact[2];
	int count = getfloorcontacts(p, contact);
	for (int i = 0; i < count; i++) {
		float normalimpulse = solvefloorcontact(p, contact[i]);

		
		solvefloorfriction(p, contact[i], 0.8f, normalimpulse);

	}
}
void solvejointmotor(joint& j) {
	float error = getjointangleerror(j);

	float strength = fmaxf(0.0f, fminf(j.strength, maxjointstrength));
	float desiredangvel = error * (PI / 180.0f) * strength;

	float relangvel = j.child->angvel - j.parent->angvel;

	float correction = desiredangvel - relangvel;

	float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
	float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
	float totalinvinertia = parentInvInertia + childInvInertia;
	if (totalinvinertia <= 0.0f) return;

	float parentweight = parentInvInertia / totalinvinertia;
	float childweight = childInvInertia / totalinvinertia;

	if (!j.parent->isstatic)
		j.parent->angvel -= correction * parentweight;

	if (!j.child->isstatic)
		j.child->angvel += correction * childweight;

}
void solvejointposition(joint& j)
{
	float2 parentanchor = getworldanchor(*j.parent, j.parentanchor);

	float2 childanchor = getworldanchor(*j.child, j.childanchor);

	float2 error = {
		childanchor.x - parentanchor.x,
		childanchor.y - parentanchor.y
	};
	float2 rp = {
		parentanchor.x - j.parent->pos.x,
		parentanchor.y - j.parent->pos.y
	};

	float2 rc = {
		childanchor.x - j.child->pos.x,
		childanchor.y - j.child->pos.y
	};
	for (int axis = 0; axis < 2; axis++) {

		float2 n;
		if (axis == 0)
			n = { 1.0f, 0.0f };
		else
			n = { 0.0f, 1.0f };

		float C;

		if (axis == 0)
			C = error.x;
		else
			C = error.y;
		if (fabsf(C) < 0.0001f) continue;

		float crossP =
			rp.x * n.y - rp.y * n.x;

		float crossC =
			rc.x * n.y - rc.y * n.x;

		float parentInvMass = j.parent->isstatic ? 0.0f : j.parent->invmass;
		float childInvMass = j.child->isstatic ? 0.0f : j.child->invmass;
		float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
		float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
		float effectiveMass =
			parentInvMass + childInvMass +
			crossP * crossP * parentInvInertia +
			crossC * crossC * childInvInertia;

		if (effectiveMass == 0.0f)
			continue;

		float correction =
			-C / effectiveMass;

		if (!j.parent->isstatic)
		{
			j.parent->pos.x -=
				n.x * correction * parentInvMass;

			j.parent->pos.y -=
				n.y * correction * parentInvMass;
		}

		if (!j.child->isstatic)
		{
			j.child->pos.x +=
				n.x * correction * childInvMass;

			j.child->pos.y +=
				n.y * correction * childInvMass;
		}

		if (!j.parent->isstatic)
		{
			float angleCorrection =
				-crossP *
				correction *
				parentInvInertia;

			j.parent->angle +=
				angleCorrection * (180.0f / PI);
		}

		if (!j.child->isstatic)
		{
			float angleCorrection =
				crossC *
				correction *
				childInvInertia;

			j.child->angle +=
				angleCorrection * (180.0f / PI);

		}
	}

	// Enforce zero relative velocity at the joint anchors after positional projection.
	parentanchor = getworldanchor(*j.parent, j.parentanchor);
	childanchor = getworldanchor(*j.child, j.childanchor);
	rp = { parentanchor.x - j.parent->pos.x, parentanchor.y - j.parent->pos.y };
	rc = { childanchor.x - j.child->pos.x, childanchor.y - j.child->pos.y };
	float parentInvMass = j.parent->isstatic ? 0.0f : j.parent->invmass;
	float childInvMass = j.child->isstatic ? 0.0f : j.child->invmass;
	float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
	float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
	for (int axis = 0; axis < 2; axis++) {
		float2 n = axis == 0 ? float2{ 1.0f, 0.0f } : float2{ 0.0f, 1.0f };
		float crossP = rp.x * n.y - rp.y * n.x;
		float crossC = rc.x * n.y - rc.y * n.x;
		float effectiveMass = parentInvMass + childInvMass +
			crossP * crossP * parentInvInertia + crossC * crossC * childInvInertia;
		if (effectiveMass <= 0.0f) continue;

		float2 parentVelocity = getcontactvel(*j.parent, rp);
		float2 childVelocity = getcontactvel(*j.child, rc);
		float relativeVelocity = (childVelocity.x - parentVelocity.x) * n.x +
			(childVelocity.y - parentVelocity.y) * n.y;
		float impulseMagnitude = -relativeVelocity / effectiveMass;
		float2 jointImpulse = { n.x * impulseMagnitude, n.y * impulseMagnitude };
		if (!j.parent->isstatic)
			impulse(*j.parent, { -jointImpulse.x, -jointImpulse.y }, rp);
		if (!j.child->isstatic)
			impulse(*j.child, jointImpulse, rc);
	}
}
void velocitycorrection(joint& j,float angle) {
	
	float relativeAngVel =
		j.child->angvel - j.parent->angvel;
	if (angle <= j.minangle && relativeAngVel < 0.0f)
	{
	float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
		float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
		float totalinvinertia = parentInvInertia + childInvInertia;

		if (totalinvinertia > 0.0f)
		{
			float parentweight =
				parentInvInertia / totalinvinertia;

			float childweight =
				childInvInertia / totalinvinertia;

			if (!j.parent->isstatic)
				j.parent->angvel +=
				relativeAngVel * parentweight;

			if (!j.child->isstatic)
				j.child->angvel -=
				relativeAngVel * childweight;
		}
	}

	if (angle >= j.maxangle && relativeAngVel > 0.0f)
	{
		float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
		float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
		float totalinvinertia = parentInvInertia + childInvInertia;

		if (totalinvinertia > 0.0f)
		{
			float parentweight =
				parentInvInertia / totalinvinertia;

			float childweight =
				childInvInertia / totalinvinertia;

			if (!j.parent->isstatic)
				j.parent->angvel +=
				relativeAngVel * parentweight;

			if (!j.child->isstatic)
				j.child->angvel -=
				relativeAngVel * childweight;
		}
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
	if (fabsf(correction) >= 0.0001f) {
		float parentInvInertia = j.parent->isstatic ? 0.0f : j.parent->invinertia;
		float childInvInertia = j.child->isstatic ? 0.0f : j.child->invinertia;
		float totalinvinertia = parentInvInertia + childInvInertia;
		if (totalinvinertia > 0.0f) {
			float parentweight = parentInvInertia / totalinvinertia;
			float childweight = childInvInertia / totalinvinertia;
			if (!j.parent->isstatic)
				j.parent->angle -= correction * parentweight;
			if (!j.child->isstatic)
				j.child->angle += correction * childweight;
		}
	}

	velocitycorrection(j, angle);
}


void solvejoints(int i) {
	for (int k = 0; k < 10; k++){
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
	part* parts[7] = {
		&b.torso, &b.leftthigh, &b.rightthigh,
		&b.leftshin, &b.rightshin, &b.leftfoot, &b.rightfoot
	};
	for (int p = 0; p < 7; ++p) {
		floorcolisionpart(*parts[p]);
		solvecontacts(*parts[p]);
	}

	// Resolve non-adjacent OBB contacts. Paired left/right limbs overlap in this
	// sagittal 2D projection, and directly connected pieces are held by joints.
	for (int iteration = 0; iteration < 3; ++iteration) {
		for (int aIndex = 0; aIndex < 7; ++aIndex) {
			for (int bIndex = aIndex + 1; bIndex < 7; ++bIndex) {
				part& a = *parts[aIndex];
				part& c = *parts[bIndex];
				bool connected =
					(&a == &b.torso && (&c == &b.leftthigh || &c == &b.rightthigh)) ||
					(&c == &b.torso && (&a == &b.leftthigh || &a == &b.rightthigh)) ||
					(&a == &b.leftthigh && &c == &b.leftshin) ||
					(&a == &b.rightthigh && &c == &b.rightshin) ||
					(&a == &b.leftshin && &c == &b.leftfoot) ||
					(&a == &b.rightshin && &c == &b.rightfoot);
				bool projectedPair =
					(&a == &b.leftthigh && &c == &b.rightthigh) ||
					(&a == &b.leftshin && &c == &b.rightshin) ||
					(&a == &b.leftfoot && &c == &b.rightfoot);
				if (connected || projectedPair) continue;

				float2 cornersA[4], cornersC[4];
				getcorners(a, cornersA);
				getcorners(c, cornersC);
				float minOverlap = 1.0e30f;
				float2 normal = { 0.0f, 0.0f };
				bool separated = false;
				for (int shape = 0; shape < 2 && !separated; ++shape) {
					float2* corners = shape == 0 ? cornersA : cornersC;
					for (int edge = 0; edge < 2; ++edge) {
						float2 e = {
							corners[edge + 1].x - corners[edge].x,
							corners[edge + 1].y - corners[edge].y
						};
						float length = sqrtf(e.x * e.x + e.y * e.y);
						if (length <= 1.0e-6f) continue;
						float2 axis = { -e.y / length, e.x / length };
						float minA = 1.0e30f, maxA = -1.0e30f;
						float minC = 1.0e30f, maxC = -1.0e30f;
						for (int k = 0; k < 4; ++k) {
							float projectionA = cornersA[k].x * axis.x + cornersA[k].y * axis.y;
							float projectionC = cornersC[k].x * axis.x + cornersC[k].y * axis.y;
							minA = fminf(minA, projectionA);
							maxA = fmaxf(maxA, projectionA);
							minC = fminf(minC, projectionC);
							maxC = fmaxf(maxC, projectionC);
						}
						if (maxA <= minC || maxC <= minA) {
							separated = true;
							break;
						}
						float overlap = fminf(maxA - minC, maxC - minA);
						if (overlap < minOverlap) {
							minOverlap = overlap;
							normal = axis;
						}
					}
				}
				if (separated) continue;

				float2 centerDelta = { c.pos.x - a.pos.x, c.pos.y - a.pos.y };
				if (centerDelta.x * normal.x + centerDelta.y * normal.y < 0.0f) {
					normal.x = -normal.x;
					normal.y = -normal.y;
				}
				float invMassA = a.isstatic ? 0.0f : a.invmass;
				float invMassC = c.isstatic ? 0.0f : c.invmass;
				float invInertiaA = a.isstatic ? 0.0f : a.invinertia;
				float invInertiaC = c.isstatic ? 0.0f : c.invinertia;
				float totalInvMass = invMassA + invMassC;
				if (totalInvMass <= 0.0f) continue;

				float correction = fmaxf(minOverlap - 0.01f, 0.0f) * 0.8f / totalInvMass;
				a.pos.x -= normal.x * correction * invMassA;
				a.pos.y -= normal.y * correction * invMassA;
				c.pos.x += normal.x * correction * invMassC;
				c.pos.y += normal.y * correction * invMassC;
				getcorners(a, cornersA);
				getcorners(c, cornersC);

				float maxProjectionA = -1.0e30f, minProjectionC = 1.0e30f;
				float2 pointA = { 0.0f, 0.0f }, pointC = { 0.0f, 0.0f };
				int countA = 0, countC = 0;
				for (int k = 0; k < 4; ++k) {
					float projectionA = cornersA[k].x * normal.x + cornersA[k].y * normal.y;
					float projectionC = cornersC[k].x * normal.x + cornersC[k].y * normal.y;
					if (projectionA > maxProjectionA + 0.01f) {
						maxProjectionA = projectionA;
						pointA = cornersA[k];
						countA = 1;
					}
					else if (fabsf(projectionA - maxProjectionA) <= 0.01f) {
						pointA.x += cornersA[k].x;
						pointA.y += cornersA[k].y;
						++countA;
					}
					if (projectionC < minProjectionC - 0.01f) {
						minProjectionC = projectionC;
						pointC = cornersC[k];
						countC = 1;
					}
					else if (fabsf(projectionC - minProjectionC) <= 0.01f) {
						pointC.x += cornersC[k].x;
						pointC.y += cornersC[k].y;
						++countC;
					}
				}
				pointA.x /= (float)countA; pointA.y /= (float)countA;
				pointC.x /= (float)countC; pointC.y /= (float)countC;
				float2 contact = { (pointA.x + pointC.x) * 0.5f, (pointA.y + pointC.y) * 0.5f };
				float2 rA = { contact.x - a.pos.x, contact.y - a.pos.y };
				float2 rC = { contact.x - c.pos.x, contact.y - c.pos.y };
				float crossA = rA.x * normal.y - rA.y * normal.x;
				float crossC = rC.x * normal.y - rC.y * normal.x;
				float effectiveMass = totalInvMass + crossA * crossA * invInertiaA + crossC * crossC * invInertiaC;
				if (effectiveMass <= 0.0f) continue;
				float2 velocityA = getcontactvel(a, rA);
				float2 velocityC = getcontactvel(c, rC);
				float normalVelocity = (velocityC.x - velocityA.x) * normal.x +
					(velocityC.y - velocityA.y) * normal.y;
				if (normalVelocity < 0.0f) {
					float normalImpulse = -normalVelocity / effectiveMass;
					float2 impulseValue = { normal.x * normalImpulse, normal.y * normalImpulse };
					if (!a.isstatic)
						impulse(a, { -impulseValue.x, -impulseValue.y }, rA);
					if (!c.isstatic)
						impulse(c, impulseValue, rC);

					velocityA = getcontactvel(a, rA);
					velocityC = getcontactvel(c, rC);
					float2 tangent = { -normal.y, normal.x };
					float tangentVelocity = (velocityC.x - velocityA.x) * tangent.x +
						(velocityC.y - velocityA.y) * tangent.y;
					float tangentCrossA = rA.x * tangent.y - rA.y * tangent.x;
					float tangentCrossC = rC.x * tangent.y - rC.y * tangent.x;
					float tangentMass = totalInvMass + tangentCrossA * tangentCrossA * invInertiaA +
						tangentCrossC * tangentCrossC * invInertiaC;
					if (tangentMass > 0.0f) {
						float tangentImpulse = -tangentVelocity / tangentMass;
						float maxTangentImpulse = 0.6f * normalImpulse;
						tangentImpulse = fmaxf(-maxTangentImpulse, fminf(tangentImpulse, maxTangentImpulse));
						float2 frictionImpulse = { tangent.x * tangentImpulse, tangent.y * tangentImpulse };
						if (!a.isstatic)
							impulse(a, { -frictionImpulse.x, -frictionImpulse.y }, rA);
						if (!c.isstatic)
							impulse(c, frictionImpulse, rC);
					}
				}
			}
		}
	}
}
void intigrate(int i, float dt) {
	body& b = robotdata[i];
	intigratepart(b.torso, dt);
	intigratepart(b.leftthigh, dt);
	intigratepart(b.rightthigh, dt);
	intigratepart(b.leftshin, dt);
	intigratepart(b.rightshin, dt);
	intigratepart(b.leftfoot, dt);
	intigratepart(b.rightfoot, dt);
}


void updateRobot(float dt=1/120.0f) {
	for (int i = 0; i < robot_count; i++) {
		intigrate(i, dt);
		for (int iteration = 0; iteration < 3; ++iteration) {
			solvejoints(i);
			floorcolision(i);
		}
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
	render.circleBatch(jointrenderdata);
}


void freedevmem() {}
