#include <iostream>
#include "norender.h"
#include "vars.h"
#include "render.h"


std::vector<quadVertex3d>data;
//inline float x1 = -10.0f, y1 = 5.0f, z1 = 10.0f;
//inline float x2 = 10.0f, y2 = 5.0f, z2 = 10.0f;
//inline float x3 = 10.0f, y3 = -5.0f, z3 = 10.0f;
//inline float x4 = -10.0f, y4 = -5.0f, z4 = 10.0f;
//
//// ========================
//// BACK
//// ========================
//inline float x5 = 10.0f, y5 = 5.0f, z5 = -10.0f;
//inline float x6 = -10.0f, y6 = 5.0f, z6 = -10.0f;
//inline float x7 = -10.0f, y7 = -5.0f, z7 = -10.0f;
//inline float x8 = 10.0f, y8 = -5.0f, z8 = -10.0f;
//
//// ========================
//// LEFT
//// ========================
//inline float x9 = -10.0f, y9 = 5.0f, z9 = -10.0f;
//inline float x10 = -10.0f, y10 = 5.0f, z10 = 10.0f;
//inline float x11 = -10.0f, y11 = -5.0f, z11 = 10.0f;
//inline float x12 = -10.0f, y12 = -5.0f, z12 = -10.0f;
//
//// ========================
//// RIGHT
//// ========================
//inline float x13 = 10.0f, y13 = 5.0f, z13 = 10.0f;
//inline float x14 = 10.0f, y14 = 5.0f, z14 = -10.0f;
//inline float x15 = 10.0f, y15 = -5.0f, z15 = -10.0f;
//inline float x16 = 10.0f, y16 = -5.0f, z16 = 10.0f;
//
//// ========================
//// TOP
//// ========================
//inline float x17 = -10.0f, y17 = 5.0f, z17 = -10.0f;
//inline float x18 = 10.0f, y18 = 5.0f, z18 = -10.0f;
//inline float x19 = 10.0f, y19 = 5.0f, z19 = 10.0f;
//inline float x20 = -10.0f, y20 = 5.0f, z20 = 10.0f;
//
//// ========================
//// BOTTOM
//// ========================
//inline float x21 = -10.0f, y21 = -5.0f, z21 = 10.0f;
//inline float x22 = 10.0f, y22 = -5.0f, z22 = 10.0f;
//inline float x23 = 10.0f, y23 = -5.0f, z23 = -10.0f;
//inline float x24 = -10.0f, y24 = -5.0f, z24 = -10.0f;

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

void getcube(std::vector<quadVertex3d> &data,float x,float y,float z, float width,float height,float r,float g,float b,float rotx,float roty,bool sidepivot) {


		quadVertex3d f1;
		quadVertex3d f2;
		quadVertex3d f3;
		quadVertex3d f5;
		quadVertex3d f6;
		quadVertex3d f4;

	if (sidepivot) {

		//sets pivot /origion point to sideward of the cube
		f1.x1 = -width; f1.y1 = height; f1.z1 = 0;
		f1.x2 = width; f1.y2 = height; f1.z2 = 0;
		f1.x3 = width; f1.y3 = -height; f1.z3 = 0;
		f1.x4 = -width; f1.y4 = -height; f1.z4 = 0;

		f1.r = r;
		f1.g = g;
		f1.b = b;




		f2.x1 = width; f2.y1 = height; f2.z1 = -2 * width;
		f2.x2 = -width; f2.y2 = height; f2.z2 = -2 * width;
		f2.x3 = -width; f2.y3 = -height; f2.z3 = -2 * width;
		f2.x4 = width; f2.y4 = -height; f2.z4 = -2 * width;

		f2.r = r;
		f2.g = g;
		f2.b = b;




		f3.x1 = -width; f3.y1 = height; f3.z1 = -2 * width;
		f3.x2 = -width; f3.y2 = height; f3.z2 = 0;
		f3.x3 = -width; f3.y3 = -height; f3.z3 = 0;
		f3.x4 = -width; f3.y4 = -height; f3.z4 = -2 * width;

		f3.r = r;
		f3.g = g;
		f3.b = b;


	


		f4.x1 = width; f4.y1 = height; f4.z1 = 0;
		f4.x2 = width; f4.y2 = height; f4.z2 = -2 * width;
		f4.x3 = width; f4.y3 = -height; f4.z3 = -2 * width;
		f4.x4 = width; f4.y4 = -height; f4.z4 = 0;

		f4.r = r;
		f4.g = g;
		f4.b = b;


		


		f5.x1 = -width; f5.y1 = height; f5.z1 = -2 * width;
		f5.x2 = width; f5.y2 = height; f5.z2 = -2 * width;
		f5.x3 = width; f5.y3 = height; f5.z3 = 0;
		f5.x4 = -width; f5.y4 = height; f5.z4 = 0;

		f5.r = r;
		f5.g = g;
		f5.b = b;


		

		f6.x1 = -width; f6.y1 = -height; f6.z1 = 0;
		f6.x2 = width; f6.y2 = -height; f6.z2 = 0;
		f6.x3 = width; f6.y3 = -height; f6.z3 = -2 * width;
		f6.x4 = -width; f6.y4 = -height; f6.z4 = -2 * width;

		f6.r = r;
		f6.g = g;
		f6.b = b;

	}
	else {




		f1.x1 = -width; f1.y1 = height; f1.z1 = width;
		f1.x2 = width; f1.y2 = height; f1.z2 = width;
		f1.x3 = width; f1.y3 = -height; f1.z3 = width;
		f1.x4 = -width; f1.y4 = -height; f1.z4 = width;
		f1.r = r; f1.g = g; f1.b = b;

		f2.x1 = width; f2.y1 = height; f2.z1 = -width;
		f2.x2 = -width; f2.y2 = height; f2.z2 = -width;
		f2.x3 = -width; f2.y3 = -height; f2.z3 = -width;
		f2.x4 = width; f2.y4 = -height; f2.z4 = -width;
		f2.r = r; f2.g = g; f2.b = b;


		f3.x1 = -width; f3.y1 = height; f3.z1 = -width;
		f3.x2 = -width; f3.y2 = height; f3.z2 = width;
		f3.x3 = -width; f3.y3 = -height; f3.z3 = width;
		f3.x4 = -width; f3.y4 = -height; f3.z4 = -width;
		f3.r = r; f3.g = g; f3.b = b;


		f4.x1 = width; f4.y1 = height; f4.z1 = width;
		f4.x2 = width; f4.y2 = height; f4.z2 = -width;
		f4.x3 = width; f4.y3 = -height; f4.z3 = -width;
		f4.x4 = width; f4.y4 = -height; f4.z4 = width;
		f4.r = r; f4.g = g; f4.b = b;


		f5.x1 = -width; f5.y1 = height; f5.z1 = -width;
		f5.x2 = width; f5.y2 = height; f5.z2 = -width;
		f5.x3 = width; f5.y3 = height; f5.z3 = width;
		f5.x4 = -width; f5.y4 = height; f5.z4 = width;
		f5.r = r; f5.g = g; f5.b = b;


		f6.x1 = -width; f6.y1 = -height; f6.z1 = width;
		f6.x2 = width; f6.y2 = -height; f6.z2 = width;
		f6.x3 = width; f6.y3 = -height; f6.z3 = -width;
		f6.x4 = -width; f6.y4 = -height; f6.z4 = -width;
		f6.r = r; f6.g = g; f6.b = b;

	}

	addpos(f1, x, y, z);
	addpos(f2, x, y, z);
	addpos(f3, x, y, z);
	addpos(f4, x, y, z);
	addpos(f5, x, y, z);
	addpos(f6, x, y, z);

	rotate(f1, rotx, roty);
	rotate(f2, rotx, roty);
	rotate(f3, rotx, roty);
	rotate(f4, rotx, roty);
	rotate(f5, rotx, roty);
	rotate(f6, rotx, roty);
}

void initrobot() {
	
}
void renderRobot() {
	render.quad3D(x1, Y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, 0.4, 0.1, 0.1);
}