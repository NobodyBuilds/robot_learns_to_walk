#include <iostream>
#include "norender.h"
#include "vars.h"
#include "render.h"
using namespace std;

std::vector<quadVertex3d>renderdata;


struct cube {
	float x, y, z;
	float r, g, b;
	float sizex, sizey, sizez;
	float rotx, roty;
};
struct sphere {
	float x, y, z, r, g, b, size;
};
std::vector<sphere> joint;

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
void resetrobot(robot& r)
{
	r.x.resize(1, 0.0f);
	r.y.resize(1, 30.0f);
	r.z.resize(1, 0.0f);

	r.torsoRX.resize(1, 0.0f);
	r.torsoRY.resize(1, 0.0f);
	r.headRX.resize(1, 0.0f);
	r.headRY.resize(1, 0.0f);

	r.leftshoulderRX.resize(1, 0.0f);
	r.leftshoulderRY.resize(1, 0.0f);
	r.rightshoulderRX.resize(1, 0.0f);
	r.rightshoulderRY.resize(1, 0.0f);

	r.leftelbowRX.resize(1, 0.0f);
	r.leftelbowRY.resize(1, 0.0f);
	r.rightelbowRX.resize(1, 0.0f);
	r.rightelbowRY.resize(1, 0.0f);

	r.leftquadRX.resize(1, 0.0f);
	r.leftquadRY.resize(1, 0.0f);
	r.rightquadRX.resize(1, 0.0f);
	r.rightquadRY.resize(1, 0.0f);

	r.leftkneeRX.resize(1, 0.0f);
	r.leftkneeRY.resize(1, 0.0f);
	r.rightkneeRX.resize(1, 0.0f);
	r.rightkneeRY.resize(1, 0.0f);
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

		float head_x = x;
		float head_y = y + (sizey * 0.625f);
		float head_z = z;

		getcube(Data,i+6, head_x, head_y, head_z, head_size_x, head_size_y, head_size_z, r, g, b, h.headRX[k], h.headRY[k], false);
		//shoulder
		float shoulder_size_x = sizex * 0.2f;;
		float shoulder_size_y = sizey * 0.5f;
		float shoulder_size_z = sizez * 0.8f;

		float shoulder_x = x + (sizex * 0.6f);
		float shoulder_y = y + (sizey * 0.5f);
		float shoulder_z = z + (sizez * 0.2f);

		//left
		getcube(Data,i+12, -shoulder_x, shoulder_y, shoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, r, g, b, h.leftshoulderRX[k], h.leftshoulderRY[k], true);
		//right
		getcube(Data,i+18, shoulder_x, shoulder_y, shoulder_z, shoulder_size_x, shoulder_size_y, shoulder_size_z, r, g, b, h.rightshoulderRX[k], h.rightshoulderRY[k], true);
		//elbow
		float elbow_size_x = sizex * 0.2f;
		float elbow_size_y = sizey * 0.5f;
		float elbow_size_z = sizez * 0.8f;

		float elbow_x = x + (sizex * 0.6f);
		float elbow_y = y;
		float elbow_z = z + (sizez * 0.2f);

		//left
		getcube(Data,i+24, -elbow_x, elbow_y, elbow_z, elbow_size_x, elbow_size_y, elbow_size_z, r, g, b, h.leftelbowRX[k], h.leftelbowRY[k], true);
		//right
		getcube(Data,i+30, elbow_x, elbow_y, elbow_z, elbow_size_x, elbow_size_y, elbow_size_z, r, g, b, h.rightelbowRX[k], h.rightelbowRY[k], true);

		//upperlegs
		float legs_size_x = sizex * 0.2f;
		float legs_size_y = sizey * 0.5f;
		float legs_size_z = sizez * 0.8f;

		float legs_x = x + (sizex * 0.3f);
		float legs_y = y - (sizey * 0.5f);
		float legs_z = z + (sizez * 0.2f);

		//left
		getcube(Data,i+36, -legs_x, legs_y, legs_z, legs_size_x, legs_size_y, legs_size_z, r, g, b, h.leftquadRX[k], h.leftquadRY[k], true);
		//right
		getcube(Data,i+42, legs_x, legs_y, legs_z, legs_size_x, legs_size_y, legs_size_z, r, g, b, h.rightquadRX[k], h.rightquadRY[k], true);

		//lowerlegs

		float lowerlegs_size_x = sizex * 0.2f;
		float lowerlegs_size_y = sizey * 0.5f;
		float lowerlegs_size_z = sizez * 0.8f;

		float lowerlegs_x = x + (sizex * 0.3f);
		float lowerlegs_y = y - sizey;
		float lowerlegs_z = z + (sizez * 0.2f);

		//left
		getcube(Data,i+48 ,-lowerlegs_x, lowerlegs_y, lowerlegs_z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, r, g, b, h.leftkneeRX[k], h.leftkneeRY[k], true);
		//right
		getcube(Data,i+54, lowerlegs_x, lowerlegs_y, lowerlegs_z, lowerlegs_size_x, lowerlegs_size_y, lowerlegs_size_z, r, g, b, h.rightkneeRX[k], h.rightkneeRY[k], true);
	}


}
void initrobot() {
	resetrobot(robodata);
	getrobot(1,renderdata,robodata);
	
}


void updaterobot(std::vector<cube>& body) {
	
}


void renderRobot() {
	
	
		
		
	getrobot(1,renderdata, robodata);
	render.quad3DBatch(renderdata);
	
}