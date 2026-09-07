#include <iostream>
#include "norender.h"
#include "render.h"
#include "vars.h"


void floordata() {
	floorquads.clear();
	bool white = true;
	for (int i = 0; i < (int)pixelx* (int)pixely; i++) {
		
			quadtexture2d t;
			if (white) {
				t.r = 0.2f;
				t.g = 0.2f;
				t.b = 0.2f;
				t.opacity = 1.0f;
				floorquads.push_back(t);
				white = false;
			}
			else {
				t.r = 0.6f;
				t.g = 0.6f;
				t.b = 0.6f;
				t.opacity = 1.0f;
				floorquads.push_back(t);
				white = true;
			}
	
		
	}
}
void initfloor() {
	floordata();
}
void renderfloor() {
	render.textureQuad3D(floorquads, floorX, floorY, floorZ, floorwidth, floorheight, (int)pixelx, (int)pixely,floorRotationX,floorRotationY);
}