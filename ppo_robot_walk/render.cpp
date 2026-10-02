#include <iostream>
#include "norender.h"
#include "render.h"
#include "vars.h"
spriteData floorsprite;
void initfloor() {
	
	floorsprite = noRender.loadSprite("D:\\visual_studio\\ppo_robot_walk\\ppo_robot_walk\\floor.png");
}

void renderfloor() {
	float step = fsize * 0.95f;
	float leftEdge = cx - noRender.getScreenWidth() * 0.5f;
	float first = floorf(leftEdge / step) * step;   
	int   n = (int)ceilf(noRender.getScreenWidth() / step) + 2;

	for (int i = 0; i < n; i++)
		render.sprite(floorsprite, first + i * step, floory, fsize);
}

