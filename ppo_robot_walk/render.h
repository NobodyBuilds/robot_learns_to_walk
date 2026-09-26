#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "norender.h"
#include <iostream>
#include <vector>
void initfloor();
void renderfloor();
void initrobot(int n);
void renderRobot();
void freedevmem();
void resetRobots();
