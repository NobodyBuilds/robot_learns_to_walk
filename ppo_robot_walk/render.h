#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "norender.h"
#include <iostream>
#include <vector>
#include "vars.h"
 
void renderfloor();
void initfloor();
void initrobot(int n);
void updateRobot(float dt= DT);
void renderRobot();
void resetrobots();
void freedevmem();
void registervbo(int n);
void unregistervbo();
__device__ void resetrobotidxkernel(body& robotdata);

