#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "norender.h"
#include <iostream>
#include <vector>
void renderfloor();
void initfloor();
void initrobot(int n);
void renderRobot();
void regdummyquad();