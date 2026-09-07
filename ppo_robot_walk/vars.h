#pragma once
#include <vector>
#include "render.h"
#include "norender.h"

//floor variables
inline std::vector<quadtexture2d> floorquads;
inline float floorwidth = 1000.0f;
inline float floorheight = 1000.0f;
inline float pixelWidth = 15.3f;
inline float pixelx = floorwidth / pixelWidth;
inline float pixely = floorheight / pixelWidth;
inline float floorY = 0.0f;
inline float floorZ = 0.0f;
inline float floorX = 0.0f;
inline float floorRotationX = 90.0f;
inline float floorRotationY = 0.0f;
/////