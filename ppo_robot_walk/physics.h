#pragma once
void updatephysics(int n);
float3 rotatepoint(float4 quat, float3 point);
float3 getworldanchor(rigidbody& part, float3 loacalanchor);