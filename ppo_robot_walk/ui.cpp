#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "vars.h"
#include "render.h"
#include "ui.h"

void renderUI() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	bool sync = false;
	ImGui::Begin("debug");
	ImGui::Text("hello");
	if(ImGui::DragFloat("floorX", &floorX, 0.1f)) { }
	if(ImGui::DragFloat("floorY", &floorY, 0.1f)) { syncvar(0); }
	ImGui::DragFloat("floorZ", &floorZ, 0.1f);
	ImGui::DragFloat("floorRotationX", &floorRotationX, 0.1f);
	ImGui::DragFloat("floorRotationY", &floorRotationY, 0.1f);
	
	ImGui::Separator();
	ImGui::Text("Robot Physics");
	if(ImGui::DragFloat("gravity", &h_gravity, 0.1f)) { syncvar(1); }
	if(ImGui::DragFloat("friction", &h_friction, 0.01f)) { syncvar(2); }
	if(ImGui::DragFloat("drag", &h_drag, 0.01f)) { syncvar(3); }
	if(ImGui::DragFloat("bounce", &h_bounce, 0.01f)) { syncvar(4); }
	if(ImGui::DragFloat("robotScale", &h_robotScale, 0.1f)) { syncvar(5); }

	ImGui::Separator();
	ImGui::Text("Robot Position (Reset)");
	if(ImGui::DragFloat("robotX", &h_robotX, 0.1f)) { syncvar(6); }
	if(ImGui::DragFloat("robotY", &h_robotY, 0.1f)) { syncvar(7); }
	if(ImGui::DragFloat("robotZ", &h_robotZ, 0.1f)) { syncvar(8); }

	ImGui::Separator();
	ImGui::Text("Robot 3D Movement (local acceleration)");
	if(ImGui::DragFloat("robotMoveX", &h_robotMoveX, 0.1f, -25.0f, 25.0f)) { syncvar(29); }
	if(ImGui::DragFloat("robotMoveY", &h_robotMoveY, 0.1f, -25.0f, 25.0f)) { syncvar(30); }
	if(ImGui::DragFloat("robotMoveZ", &h_robotMoveZ, 0.1f, -25.0f, 25.0f)) { syncvar(31); }

	ImGui::Separator();
	ImGui::Text("Robot Joints");
	if(ImGui::DragFloat("leftShoulderJoint", &h_leftShoulderJoint, 1.0f)) { syncvar(9); }
	if(ImGui::DragFloat("rightShoulderJoint", &h_rightShoulderJoint, 1.0f)) { syncvar(10); }
	if(ImGui::DragFloat("leftShoulderJointSideways", &h_leftShoulderJointSideways, 1.0f, -90.0f, 90.0f)) { syncvar(21); }
	if(ImGui::DragFloat("rightShoulderJointSideways", &h_rightShoulderJointSideways, 1.0f, -90.0f, 90.0f)) { syncvar(22); }
	if(ImGui::DragFloat("leftShoulderJointTwist", &h_leftShoulderJointTwist, 1.0f, -90.0f, 90.0f)) { syncvar(23); }
	if(ImGui::DragFloat("rightShoulderJointTwist", &h_rightShoulderJointTwist, 1.0f, -90.0f, 90.0f)) { syncvar(24); }
	if(ImGui::DragFloat("leftElbowJoint", &h_leftElbowJoint, 1.0f)) { syncvar(11); }
	if(ImGui::DragFloat("rightElbowJoint", &h_rightElbowJoint, 1.0f)) { syncvar(12); }
	if(ImGui::DragFloat("hipjoints", &h_hipjoints, 1.0f)) { syncvar(13); }
	if(ImGui::DragFloat("hipJointSideways", &h_hipJointSideways, 1.0f)) { syncvar(14); }
	if(ImGui::DragFloat("leftUpperLegJoint", &h_leftUpperLegJoint, 1.0f)) { syncvar(15); }
	if(ImGui::DragFloat("rightUpperLegJoint", &h_rightUpperLegJoint, 1.0f)) { syncvar(16); }
	if(ImGui::DragFloat("leftHipJointSideways", &h_leftHipJointSideways, 1.0f, -10.0f, 45.0f)) { syncvar(25); }
	if(ImGui::DragFloat("rightHipJointSideways", &h_rightHipJointSideways, 1.0f, -10.0f, 45.0f)) { syncvar(26); }
	if(ImGui::DragFloat("leftHipJointTwist", &h_leftHipJointTwist, 1.0f, -45.0f, 45.0f)) { syncvar(27); }
	if(ImGui::DragFloat("rightHipJointTwist", &h_rightHipJointTwist, 1.0f, -45.0f, 45.0f)) { syncvar(28); }
	if(ImGui::DragFloat("leftKneeJoint", &h_leftKneeJoint, 1.0f)) { syncvar(17); }
	if(ImGui::DragFloat("rightKneeJoint", &h_rightKneeJoint, 1.0f)) { syncvar(18); }

	if(ImGui::DragFloat("floorwidth", &floorwidth, 1.0f)) {
		pixelx = floorwidth / pixelWidth;
		initfloor();
	}
	if(ImGui::DragFloat("floorheight", &floorheight, 1.0f)) {
		pixely = floorheight / pixelWidth;
		initfloor();
	}
	if(ImGui::DragFloat("pixelWidth", &pixelWidth, 0.1f)) {
		pixelx = floorwidth / pixelWidth;
		pixely = floorheight / pixelWidth;
		initfloor();
	}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
