#include <iostream>
#include <cfloat>
#include <cmath>
#include <vector>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "vars.h"
#include "render.h"
#include "ui.h"
#include "network.h"


	

void renderUI() {
	//static std::vector<float> learningCurve;
	//static float lastLoss = 0.0f;
	//static bool hasLoss = false;
	//if (std::isfinite(mloss) && (!hasLoss || mloss != lastLoss)) {
	//	learningCurve.push_back(mloss);
	//	lastLoss = mloss;
	//	hasLoss = true;
	//	if (learningCurve.size() > 512) learningCurve.erase(learningCurve.begin());
	//}
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	//bool sync = false;
	ImGui::Begin("debug");
	ImGui::Text("fps: %3f  time: %3f", avgFps, timer);
	
	//if (ImGui::InputInt("num robots", &sample_robot_count, 1, 100)) {
		//restart();
	//}
	//ImGui::DragFloat3("pos", &test.position.x,0.1f,-1000.0f,1000.0f);
	//ImGui::DragFloat4("quat", &test.quatrotation.x,0.1f,-1000.0f,1000.0f);

	
	
	

	

	//
	//ImGui::Text("Gen: %d  rollout gen %d  buffer %d / %d  time:%f", gen, rolloutstep, step * robot_count, replaybuffersize, rollout_time);
	//ImGui::Text("mse: %5f prev %5f ", mloss, oldmloss);
	//ImGui::Text("Learning curve");
	//if (!learningCurve.empty()) {
	//	ImGui::PlotLines("##learning_curve", learningCurve.data(), (int)learningCurve.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 140.0f));
	//}
	//if(ImGui::DragFloat("floorX", &floorX, 0.1f)) { }
	//if(ImGui::DragFloat("floorY", &floorY, 0.1f)) { syncvar(0); }
	//ImGui::DragFloat("floorZ", &floorZ, 0.1f);
	//ImGui::DragFloat("floorRotationX", &floorRotationX, 0.1f);
	//ImGui::DragFloat("floorRotationY", &floorRotationY, 0.1f);
	

	



	/*ImGui::Checkbox("run ai", &run_ai);
	if (run_ai) { ImGui::Checkbox("training", &training); }
	if (training) { ImGui::InputInt("rollout epochs", &rollout_epoch,1,100); }*/

	//ImGui::Text("rewards");
	//ImGui::DragFloat("alive reward", &alive, 0.01f, 0.0f, 100.0f);
	//ImGui::DragFloat("dead reward", &dead, 0.01f, 0.0f, 100.0f);
	//ImGui::DragFloat("win reward", &win, 0.01f, 0.0f, 100.0f);
	//ImGui::DragFloat("reach target reward", &reachtarget, 0.01f, 0.0f, 100.0f);
	//ImGui::DragFloat("feet touching reward", &feettouching, 0.01f, -100.0f, 100.0f);
	//ImGui::DragFloat("hand touching reward", &handtouching, 0.01f, -100.0f, 100.0f);
	//ImGui::DragFloat("head touching reward", &headtouching, 0.01f, -100.0f, 100.0f);
	//ImGui::DragFloat("torso touching reward", &torsotouching, 0.01f, -100.0f, 100.0f);
	//ImGui::Separator();
	

	//ImGui::DragFloat("target x ", &targetx, 0.1f, 0.0f, 1000.0f);
	//ImGui::DragFloat("target z ", &targetz, 0.1f, 0.0f, 1000.0f);
	//ImGui::DragFloat("target size ", &targetradious, 0.1f, 0.0f, 1000.0f);
	//ImGui::DragFloat("target rot x ", &trx, 0.1f, -360.0f, 360.0f);
	//ImGui::DragFloat("target rot y ", &Try, 0.1f, -360.0f, 360.0f);

	//if(ImGui::DragFloat("floorwidth", &floorwidth, 1.0f)) {
	//	pixelx = floorwidth / pixelWidth;
	//	initfloor();
	//}
	//if(ImGui::DragFloat("floorheight", &floorheight, 1.0f)) {
	//	pixely = floorheight / pixelWidth;
	//	initfloor();
	//}
	//if(ImGui::DragFloat("pixelWidth", &pixelWidth, 0.1f)) {
	//	pixelx = floorwidth / pixelWidth;
	//	pixely = floorheight / pixelWidth;
	//	initfloor();
	//}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
