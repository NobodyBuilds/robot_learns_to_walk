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
	
	ImGui::Begin("debug");
	ImGui::Text("fps: %3f  time: %3f", avgFps, timer);
	
	
	//if (ImGui::InputInt("num robots", &sample_robot_count, 1, 100)) {
		//restart();
	//}

	
	
	ImGui::DragFloat("test x", &tx, 1.0f);
	
	ImGui::DragFloat("camera x", &cx, 1.0f);
	ImGui::DragFloat("camera y", &cy, 1.0f);
	
	drawhelperui();
	
	
	//
	//ImGui::Text("Gen: %d  rollout gen %d  buffer %d / %d  time:%f", gen, rolloutstep, step * robot_count, replaybuffersize, rollout_time);
	//ImGui::Text("mse: %5f prev %5f ", mloss, oldmloss);
	//ImGui::Text("Learning curve");
	//if (!learningCurve.empty()) {
	//	ImGui::PlotLines("##learning_curve", learningCurve.data(), (int)learningCurve.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 140.0f));
	//}
	//if(ImGui::DragFloat("floorX", &floorX, 0.1f)) { }
	//if(ImGui::DragFloat("floorY", &floorY, 0.1f)) { syncvar(0); }
	ImGui::DragFloat("floorx", &floorx, 0.1f);
	ImGui::DragFloat("floory", &floory, 0.1f);
	ImGui::DragFloat("floor size", &fsize, 0.1f);
	
	



	/*ImGui::Checkbox("run ai", &run_ai);
	if (run_ai) { ImGui::Checkbox("training", &training); }
	if (training) { ImGui::InputInt("rollout epochs", &rollout_epoch,1,100); }*/

	//ImGui::Text("rewards");
	//ImGui::DragFloat("alive reward", &alive, 0.01f, 0.0f, 100.0f);
	//ImGui::DragFloat("dead reward", &dead, 0.01f, 0.0f, 100.0f);

	//ImGui::Separator();
	

	
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
