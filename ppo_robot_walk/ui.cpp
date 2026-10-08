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
	static std::vector<float> learningCurve;
	static float lastLoss = 0.0f;
	static bool hasLoss = false;
	if (std::isfinite(mloss) && (!hasLoss || mloss != lastLoss)) {
		learningCurve.push_back(mloss);
		lastLoss = mloss;
		hasLoss = true;
		if (learningCurve.size() > 512) learningCurve.erase(learningCurve.begin());
	}
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	
	ImGui::Begin("debug");
	ImGui::Text("fps: %3f  time: %3f", avgFps, timer);
	ImGui::Text("Gen: %d  rollout gen %d  buffer %d / %d  time:%f", gen, rolloutstep, step * robot_count, replaybuffersize, rollout_time);
	ImGui::Text("mse: %5f prev %5f ", mloss, oldmloss);
	ImGui::Spacing();
	ImGui::Text("Learning curve");
	if (!learningCurve.empty()) {
		ImGui::PlotLines("##learning_curve", learningCurve.data(), (int)learningCurve.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 140.0f));
	}
	ImGui::Spacing();
	
	ImGui::InputInt("num robots", &sample_robot_count, 1, 100);
	if (ImGui::Button("restart")) {
		restart();
	}
	
	
	ImGui::Spacing();

	
	



	ImGui::Checkbox("run ai", &run_ai);
	if (run_ai) { ImGui::Checkbox("training", &training); }
	if (training) { ImGui::InputInt("rollout epochs", &rollout_epoch,1,100); }

	ImGui::Text("rewards");
	ImGui::DragFloat("alive reward", &alive, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("dead penalty", &dead, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("feet reward", &feettouching, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("straight reward", &yup, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("dist reward", &distr, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("forward reward", &forwardr, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("reverse penalty", &reversepenalty, 0.01f, 0.0f, 1.0f);

	//ImGui::Separator();
	ImGui::Spacing();
	ImGui::Spacing();
	

	drawhelperui();
	
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
