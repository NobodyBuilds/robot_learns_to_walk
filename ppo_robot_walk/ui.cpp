#include <iostream>
#include <cfloat>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
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
	ImGui::Text("GAE squared: %5f prev %5f ", mloss, oldmloss);
	ImGui::Text("dist: %f", tx - 500.0f);
	ImGui::Spacing();
	ImGui::Text("GAE squared history");
	if (!learningCurve.empty()) {
		ImGui::PlotLines("##learning_curve", learningCurve.data(), (int)learningCurve.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 140.0f));
	}
	ImGui::Spacing();

	ImGui::InputInt("num robots", &sample_robot_count, 1, 100);
	if (ImGui::Button("restart")) {
		restart();
	}
	/*if (ImGui::Button("new policy (back up checkpoint)")) {
		restart(true);
	}*/
	ImGui::Checkbox("no accumulator ", &fasttrain);

	ImGui::Spacing();






	ImGui::Checkbox("run ai", &run_ai);
	if (run_ai) { ImGui::Checkbox("training", &training); }
	if (training) { ImGui::InputInt("rollout epochs", &rollout_epoch, 1, 100); }

	ImGui::Checkbox("death on knee", &legalive);
	ImGui::Checkbox("add jitter", &addjitter);
	if (addjitter) {
		ImGui::DragFloat("jitter magnitude", &magnitude, 10.0f, 0.0f, 5000.0f);
		ImGui::DragFloat("jitter threshold", &threshold, 0.01f, 0.05f, 10.0f);
	}
	ImGui::Checkbox("auto reset robot", &autoreset);
	if (!autoreset) {
		if (ImGui::Button("reset robots")) {
			timer -= gentime;
			gen++;
			cx = 950.0f;
			resetrobots();
		}
	}
	ImGui::DragFloat("generation time", &gentime,0.1f);
	ImGui::Spacing();
	ImGui::Text("rewards");
	ImGui::DragFloat("alive reward", &aliver, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("dead penalty", &dead, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("straight reward", &yup, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("forward reward", &forwardr, 1.0f, 0.0f, 500.0f);
	ImGui::DragFloat("target forward speed", &targetForwardSpeed, 1.0f, 10.0f, 500.0f);
	ImGui::DragFloat("reverse penalty", &reversepenalty, 1.0f, 0.0f, 500.0f);
	/*ImGui::DragFloat("forward stride reward", &steppingRewardWeight, 0.01f, 0.0f, 10.0f);
	ImGui::DragFloat("alternating touchdown reward", &alternatingStepRewardWeight, 0.01f, 0.0f, 20.0f);
	ImGui::DragFloat("swing clearance reward / sec", &swingClearanceRewardWeight, 0.01f, 0.0f, 20.0f);
	ImGui::DragFloat("sliding penalty", &slidingPenaltyWeight, 0.01f, 0.0f, 20.0f);*/



	//ImGui::Separator();
	ImGui::Spacing();
	ImGui::Spacing();


	drawhelperui();

	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
namespace {
	// one row per ImGui-bound variable. add a row here = saved AND loaded.
	struct SettingEntry {
		enum Type { Bool, Int, Float };
		const char* key;
		Type type;
		void* ptr;
	};

	const std::vector<SettingEntry>& settingsTable() {
		static const std::vector<SettingEntry> table = {
			{ "sample_robot_count",   SettingEntry::Int,   &sample_robot_count },
			{ "run_ai",               SettingEntry::Bool,  &run_ai },
			{ "training",             SettingEntry::Bool,  &training },
			{ "rollout_epoch",        SettingEntry::Int,   &rollout_epoch },
			{ "legalive",             SettingEntry::Bool,  &legalive },
			{ "alive",                SettingEntry::Float, &aliver },
			{ "dead",                 SettingEntry::Float, &dead },
			{ "yup",                  SettingEntry::Float, &yup },
			{ "forwardr",             SettingEntry::Float, &forwardr },
			{ "targetForwardSpeed",   SettingEntry::Float, &targetForwardSpeed },
			{ "reversepenalty",       SettingEntry::Float, &reversepenalty },
			{ "steppingRewardWeight", SettingEntry::Float, &steppingRewardWeight },
			{ "slidingPenaltyWeight", SettingEntry::Float, &slidingPenaltyWeight },
			{ "alternatingStepRewardWeight", SettingEntry::Float, &alternatingStepRewardWeight },
			{ "swingClearanceRewardWeight", SettingEntry::Float, &swingClearanceRewardWeight },
			{ "addjitter",            SettingEntry::Bool,  &addjitter },
			{ "magnitude",            SettingEntry::Float, &magnitude },
			{ "threshold",            SettingEntry::Float, &threshold },
		};
		return table;
	}
}

void savesettings() {
	std::ofstream file("settings.txt");
	if (!file) {
		printf("could not write settings.txt \n");
		return;
	}
	file << std::setprecision(9); // enough digits for float to round-trip exactly
	for (const SettingEntry& s : settingsTable()) {
		file << s.key << '=';
		switch (s.type) {
		case SettingEntry::Bool:  file << (*static_cast<bool*>(s.ptr) ? 1 : 0); break;
		case SettingEntry::Int:   file << *static_cast<int*>(s.ptr);            break;
		case SettingEntry::Float: file << *static_cast<float*>(s.ptr);          break;
		}
		file << '\n';
	}
}

void loadsettings() {
	std::ifstream file("settings.txt");
	if (!file) {
		printf("settings file not found \n");
		return;
	}
	std::string line;
	while (std::getline(file, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back(); // CRLF safe
		size_t eq = line.find('=');
		if (eq == std::string::npos) continue;
		const std::string key = line.substr(0, eq);
		const std::string val = line.substr(eq + 1);

		for (const SettingEntry& s : settingsTable()) {
			if (key != s.key) continue;
			try {
				switch (s.type) {
				case SettingEntry::Bool:  *static_cast<bool*>(s.ptr) = (std::stoi(val) != 0); break;
				case SettingEntry::Int:   *static_cast<int*>(s.ptr) = std::stoi(val);        break;
				case SettingEntry::Float: *static_cast<float*>(s.ptr) = std::stof(val);        break;
				}
			}
			catch (const std::exception&) {
				printf("bad value for '%s' in settings.txt, keeping default \n", s.key);
			}
			break;
		}
		// unknown keys are ignored; missing keys keep their defaults
	}
}
