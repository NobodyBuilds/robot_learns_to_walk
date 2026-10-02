#pragma once
#include <vector>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
void renderUI();

struct imguidata {
	const char* label;
	float* val;
	float speed;
};
inline std::vector<imguidata> imguihelper;

inline void dragfloat(const char* label, float* val, float speed) {
	imguidata a;
	a.label = label;
	a.val = val;
	a.speed = speed;
	imguihelper.push_back(a);
}

inline void drawhelperui() {
	int n = imguihelper.size();
	for (int i = 0; i < n; i++) {
		imguidata a = imguihelper[i];
		ImGui::DragFloat(a.label, a.val, a.speed);
	}
}
