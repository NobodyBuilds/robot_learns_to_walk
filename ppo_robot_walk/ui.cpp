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

	ImGui::Begin("debug");
	ImGui::Text("hello");
	ImGui::DragFloat("floorX", &floorX, 0.1f);
	ImGui::DragFloat("floorY", &floorY, 0.1f);
	ImGui::DragFloat("floorZ", &floorZ, 0.1f);
	ImGui::DragFloat("floorRotationX", &floorRotationX, 0.1f);
	ImGui::DragFloat("floorRotationY", &floorRotationY, 0.1f);
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