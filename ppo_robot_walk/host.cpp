#include <iostream>
#include "norender.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "D:\visual_studio\noRender\glfw-3.4.bin.WIN64\include\GLFW\glfw3.h"
#include "vars.h"
#include"render.h"
#include "ui.h"
int main()
{
	
	noRender.createWindow(1900, 1200, "PPO Robot Walk", 1);
	noRender.setup3D();
	noRender.setupCamera();
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();

	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(noRender.getWindowHandle(), true);
	const char* glsl_version = "#version 330";
	ImGui_ImplOpenGL3_Init(glsl_version);
	noRender.movementSpeed = 20.0f;
	initfloor();
	initrobot();
	while(noRender.isWindowOpen())
	{
		noRender.updateCamera();
		noRender.setInputBlocked(io.WantCaptureMouse || io.WantCaptureKeyboard);
		noRender.pollEvents();
		noRender.clearScreen(0.1f, 0.1f, 0.1f);
		updaterobot();
		renderfloor();
		renderRobot();
		renderUI();
		noRender.swapBuffers();
	}
	noRender.closeWindow();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	return 0;
}