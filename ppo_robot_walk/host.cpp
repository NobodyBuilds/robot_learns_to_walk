#include <iostream>
#include "norender.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "D:\visual_studio\noRender\glfw-3.4.bin.WIN64\include\GLFW\glfw3.h"
#include "vars.h"
#include"render.h"
#include "ui.h"
#include "network.h"
int main()
{
	
	noRender.createWindow(1900, 1200, "PPO Robot Walk", 1);
	noRender.setup2D();
	noRender.setupCamera();
	cx = noRender.getCameraX();
	cy = noRender.getCameraY();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();

	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(noRender.getWindowHandle(), true);
	const char* glsl_version = "#version 330";
	ImGui_ImplOpenGL3_Init(glsl_version);
	noRender.movementSpeed = 20.0f;
	initrobot(robot_count);
	//initnetwork();
	initfloor();
	double lastTime = glfwGetTime();
	double fpsClock = lastTime;
	
	float br = 0.22f;
	float bg = 0.57f;
	float bb = 0.8f;

	while(noRender.isWindowOpen())
	{
		noRender.updateCamera(true);
		noRender.setcamerapos(cx, cy, 0);
		double now = glfwGetTime();
		double frameTime = now - lastTime;
		lastTime = now;
		noRender.setInputBlocked(io.WantCaptureMouse || io.WantCaptureKeyboard);
		noRender.pollEvents();
		noRender.clearScreen(br, bg, bb);
		if (run_ai) {
			//run_network();
		}
		else {

	 	updateRobot();
		}
		timer += static_cast<float>(frameTime);
		renderfloor();
		cx = (tx<950.0f)?cx:tx;
		cx = (tx > 0.0f) ? cx : tx+950.0f;
		renderRobot();
		renderUI();

		if (timer >= gentime) {
			timer -= gentime;
			gen++;
			resetrobots();
		}
		noRender.swapBuffers();

		double elapsed = now - fpsClock;
		fpsClock = now;
		fps = (elapsed > 0.0) ? 1.0 / elapsed : fps;
		fpsTimer += (float)elapsed;
		fpsCount++;

		if (fpsTimer >= 0.5f) {
			avgFps = fpsCount / fpsTimer;
			fpsTimer = 0.f;
			fpsCount = 0;
		}
	}

	//save_weights();
	noRender.closeWindow();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//cudafree();
	freedevmem();
	unregistervbo();

	return 0;
}
