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
	initnetwork();
	double lastTime = glfwGetTime();
	double fpsClock = lastTime;
	spriteData target = noRender.loadSprite("D:\\visual_studio\\ppo_robot_walk\\ppo_robot_walk\\circle.png");
	while(noRender.isWindowOpen())
	{
		double now = glfwGetTime();
		double frameTime = now - lastTime;
		lastTime = now;
		noRender.updateCamera();
		noRender.setInputBlocked(io.WantCaptureMouse || io.WantCaptureKeyboard);
		noRender.pollEvents();
		noRender.clearScreen(0.1f, 0.1f, 0.1f);
		if (run_ai) {
			run_network();
		}
		else {
			updaterobot();
		}
		timer += dt;
		renderfloor();
		render.sprite3D(target, targetx, 0.5f, targetz, targetradious, trx, Try);
		renderRobot();
		renderUI();

		if (timer >= gentime) {
			timer = 0.0f;
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
	save_weights();
	noRender.closeWindow();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	cudafree();
	
	return 0;
}