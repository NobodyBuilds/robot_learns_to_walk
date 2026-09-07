@echo off
cd /d "%~dp0"
if not exist "..\third_party\bullet3\build\msvc\bullet3.lib" call build_bullet.bat
if errorlevel 1 exit /b 1
nvcc -std=c++17 -rdc=true -Xcompiler "/std:c++17 /MD" -o D:\visual_studio\ppo_robot_walk\robot_walks.exe host.cpp render.cpp ui.cpp robot.cu D:\glad\src\glad.c ^
  "D:\visual_studio\imgui-1.91.1\imgui.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_draw.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_tables.cpp" ^
  "D:\visual_studio\imgui-1.91.1\imgui_widgets.cpp" ^
  "D:\visual_studio\imgui-1.91.1\backends\imgui_impl_glfw.cpp" ^
  "D:\visual_studio\imgui-1.91.1\backends\imgui_impl_opengl3.cpp" ^
  -I"D:\visual_studio\noRender\noRender" ^
  -I"D:\glad\include" ^
  -I"D:\visual_studio\glfw-3.4.bin.WIN64\include" ^
  -I"D:\visual_studio\imgui-1.91.1" ^
  -I"D:\visual_studio\imgui-1.91.1\backends" ^
  -I"..\third_party\bullet3\src" ^
  -L"..\third_party\bullet3\build\msvc" ^
  -lbullet3 ^
  -L"D:\visual_studio\noRender\x64\Release" ^
  -L"D:\visual_studio\glfw-3.4.bin.WIN64\lib-vc2022" ^
  -lnoRender -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32 ^
  -Xlinker "/LTCG"
if errorlevel 1 (
  echo compilation failed
  pause
  exit /b 1
)
echo done
pause


