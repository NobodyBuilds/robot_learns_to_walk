# Robot Learns to Walk

A 2D legged robot learns to walk in a simulated environment using deep reinforcement learning. This repository contains the robot-specific training code and uses [noRender](https://github.com/NobodyBuilds/noRender) for rendering.

The PPO training builds on the related work in [deep_reinforcement_learning](https://github.com/NobodyBuilds/deep_reinforcement_learning).

## Build

On Windows, run [compile.bat](ppo_robot_walk/compile.bat) to compile the project. The batch file has machine-specific paths, so update them for your setup first. Check the Visual Studio 2022 x64 environment, the CUDA Toolkit `nvcc` command, the project directory, and the paths for noRender, GLFW, GLAD, and Dear ImGui. If compilation fails, verify the `nvcc` and dependency include/library paths.

## Demo

https://github.com/user-attachments/assets/4257dda6-f6a1-4a0f-9c27-bdb9b537b681
