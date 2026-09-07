@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0.."
set "BULLET_ROOT=%PROJECT_ROOT%\third_party\bullet3"
set "BUILD_ROOT=%BULLET_ROOT%\build\msvc"
set "OBJECT_ROOT=%BUILD_ROOT%\obj"

if not exist "%OBJECT_ROOT%" mkdir "%OBJECT_ROOT%"
del /q "%OBJECT_ROOT%\*.obj" >nul 2>nul

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

for /r "%BULLET_ROOT%\src\LinearMath" %%F in (*.cpp) do (
    cl /nologo /c /O2 /EHsc /MD /D_CRT_SECURE_NO_WARNINGS /I"%BULLET_ROOT%\src" /I"%BULLET_ROOT%\src\Bullet3Common" /Fo"%OBJECT_ROOT%\%%~nF.obj" "%%F"
    if errorlevel 1 exit /b 1
)
for /r "%BULLET_ROOT%\src\BulletCollision" %%F in (*.cpp) do (
    cl /nologo /c /O2 /EHsc /MD /D_CRT_SECURE_NO_WARNINGS /I"%BULLET_ROOT%\src" /I"%BULLET_ROOT%\src\Bullet3Common" /Fo"%OBJECT_ROOT%\%%~nF.obj" "%%F"
    if errorlevel 1 exit /b 1
)
for /r "%BULLET_ROOT%\src\BulletDynamics" %%F in (*.cpp) do (
    cl /nologo /c /O2 /EHsc /MD /D_CRT_SECURE_NO_WARNINGS /I"%BULLET_ROOT%\src" /I"%BULLET_ROOT%\src\Bullet3Common" /Fo"%OBJECT_ROOT%\%%~nF.obj" "%%F"
    if errorlevel 1 exit /b 1
)

lib /nologo /OUT:"%BUILD_ROOT%\bullet3.lib" "%OBJECT_ROOT%\*.obj"
if errorlevel 1 exit /b 1
echo Bullet Physics static library built: %BUILD_ROOT%\bullet3.lib
exit /b 0
