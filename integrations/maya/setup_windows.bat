@echo off
rem One-time setup for GEM-X Live on Windows.
rem   1. fetches and builds gem-x.cpp (MSVC + Vulkan)
rem   2. creates the capture server's Python environment
rem   3. downloads the GGUF models (~4 GB) and checks their SHA-256
rem   4. installs the Maya module
rem Needs: Git, Visual Studio 2022 C++ tools, CMake, Ninja, Vulkan SDK, Python 3.10+ (uv optional).
setlocal enabledelayedexpansion
set HERE=%~dp0
set REPO=%HERE%..\..
set GEMX=%REPO%\third_party\gem-x.cpp

echo == gem-x.cpp source
pushd "%REPO%"
git submodule update --init third_party/gem-x.cpp || goto :fail
popd
pushd "%GEMX%"
git submodule update --init --recursive || goto :fail
popd

echo == gem-x.cpp build
if exist "%GEMX%\build\win-vulkan\gemx.dll" (
  echo already built
) else (
  call "%GEMX%\scripts\build_windows.bat" vulkan || goto :fail
)

echo == capture server environment
pushd "%HERE%server"
if not exist ".venv\Scripts\python.exe" (
  where uv >nul 2>nul
  if !errorlevel!==0 (
    uv venv .venv --python 3.12 || goto :fail_pop
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail_pop
  ) else (
    py -3 -m venv .venv || goto :fail_pop
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail_pop
  )
)

echo == models
.venv\Scripts\python.exe fetch_models.py --dest "%GEMX%\generated\reference" || goto :fail_pop
popd

echo == Maya module
"%HERE%server\.venv\Scripts\python.exe" "%HERE%install.py" || goto :fail

echo.
echo Done. Start Maya, load gemxLive.py in the Plug-in Manager (or drag install.py into a viewport),
echo then open GEM-X ^> Live Capture.
exit /b 0

:fail_pop
popd
:fail
echo Setup failed.
exit /b 1
