@echo off
setlocal

call "D:\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -Gpu -Generator "NMake Makefiles" -Qt6Dir "D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6" -CudaRoot "D:\CUDA" -VsRoot "D:\Microsoft Visual Studio"
exit /b %errorlevel%
