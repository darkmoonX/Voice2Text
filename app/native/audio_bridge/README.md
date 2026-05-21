# Audio Capture Bridge

This folder contains the in-repo C++ bridge used by the Python runtime for
audio capture in:

- `loopback` mode (WASAPI loopback)
- `app` mode (Application Loopback Capture with process targeting)

Current bridge stream behavior:
- Output sample rate follows device mix format.
- Output channel count follows device mix format (no forced mono downmix).
- PCM payload format is interleaved `int16` (little-endian).
- MSVC build now reports process-loopback capability correctly via `--probe-process-loopback`.

## Build

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1
```

Load MSVC environment in PowerShell (Thomas local VS path):

```powershell
cd app
.\scripts\enter_msvc_env.ps1
cl
```

Machine preset (Thomas local env):

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1 `
  -Generator "MinGW Makefiles" `
  -MakeProgram "D:\MinGW\bin\mingw32-make.exe" `
  -CCompiler "D:\MinGW\bin\gcc.exe" `
  -CxxCompiler "D:\MinGW\bin\g++.exe"
```

MSVC preset (Visual Studio generator):

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1 `
  -Generator "Visual Studio 17 2022" `
  -Qt6Dir "D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6"
```

Optional: load from a custom VS install root

```powershell
cd app
.\scripts\enter_msvc_env.ps1 -VsRoot "D:\Microsoft Visual Studio\18\Community" -Arch x64 -HostArch x64
```

MSVC preset (Ninja + MSVC toolchain):

```powershell
# Run inside "x64 Native Tools Command Prompt for VS 2022"
cd app\native\audio_bridge
.\build_bridge.ps1 `
  -Generator "Ninja Multi-Config" `
  -Qt6Dir "D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6" `
  -CCompiler "cl.exe" `
  -CxxCompiler "cl.exe"
```

Dependency-only deploy (no rebuild):

```powershell
.\build_bridge.ps1 -DeployOnly `
  -Generator "MinGW Makefiles" `
  -MakeProgram "D:\MinGW\bin\mingw32-make.exe" `
  -CCompiler "D:\MinGW\bin\gcc.exe" `
  -CxxCompiler "D:\MinGW\bin\g++.exe"
```

## Script Parameters (`build_bridge.ps1`)

- `-Qt6Dir`
  - Path to Qt6 CMake config folder (must contain `Qt6Config.cmake`).
  - Example: `D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6`.
- `-Generator`
  - CMake generator name.
  - Common values:
    - `MinGW Makefiles`
    - `Visual Studio 17 2022`
    - `Ninja Multi-Config`
- `-MakeProgram`
  - Build tool executable path (mainly for MinGW).
  - Example: `D:\MinGW\bin\mingw32-make.exe`.
- `-CCompiler`
  - C compiler path/name passed to CMake (`CMAKE_C_COMPILER`).
  - MinGW example: `D:\MinGW\bin\gcc.exe`.
  - MSVC example: `cl.exe` (use VS Native Tools environment).
- `-CxxCompiler`
  - C++ compiler path/name passed to CMake (`CMAKE_CXX_COMPILER`).
  - MinGW example: `D:\MinGW\bin\g++.exe`.
  - MSVC example: `cl.exe` (use VS Native Tools environment).
- `-BuildDir`
  - Build directory name/path (default: `build`).
- `-DeployOnly`
  - Skip rebuild, only deploy runtime dependencies to `app/src/runtime_bin`.
  - Useful after compiler/Qt runtime changes.

Expected output:

- `app/src/runtime_bin/voice2text_capture_bridge.exe`

## Runtime Integration

Python side resolver:

- `app/src/app/capture/cpp_backend.py`

Resolution priority:

1. `VOICE2TEXT_CPP_CAPTURE_BRIDGE` (env override)
2. `app/src/runtime_bin/voice2text_capture_bridge.exe`
3. local `app/native/audio_bridge/build/...` outputs

Quick capability check:

```powershell
app\src\runtime_bin\voice2text_capture_bridge.exe --probe-process-loopback
```

If app mode logs `ActivateAudioInterfaceAsync failed (hr=0x8000000e)`, rebuild bridge with latest sources; current implementation includes `IAgileObject` on completion handler and STA COM init for app-mode activation.
App-process-loopback now first queries the active render endpoint mix format (for example `48000 Hz / 2 ch / 32-bit`) and uses that for initialization.
If endpoint mix-format query is unavailable, bridge falls back to PCM `44100 Hz / 2 ch / 16-bit` for process-loopback initialization.
Current app-process-loopback initialization also uses loopback-compatible stream flags and zero-duration shared buffer parameters (Microsoft sample style) to avoid `IAudioClient initialize failed` startup exits.
