# Audio Capture Bridge

In-repo C++ capture bridge used by Python runtime for:

- `loopback` mode (WASAPI loopback)
- `app` mode (Application Loopback Capture with process targeting)

This document is bridge-specific (build parameters and runtime behavior).
For full operator workflow, use [docs/build-and-run.md](/D:/project/Voice2Text/docs/build-and-run.md).

## Stream Behavior

- Output sample rate: follows active device/endpoint format.
- Output channels: follows active device/endpoint format (no forced mono).
- PCM payload: interleaved `int16` little-endian.

## Build

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1
```

Expected output:
- `app/src/runtime_bin/voice2text_capture_bridge.exe`

## Common Presets

### MinGW (Thomas local preset)

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1 `
  -Generator "MinGW Makefiles" `
  -MakeProgram "D:\MinGW\bin\mingw32-make.exe" `
  -CCompiler "D:\MinGW\bin\gcc.exe" `
  -CxxCompiler "D:\MinGW\bin\g++.exe"
```

### MSVC (Visual Studio generator)

```powershell
cd app\native\audio_bridge
.\build_bridge.ps1 `
  -Generator "Visual Studio 17 2022" `
  -Qt6Dir "D:\Qt\6.11.0\msvc2022_64\lib\cmake\Qt6"
```

### Deploy runtime dependencies only

```powershell
.\build_bridge.ps1 -DeployOnly `
  -Generator "MinGW Makefiles" `
  -MakeProgram "D:\MinGW\bin\mingw32-make.exe" `
  -CCompiler "D:\MinGW\bin\gcc.exe" `
  -CxxCompiler "D:\MinGW\bin\g++.exe"
```

## `build_bridge.ps1` Parameters

- `-Qt6Dir`: Qt6 CMake config folder (`Qt6Config.cmake`)
- `-Generator`: CMake generator
- `-MakeProgram`: explicit make program path (mainly MinGW)
- `-CCompiler`: C compiler path/name
- `-CxxCompiler`: C++ compiler path/name
- `-BuildDir`: custom build directory
- `-DeployOnly`: only deploy runtime DLL dependencies to runtime bin

## Runtime Integration

Python side resolver:
- `app/src/voice2text/capture/cpp_backend.py`

Resolution priority:
1. `VOICE2TEXT_CPP_CAPTURE_BRIDGE` env override
2. `app/src/runtime_bin/voice2text_capture_bridge.exe`
3. local build outputs under `app/native/audio_bridge/build*`

Capability probe:

```powershell
app\src\runtime_bin\voice2text_capture_bridge.exe --probe-process-loopback
```

## Notes

- If process-loopback endpoint mix format query fails, bridge falls back to PCM `44100/2ch/16-bit` for initialization safety.
- If Python sees bridge startup failure or unavailable capability, runtime can fall back to Python capture backend.
