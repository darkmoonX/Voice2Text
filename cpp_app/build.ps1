param(
    [switch]$Gpu = $true,
    [switch]$CpuOnly,
    [string]$Qt6Dir = "",
    [string]$CudaRoot = "D:\CUDA",
    [string]$VsRoot = "D:\Microsoft Visual Studio",
    [string]$Generator = "",
    [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-WinDeployQt {
    param([string]$ResolvedQt6Dir)
    if (-not $ResolvedQt6Dir) { return "" }
    $cmakeQt6Dir = [System.IO.DirectoryInfo]::new($ResolvedQt6Dir)
    $qtRoot = $cmakeQt6Dir.Parent.Parent.Parent.FullName
    $candidate = Join-Path $qtRoot "bin\\windeployqt.exe"
    if (Test-Path $candidate) { return $candidate }
    return ""
}

function Sync-NativeDepsToExeDir {
    param([string]$BuildDir)
    $exePathCandidates = @(
        (Join-Path $BuildDir "Release\\voice2text_cpp.exe"),
        (Join-Path $BuildDir "voice2text_cpp.exe")
    )
    $exePath = $exePathCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $exePath) { return }
    $exeDir = Split-Path -Path $exePath -Parent

    $dllDirs = @(
        (Join-Path $BuildDir "bin\\Release"),
        (Join-Path $BuildDir "bin")
    ) | Where-Object { Test-Path $_ }

    foreach ($dllDir in $dllDirs) {
        Get-ChildItem -Path $dllDir -Filter *.dll -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $exeDir $_.Name) -Force
        }
    }
}

function Deploy-QtRuntime {
    param([string]$BuildDir, [string]$ResolvedQt6Dir)
    $exePathCandidates = @(
        (Join-Path $BuildDir "Release\\voice2text_cpp.exe"),
        (Join-Path $BuildDir "voice2text_cpp.exe")
    )
    $exePath = $exePathCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not (Test-Path $exePath)) { return }
    $windeployqt = Resolve-WinDeployQt -ResolvedQt6Dir $ResolvedQt6Dir
    if (-not $windeployqt) {
        Write-Warning "windeployqt.exe not found. Runtime may require manual PATH setup."
        return
    }
    & $windeployqt --release --no-translations --no-system-d3d-compiler --no-opengl-sw $exePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "windeployqt failed. Runtime may require manual PATH setup."
    }
}

function Resolve-Qt6Dir {
    param([string]$PreferredQt6Dir)

    if ($PreferredQt6Dir) {
        $configFile = Join-Path $PreferredQt6Dir "Qt6Config.cmake"
        if (Test-Path $configFile) {
            return $PreferredQt6Dir
        }
    }

    if ($env:Qt6_DIR) {
        $configFile = Join-Path $env:Qt6_DIR "Qt6Config.cmake"
        if (Test-Path $configFile) {
            return $env:Qt6_DIR
        }
    }

    if (Test-Path "D:\Qt") {
        $candidates = Get-ChildItem -Path "D:\Qt" -Filter "Qt6Config.cmake" -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
        if ($candidates -and $candidates.Count -gt 0) {
            return $candidates[0].DirectoryName
        }
    }

    return $null
}

function Resolve-VsInstance {
    param([string]$VisualStudioRoot, [string]$VisualStudioGenerator)

    if ($VisualStudioGenerator -notlike "Visual Studio*") {
        return ""
    }

    $knownCandidates = @(
        "18\Community",
        "18\Professional",
        "18\Enterprise",
        "18\BuildTools",
        "2022\Community",
        "2022\Professional",
        "2022\Enterprise",
        "2022\BuildTools"
    )

    foreach ($suffix in $knownCandidates) {
        $candidate = Join-Path $VisualStudioRoot $suffix
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return ""
}

function Resolve-Generator {
    param([string]$PreferredGenerator, [string]$VisualStudioRoot)

    if ($PreferredGenerator) {
        return $PreferredGenerator
    }

    if (Test-Path (Join-Path $VisualStudioRoot "18")) {
        return "Visual Studio 18 2026"
    }

    if (Test-Path (Join-Path $VisualStudioRoot "2022")) {
        return "Visual Studio 17 2022"
    }

    return "Ninja Multi-Config"
}

function Resolve-VsDevCmd {
    param([string]$VisualStudioRoot)

    $candidates = @(
        "18\Community\Common7\Tools\VsDevCmd.bat",
        "18\Professional\Common7\Tools\VsDevCmd.bat",
        "18\Enterprise\Common7\Tools\VsDevCmd.bat",
        "18\BuildTools\Common7\Tools\VsDevCmd.bat",
        "2022\Community\Common7\Tools\VsDevCmd.bat",
        "2022\Professional\Common7\Tools\VsDevCmd.bat",
        "2022\Enterprise\Common7\Tools\VsDevCmd.bat",
        "2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    )

    foreach ($suffix in $candidates) {
        $candidate = Join-Path $VisualStudioRoot $suffix
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return ""
}

function Resolve-VcVars64 {
    param([string]$VisualStudioRoot)

    $candidates = @(
        "18\Community\VC\Auxiliary\Build\vcvars64.bat",
        "18\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "18\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($suffix in $candidates) {
        $candidate = Join-Path $VisualStudioRoot $suffix
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return ""
}

function Resolve-Qt6MingwDir {
    param([string]$CurrentQt6Dir)

    if ($CurrentQt6Dir -and $CurrentQt6Dir -match "mingw") {
        return $CurrentQt6Dir
    }

    $preferred = "D:\Qt\6.11.0\mingw_64\lib\cmake\Qt6"
    if (Test-Path (Join-Path $preferred "Qt6Config.cmake")) {
        return $preferred
    }

    if (Test-Path "D:\Qt") {
        $candidates = Get-ChildItem -Path "D:\Qt" -Filter "Qt6Config.cmake" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "mingw_64\\lib\\cmake\\Qt6\\Qt6Config\.cmake$" } |
            Sort-Object FullName -Descending
        if ($candidates -and $candidates.Count -gt 0) {
            return $candidates[0].DirectoryName
        }
    }

    return ""
}

$resolvedQt6Dir = Resolve-Qt6Dir -PreferredQt6Dir $Qt6Dir
if (-not $resolvedQt6Dir) {
    Write-Error "Qt6_DIR not found. Provide -Qt6Dir or install Qt under D:\Qt."
    exit 1
}
$env:Qt6_DIR = $resolvedQt6Dir

if (Test-Path $CudaRoot) {
    $env:CUDAToolkit_ROOT = $CudaRoot

    $cudaBin = Join-Path $CudaRoot "bin"
    $cudaBinX64 = Join-Path $cudaBin "x64"

    foreach ($pathEntry in @($cudaBin, $cudaBinX64)) {
        if ((Test-Path $pathEntry) -and ($env:Path -notlike "*$pathEntry*")) {
            $env:Path = "$pathEntry;$env:Path"
        }
    }
}

if ($CpuOnly) {
    $Gpu = $false
}

if ($Gpu -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Warning "nvcc not found in PATH. Falling back to CPU build."
    $Gpu = $false
}

$buildDir = if ($Gpu) { "build-vs-gpu" } else { "build-vs-cpu" }
$cudaFlag = if ($Gpu) { "ON" } else { "OFF" }
$effectiveGenerator = Resolve-Generator -PreferredGenerator $Generator -VisualStudioRoot $VsRoot

if (Test-Path (Join-Path $buildDir "CMakeCache.txt")) {
    Remove-Item -Path $buildDir -Recurse -Force
}

$configureArgs = @(
    "-S", ".",
    "-B", $buildDir,
    "-G", $effectiveGenerator,
    "-DENABLE_WHISPER_CUDA=$cudaFlag",
    "-DWHISPER_CUDA_ARCHITECTURES=86",
    "-DQt6_DIR=$resolvedQt6Dir"
)

if ($effectiveGenerator -like "Visual Studio*") {
    $configureArgs += @("-A", $Arch)
}

if (Test-Path $CudaRoot) {
    $configureArgs += "-DCUDAToolkit_ROOT=$CudaRoot"
}

$generatorInstance = Resolve-VsInstance -VisualStudioRoot $VsRoot -VisualStudioGenerator $effectiveGenerator
if ($generatorInstance) {
    $configureArgs += "-DCMAKE_GENERATOR_INSTANCE=$generatorInstance"
}

$commonCMakeDefs = @(
    "-DENABLE_WHISPER_CUDA=$cudaFlag",
    "-DWHISPER_CUDA_ARCHITECTURES=86",
    "-DQt6_DIR=$resolvedQt6Dir"
)
if (Test-Path $CudaRoot) {
    $commonCMakeDefs += "-DCUDAToolkit_ROOT=$CudaRoot"
}

Write-Host "Configuring with Qt6_DIR=$resolvedQt6Dir"
Write-Host "Using CMake generator: $effectiveGenerator"
if (Test-Path $CudaRoot) {
    Write-Host "Using CUDAToolkit_ROOT=$CudaRoot"
}
if ($generatorInstance) {
    Write-Host "Using CMAKE_GENERATOR_INSTANCE=$generatorInstance"
}
Write-Host "CUDA enabled: $Gpu"

cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    $vsFallbackTried = $false

    if ($effectiveGenerator -like "Visual Studio*") {
        $vcVars64 = Resolve-VcVars64 -VisualStudioRoot $VsRoot
        $vsDevCmd = Resolve-VsDevCmd -VisualStudioRoot $VsRoot
        $fallbackSetupCmd = if ($vcVars64) { $vcVars64 } else { $vsDevCmd }
        if ($fallbackSetupCmd) {
            $vsFallbackTried = $true
            $fallbackGenerator = "NMake Makefiles"
            $fallbackPrefixCmd = ""
            $fallbackDefs = @()
            $fallbackBuildDir = "$buildDir-nmake"
            $fallbackDefs += "-DCMAKE_BUILD_TYPE=Release"

            $jomExe = "D:\Qt\Tools\QtCreator\bin\jom\jom.exe"
            if (Test-Path $jomExe) {
                $fallbackGenerator = "NMake Makefiles JOM"
                $fallbackDefs += "-DCMAKE_MAKE_PROGRAM=$jomExe"
            }

            Write-Warning "Visual Studio generator configure failed. Trying VsDevCmd + $fallbackGenerator fallback."

            if (Test-Path $fallbackBuildDir) {
                try {
                    Remove-Item -Path $fallbackBuildDir -Recurse -Force
                } catch {
                    $fallbackBuildDir = "$buildDir-nmake-$([DateTime]::Now.ToString('yyyyMMddHHmmss'))"
                    Write-Warning "Could not clean fallback build directory. Using $fallbackBuildDir instead."
                }
            }

            $defsForCmd = (($commonCMakeDefs + $fallbackDefs) | ForEach-Object { "`"$_`"" }) -join " "
            $configureCmd = "cmake -S . -B `"$fallbackBuildDir`" -G `"$fallbackGenerator`" $defsForCmd"
            $buildCmd = "cmake --build `"$fallbackBuildDir`" --config Release"
            $setupArgs = if ($fallbackSetupCmd -eq $vsDevCmd -and $Arch) { " -arch=$Arch" } else { "" }
            $fullCmd = "call `"$fallbackSetupCmd`"$setupArgs && $fallbackPrefixCmd$configureCmd && $buildCmd"

            cmd /c $fullCmd
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "MSVC fallback failed. cl.exe or Windows SDK toolset may be missing from installed workloads."
            } else {
                Sync-NativeDepsToExeDir -BuildDir $fallbackBuildDir
                Deploy-QtRuntime -BuildDir $fallbackBuildDir -ResolvedQt6Dir $resolvedQt6Dir
                Write-Host "Build completed via MSVC fallback. Output directory: $fallbackBuildDir"
                exit 0
            }
        }
    }

    $mingwGpp = "D:\Qt\Tools\mingw1310_64\bin\g++.exe"
    $mingwQt6Dir = Resolve-Qt6MingwDir -CurrentQt6Dir $resolvedQt6Dir
    if ((Test-Path $mingwGpp) -and $mingwQt6Dir) {
        Write-Warning "Trying MinGW CPU fallback build (no CUDA)."

        $mingwBuildDir = "build-mingw-cpu"
        if (Test-Path $mingwBuildDir) {
            Remove-Item -Path $mingwBuildDir -Recurse -Force
        }

        $oldPath = $env:Path
        $oldQt6Dir = $env:Qt6_DIR
        $env:Path = "D:\Qt\Tools\mingw1310_64\bin;$env:Path"
        $env:Qt6_DIR = $mingwQt6Dir

        cmake -S . -B $mingwBuildDir -G "MinGW Makefiles" -DQt6_DIR=$mingwQt6Dir -DCMAKE_PREFIX_PATH=$mingwQt6Dir -DENABLE_WHISPER_CUDA=OFF
        if ($LASTEXITCODE -eq 0) {
            cmake --build $mingwBuildDir --config Release
        }

        $env:Path = $oldPath
        $env:Qt6_DIR = $oldQt6Dir

        if ($LASTEXITCODE -eq 0) {
            Sync-NativeDepsToExeDir -BuildDir $mingwBuildDir
            Write-Host "Build completed via MinGW fallback. Output directory: $mingwBuildDir"
            exit 0
        }
    }

    if ($vsFallbackTried) {
        throw "CMake configure/build failed. VS toolchain unavailable and MinGW fallback also failed."
    }

    throw "CMake configure failed."
}

cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed."
}

Sync-NativeDepsToExeDir -BuildDir $buildDir
Deploy-QtRuntime -BuildDir $buildDir -ResolvedQt6Dir $resolvedQt6Dir

Write-Host "Build completed. Output directory: $buildDir"
