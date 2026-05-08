cmake_minimum_required(VERSION 3.22)
project(voice2text_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(WIN32 AND MINGW)
    # MinGW headers may miss Windows 8+ thread power throttling symbols used by ggml.
    add_compile_definitions(_WIN32_WINNT=0x0601 WINVER=0x0601)
endif()

option(ENABLE_WHISPER_CUDA "Enable CUDA/cuBLAS acceleration in whisper.cpp" ON)
set(WHISPER_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures for whisper.cpp (RTX 3060=86)")
set(
    WHISPER_CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp"
    CACHE PATH
    "Path to whisper.cpp repository"
)

if(NOT DEFINED Qt6_DIR AND DEFINED ENV{Qt6_DIR})
    set(Qt6_DIR "$ENV{Qt6_DIR}" CACHE PATH "Qt6Config.cmake directory from environment" FORCE)
endif()

if(NOT DEFINED Qt6_DIR AND EXISTS "D:/Qt")
    file(
        GLOB_RECURSE QT6_CONFIG_CANDIDATES
        LIST_DIRECTORIES FALSE
        "D:/Qt/*/msvc*/lib/cmake/Qt6/Qt6Config.cmake"
    )
    if(QT6_CONFIG_CANDIDATES)
        list(SORT QT6_CONFIG_CANDIDATES)
        list(REVERSE QT6_CONFIG_CANDIDATES)
        list(GET QT6_CONFIG_CANDIDATES 0 QT6_CONFIG_PATH)
        get_filename_component(QT6_CONFIG_DIR "${QT6_CONFIG_PATH}" DIRECTORY)
        set(Qt6_DIR "${QT6_CONFIG_DIR}" CACHE PATH "Auto-detected Qt6Config.cmake directory" FORCE)
        message(STATUS "Auto-detected Qt6_DIR=${Qt6_DIR}")
    endif()
endif()

if(NOT DEFINED CUDAToolkit_ROOT)
    if(DEFINED ENV{CUDAToolkit_ROOT})
        set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}" CACHE PATH "CUDA root from environment" FORCE)
    elseif(DEFINED ENV{CUDA_PATH})
        set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}" CACHE PATH "CUDA root from CUDA_PATH" FORCE)
    elseif(EXISTS "D:/CUDA")
        set(CUDAToolkit_ROOT "D:/CUDA" CACHE PATH "CUDA root auto-detected from D:/CUDA" FORCE)
    endif()
endif()

if(ENABLE_WHISPER_CUDA)
    find_program(
        NVCC_EXECUTABLE
        nvcc
        HINTS
            "${CUDAToolkit_ROOT}/bin"
            "$ENV{CUDA_PATH}/bin"
            "D:/CUDA/bin"
    )
    if(NOT NVCC_EXECUTABLE)
        message(WARNING "nvcc not found in PATH. CUDA acceleration is disabled for this configure run.")
        set(ENABLE_WHISPER_CUDA OFF CACHE BOOL "" FORCE)
    endif()
endif()

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

if(EXISTS "${WHISPER_CPP_DIR}/CMakeLists.txt")
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

    if(ENABLE_WHISPER_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES ${WHISPER_CUDA_ARCHITECTURES} CACHE STRING "" FORCE)
        endif()

        if(MSVC)
            # Reduce CUDA+MSVC warning flood (code page and known nvcc diagnostics)
            # to keep long NMake builds stable and readable.
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819 --diag-suppress=177 --diag-suppress=221")
        endif()
    else()
        # Prevent stale cache values from previous GPU builds.
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    endif()

    add_subdirectory("${WHISPER_CPP_DIR}" "${CMAKE_BINARY_DIR}/whisper_cpp")

    if(ENABLE_WHISPER_CUDA AND TARGET ggml-cuda)
        target_compile_options(ggml-cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
            $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=221>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
        )
    endif()

    set(HAS_WHISPER_CPP TRUE)
else()
    message(WARNING "whisper.cpp not found at ${WHISPER_CPP_DIR}; building without STT backend.")
    set(HAS_WHISPER_CPP FALSE)
endif()

message(STATUS "ENABLE_WHISPER_CUDA=${ENABLE_WHISPER_CUDA}")
if(ENABLE_WHISPER_CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(SRC_FILES
    src/main.cpp
    src/app_controller.cpp
    src/runtime_logger.cpp
    src/settings_dialog.cpp
    src/subtitle_overlay_window.cpp
    src/tray_controller.cpp
    src/whisper_config.cpp
    src/audio/loopback_capture.cpp
    src/stt/whisper_engine.cpp
    src/translation/translation_engine.cpp
)

set(HDR_FILES
    src/app_controller.h
    src/runtime_logger.h
    src/settings_dialog.h
    src/subtitle_overlay_window.h
    src/tray_controller.h
    src/whisper_config.h
    src/audio/loopback_capture.h
    src/stt/whisper_engine.h
    src/translation/translation_engine.h
)

add_executable(voice2text_cpp ${SRC_FILES} ${HDR_FILES})
target_include_directories(voice2text_cpp PRIVATE src)

target_link_libraries(voice2text_cpp PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets)

if(WIN32 AND ENABLE_WHISPER_CUDA)
    set(CUDA_RUNTIME_DIR_CANDIDATES
        "${CUDAToolkit_ROOT}/bin/x64"
        "${CUDAToolkit_ROOT}/bin"
        "$ENV{CUDA_PATH}/bin/x64"
        "$ENV{CUDA_PATH}/bin"
        "D:/CUDA/bin/x64"
        "D:/CUDA/bin"
    )

    set(CUDA_RUNTIME_DIR "")
    foreach(_cuda_dir IN LISTS CUDA_RUNTIME_DIR_CANDIDATES)
        if(EXISTS "${_cuda_dir}/cublas64_13.dll")
            set(CUDA_RUNTIME_DIR "${_cuda_dir}")
            break()
        endif()
    endforeach()

    if(CUDA_RUNTIME_DIR)
        set(CUDA_RUNTIME_DLLS
            cublas64_13.dll
            cublasLt64_13.dll
            cudart64_13.dll
        )

        add_custom_command(TARGET voice2text_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:voice2text_cpp>/bin"
            COMMENT "Preparing CUDA runtime directory"
        )

        foreach(_cuda_dll IN LISTS CUDA_RUNTIME_DLLS)
            if(EXISTS "${CUDA_RUNTIME_DIR}/${_cuda_dll}")
                add_custom_command(TARGET voice2text_cpp POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CUDA_RUNTIME_DIR}/${_cuda_dll}"
                        "$<TARGET_FILE_DIR:voice2text_cpp>/bin/${_cuda_dll}"
                    COMMENT "Copying ${_cuda_dll} from ${CUDA_RUNTIME_DIR}"
                )
            else()
                message(WARNING "CUDA runtime DLL not found: ${CUDA_RUNTIME_DIR}/${_cuda_dll}")
            endif()
        endforeach()

        message(STATUS "CUDA runtime DLL source: ${CUDA_RUNTIME_DIR}")
    else()
        message(WARNING "cublas64_13.dll not found in known CUDA runtime directories; GPU runtime may fail unless PATH includes CUDA DLL location.")
    endif()
endif()

if(HAS_WHISPER_CPP)
    if(TARGET whisper)
        target_link_libraries(voice2text_cpp PRIVATE whisper)
    elseif(TARGET whisper_cpp)
        target_link_libraries(voice2text_cpp PRIVATE whisper_cpp)
    endif()
    target_compile_definitions(voice2text_cpp PRIVATE HAS_WHISPER=1)
else()
    target_compile_definitions(voice2text_cpp PRIVATE HAS_WHISPER=0)
endif()

if(WIN32)
    target_link_libraries(voice2text_cpp PRIVATE ole32 avrt runtimeobject)
endif()