cmake_minimum_required(VERSION 3.22)
project(voice2text_capture_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)

if(WIN32 AND MINGW)
    add_compile_definitions(_WIN32_WINNT=0x0601 WINVER=0x0601)
endif()

find_package(Qt6 REQUIRED COMPONENTS Core)

add_executable(
    voice2text_capture_bridge
    src/main.cpp
    src/discovery.cpp
    src/loopback_capture.cpp
    src/discovery.h
    src/loopback_capture.h
)

target_include_directories(voice2text_capture_bridge PRIVATE src)
target_link_libraries(voice2text_capture_bridge PRIVATE Qt6::Core)

if(WIN32)
    target_link_libraries(voice2text_capture_bridge PRIVATE ole32 avrt runtimeobject uuid propsys mmdevapi)
endif()
