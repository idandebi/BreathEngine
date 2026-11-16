cmake_minimum_required(VERSION 3.22.1)
project(breathengine_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared C++ core lives at the repository root (BreathEngine/cpp).
set(BE_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp)

# OpenCV comes from the org.opencv:opencv Maven AAR via prefab.
find_package(OpenCV REQUIRED CONFIG)

add_library(breathengine_jni SHARED
    jni_bridge.cpp
    ${BE_CORE_DIR}/src/engine.cpp
    ${BE_CORE_DIR}/src/c_api.cpp)

target_include_directories(breathengine_jni PRIVATE ${BE_CORE_DIR}/include)
target_compile_definitions(breathengine_jni PRIVATE BE_VERSION_STRING="0.1.0")
target_compile_options(breathengine_jni PRIVATE -Wall -Wextra -fvisibility=hidden)
target_link_libraries(breathengine_jni PRIVATE OpenCV::opencv_java4 jnigraphics log)
target_link_options(breathengine_jni PRIVATE "-Wl,-z,max-page-size=16384")
