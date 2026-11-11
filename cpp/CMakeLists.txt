cmake_minimum_required(VERSION 3.18)
project(BreathEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BE_BUILD_SHARED "Build breath_engine as a shared library" OFF)
option(BE_BUILD_TESTS "Build unit tests and the parity tool" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

if(BE_BUILD_SHARED)
  add_library(breath_engine SHARED src/engine.cpp src/c_api.cpp)
  target_compile_definitions(breath_engine PRIVATE BE_BUILD_SHARED)
else()
  add_library(breath_engine STATIC src/engine.cpp src/c_api.cpp)
endif()
target_include_directories(breath_engine PUBLIC include)
target_link_libraries(breath_engine PUBLIC ${OpenCV_LIBS})
target_compile_definitions(breath_engine PRIVATE BE_VERSION_STRING="${PROJECT_VERSION}")
set_target_properties(breath_engine PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(MSVC)
  target_compile_options(breath_engine PRIVATE /W4 /permissive-)
else()
  target_compile_options(breath_engine PRIVATE -Wall -Wextra)
endif()

if(BE_BUILD_TESTS)
  enable_testing()
  add_executable(be_tests tests/test_core.cpp)
  target_link_libraries(be_tests PRIVATE breath_engine)
  add_test(NAME be_tests COMMAND be_tests)

  add_executable(be_parity tools/parity.cpp)
  target_link_libraries(be_parity PRIVATE breath_engine)

  if(WIN32)
    # Copy the OpenCV runtime next to the executables.
    file(GLOB _ocv_dlls "${OpenCV_DIR}/x64/*/bin/opencv_world*.dll"
                        "${OpenCV_DIR}/../bin/opencv_world*.dll" "${OpenCV_DIR}/bin/opencv_world*.dll")
    foreach(_t be_tests be_parity)
      foreach(_dll ${_ocv_dlls})
        add_custom_command(TARGET ${_t} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" $<TARGET_FILE_DIR:${_t}>)
      endforeach()
    endforeach()
  endif()
endif()
