cmake_minimum_required(VERSION 3.20)
project(regret LANGUAGES CXX VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings/module.cpp)
target_include_directories(_core PRIVATE cpp/include)
target_compile_definitions(_core PRIVATE REGRET_VERSION="${SKBUILD_PROJECT_VERSION}")

install(TARGETS _core LIBRARY DESTINATION regret)
