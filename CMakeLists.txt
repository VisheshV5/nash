cmake_minimum_required(VERSION 3.20)

if(DEFINED SKBUILD_PROJECT_VERSION)
  set(REGRET_VERSION ${SKBUILD_PROJECT_VERSION})
else()
  set(REGRET_VERSION 0.0.0)
endif()
project(regret LANGUAGES CXX VERSION ${REGRET_VERSION})

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

# Native core, shared by the Python extension and the C++ tests.
add_library(regret_native STATIC
  cpp/src/cards/evaluator.cpp
  cpp/src/cards/isomorphism.cpp
  cpp/src/engine/engine.cpp
  cpp/src/cfr/infoset_store.cpp
  cpp/src/cfr/solver.cpp
  cpp/src/abstraction/card_features.cpp
  cpp/src/abstraction/builders.cpp
  cpp/src/abstraction/actions.cpp
  cpp/src/games/nlhe.cpp
)
target_include_directories(regret_native PUBLIC cpp/include)
set_target_properties(regret_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(regret_native PUBLIC Threads::Threads)

if(DEFINED SKBUILD_PROJECT_NAME)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core cpp/bindings/module.cpp)
  target_link_libraries(_core PRIVATE regret_native)
  target_compile_definitions(_core PRIVATE REGRET_VERSION="${REGRET_VERSION}")
  install(TARGETS _core LIBRARY DESTINATION regret)
endif()

# Exhaustive C++ checks (slow ones behind --slow). Build with -DREGRET_BUILD_TESTS=ON.
option(REGRET_BUILD_TESTS "Build the C++ test executable" OFF)
if(REGRET_BUILD_TESTS)
  enable_testing()
  add_executable(regret_tests cpp/tests/test_main.cpp)
  target_link_libraries(regret_tests PRIVATE regret_native)
  add_test(NAME regret_tests COMMAND regret_tests)
endif()
