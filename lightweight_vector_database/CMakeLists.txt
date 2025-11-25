# CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(HNSW_VECTOR_DB)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1 # Use a specific tag for stability
)
FetchContent_MakeAvailable(pybind11)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the core vector database library
add_library(vector_database INTERFACE)
target_include_directories(vector_database INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add Python bindings
pybind11_add_module(vector_database_bindings src/bindings.cpp)
target_link_libraries(vector_database_bindings PUBLIC vector_database)
target_include_directories(vector_database_bindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Define the main test executable
add_executable(hnsw_test test/test.cpp)
target_link_libraries(hnsw_test PUBLIC vector_database)

# Define the deletion test executable
add_executable(delete_test test/test_delete.cpp)
target_link_libraries(delete_test PUBLIC vector_database)

# Define the update test executable
add_executable(update_test test/test_update.cpp)
target_link_libraries(update_test PUBLIC vector_database)

# Define the SQ test executable
add_executable(sq_test test/test_sq.cpp)
target_link_libraries(sq_test PUBLIC vector_database)

# Enable testing with CTest
enable_testing()

# Add test cases that run the test executables
add_test(NAME RunHNSWTests COMMAND hnsw_test)
add_test(NAME RunHNSWDeleteTests COMMAND delete_test)
add_test(NAME RunHNSWUpdateTests COMMAND update_test)
add_test(NAME RunHNSWSQTests COMMAND sq_test)

# Add the showcase applications
add_subdirectory(showcase_apps)
