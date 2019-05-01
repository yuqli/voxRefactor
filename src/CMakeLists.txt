cmake_minimum_required(VERSION 3.9)
project(refactor LANGUAGES CUDA CXX)
project(refactor)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})

add_executable(refactor main.cu)
target_link_libraries( refactor ${Boost_LIBRARIES})
