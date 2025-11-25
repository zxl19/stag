cmake_minimum_required(VERSION 3.10)
project(Stag)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

IF (WIN32)
  set(CMAKE_DEBUG_POSTFIX "d")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
ENDIF (WIN32)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -w  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 3 REQUIRED)

file(GLOB SRC_FILE1 "src/*.c*")
file(GLOB SRC_FILE2 "src/ED/*.c*")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    src/
    src/ED/
)
add_library(
    staglib
    ${SRC_FILE1}
    ${SRC_FILE2}
        src/Stag.cpp
        src/Stag.h
)
target_link_libraries(
    staglib
    ${OpenCV_LIBS}
)

add_executable(
    stag_example
    main.cpp
)

target_link_libraries(
    stag_example
    staglib
)

IF (MSVC)
  set_target_properties(stag_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set_target_properties(stag_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
  set_target_properties(stag_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
ENDIF (MSVC)
