cmake_minimum_required(VERSION 3.3)
project(PMenhence)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp PMenhence.h)

find_package(OpenCV REQUIRED)
add_executable(PMenhence ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})