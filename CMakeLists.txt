cmake_minimum_required(VERSION 3.14)

project(3d_system LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

# Define a variable for 3D_SYSTEM_LIBRARY
set(3D_SYSTEM_LIBRARY TRUE)

add_library(3d_system SHARED
  3d_system_global.h
  pcl_3d.cpp
  pcl_3d.h
)

target_link_libraries(3d_system PRIVATE Qt${QT_VERSION_MAJOR}::Core)

# Pass the variable as a definition to the compiler
target_compile_definitions(3d_system PRIVATE SYSTEM_3D_LIBRAR)
