cmake_minimum_required(VERSION 3.14)

file(GLOB PUBLIC_HEADERS "*.h")

project(3DSystem VERSION 1.0 LANGUAGES C CXX)

# Set C++17 as the standard and require it
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable AUTOMOC, AUTOUIC, and AUTORCC for Qt
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find required packages
find_package(QT NAMES Qt6 COMPONENTS Core REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io filters visualization segmentation surface)
find_package(CGAL)

# Include directories for PCL and Qt
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# Define the library target
add_library(3DSystem SHARED
    3d_system_global.h
    pcl_3d.cpp
    pcl_3d.h
    ply_processor.cpp
    ply_processor.h
    ply_segmentation.cpp
    ply_segmentation.h
    clusters.h
)

# Specify libraries to link with target
target_link_libraries(3DSystem PRIVATE ${PCL_LIBRARIES} CGAL::CGAL)

# Define compilation definitions, if necessary
target_compile_definitions(3DSystem PRIVATE SYSTEM_3D_LIBRARY)

target_include_directories(3DSystem
    PUBLIC .
)

# Set target properties if needed (like version, SOVERSION, etc.)
set_target_properties(3DSystem PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
)

include(GNUInstallDirs)

# Install rules, if required
# Put library in /home/suleyman/Desktop/MasterThesis/library directory and headers in include directory
install(TARGETS 3DSystem
    LIBRARY DESTINATION /home/suleyman/Desktop/MasterThesis/library/lib
    PUBLIC_HEADER DESTINATION /home/suleyman/Desktop/MasterThesis/library/lib/include
)
# install(TARGETS 3DSystem
#     LIBRARY DESTINATION /Users/suleymanismaylov/Desktop/MasterThesis/Library/lib
#     PUBLIC_HEADER DESTINATION /Users/suleymanismaylov/Desktop/MasterThesis/Library/lib/include
# )
