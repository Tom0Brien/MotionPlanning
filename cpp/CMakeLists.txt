cmake_minimum_required(VERSION 3.12)

# Set the project name
project(motion_planning)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force coloured compiler output
add_compile_options(-fdiagnostics-color)

# CXX flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_PREFIX_PATH "$ENV{HOME}/.local/lib/python3.10/site-packages/cmeel.prefix/lib/cmake")

find_package(Catch2 2.13 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS filesystem log program_options)
find_package(NLopt REQUIRED)
#find_package(coal REQUIRED)


# PCL
find_package(PCL 1.7 REQUIRED COMPONENTS common io filters segmentation
                                         visualization)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_libraries(${PCL_LIBRARIES})

# OpenCV
find_package(OpenCV 4 REQUIRED)
link_libraries(${OpenCV_LIBS})

# libLas
find_package(libLAS REQUIRED)
link_libraries(${libLAS_LIBRARIES})

# yaml-cpp
find_package(yaml-cpp REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS} ${libLAS_INCLUDE_DIRS})

# Target names
set(TARGET_LIB common)
set(TARGET_BIN ${PROJECT_NAME})
set(TARGET_TEST tests)

# Libraries
set(LIBS Eigen3::Eigen yaml-cpp ${NLOPT_LIBRARIES})

# Source files

set(DIR_SRC src)
set(DIR_TEST_SRC test/src)

# Application
file(GLOB SRC_BIN CONFIGURE_DEPENDS ${DIR_SRC}/*.cpp ${DIR_SRC}/*.h
     ${DIR_SRC}/*.hpp)

# Static library
set(SRC_LIB ${SRC_BIN})
list(FILTER SRC_LIB EXCLUDE REGEX ".*main.cpp$")

# Unit tests
file(GLOB SRC_TEST CONFIGURE_DEPENDS ${DIR_TEST_SRC}/*.cpp ${DIR_TEST_SRC}/*.h
     ${DIR_TEST_SRC}/*.hpp)

# Targets

# Static library
if(SRC_LIB)
  add_library(${TARGET_LIB} ${SRC_LIB} INTERFACE)
endif()

# Application
add_executable(${TARGET_BIN} src/main.cpp)
if(SRC_LIB)
  target_link_libraries(${TARGET_BIN} ${TARGET_LIB})
endif()
target_link_libraries(${TARGET_BIN} ${LIBS})

# Unit tests add_executable(${TARGET_TEST} ${SRC_TEST}) if(SRC_LIB)
# target_link_libraries(${TARGET_TEST} ${TARGET_LIB}) endif()
# target_link_libraries(${TARGET_TEST} ${LIBS})
# target_link_libraries(${TARGET_TEST} Catch2::Catch2)

# # Run unit tests after building executables add_custom_target( run_tests ALL
# COMMAND ${TARGET_TEST} --use-colour yes DEPENDS ${TARGET_TEST} DEPENDS
# ${TARGET_BIN} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} COMMENT "Running tests")

# CTest integration

# include(CTest) include(Catch) catch_discover_tests(${TARGET_TEST})

set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)
