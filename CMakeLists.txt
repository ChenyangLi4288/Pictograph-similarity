cmake_minimum_required(VERSION 3.10)
project(ImageSimilarity)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(image_similarity image_similarity.cpp)

# Link OpenCV libraries
target_link_libraries(image_similarity ${OpenCV_LIBS})

# Set output directory
set_target_properties(image_similarity PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ==============================================================================
# Testing Configuration
# ==============================================================================

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Google Test found - building tests")
    
    # Include Google Test directories
    include_directories(${GTEST_INCLUDE_DIRS})
    
    # Add test executable
    add_executable(image_similarity_test image_similarity_test.cpp)
    
    # Link test libraries
    target_link_libraries(image_similarity_test 
        ${OpenCV_LIBS}
        GTest::GTest
        GTest::Main
    )
    
    # Add test to CTest
    add_test(NAME ImageSimilarityTests COMMAND image_similarity_test)
    
    # Set output directory for test executable
    set_target_properties(image_similarity_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    
    message(STATUS "Test executable will be created at: ${CMAKE_BINARY_DIR}/tests/image_similarity_test")
else()
    message(WARNING "Google Test not found - tests will not be built")
    message(WARNING "To install Google Test:")
    message(WARNING "  Ubuntu/Debian: sudo apt install libgtest-dev")
    message(WARNING "  macOS: brew install googletest")
    message(WARNING "  Windows (MSYS2): pacman -S mingw-w64-x86_64-gtest")
endif()