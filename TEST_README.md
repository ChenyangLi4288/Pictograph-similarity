# Image Similarity Analyzer - Test Suite

This directory contains comprehensive unit tests for the Image Similarity Analyzer project.

## Test Coverage

The test suite covers:

### 1. **ImageData Structure Tests**
- Initialization and assignment
- Copy and move semantics
- Memory management

### 2. **Histogram Computation Tests**
- Valid image processing
- Identical images producing same histograms
- Different colors producing different histograms
- Edge cases (black/white images, gradients)
- Normalization verification

### 3. **Image Preprocessing Tests**
- Image resizing to standard dimensions (128x128)
- Edge detection with Canny algorithm
- Binary threshold application
- Handling various image sizes and types

### 4. **Similarity Calculation Tests**
- Identical images (high similarity expected)
- Entirely different images (low similarity)
- Similar shapes (moderate-high similarity)
- Different shapes (lower similarity)
- Range validation [0, 1]
- Symmetry property (sim(A,B) = sim(B,A))
- Transitive property testing

### 5. **CSV Export Tests**
- File creation and writing
- Proper number formatting (4 decimal places)
- CSV structure validation

### 6. **Edge Case Tests**
- Empty directories
- Single image
- Small images (1x1)
- Large images (4000x3000)
- Non-square images
- Extreme brightness (all black/all white)

### 7. **File System Tests**
- Multiple image formats (PNG, JPG, BMP)
- Recursive directory traversal
- Case-insensitive extension handling

### 8. **Performance & Stress Tests**
- Multiple images processing
- Large similarity matrix computation (20x20)
- Matrix property verification (symmetry, diagonal, range)

### 9. **Integration Tests**
- Full pipeline from image loading to similarity computation
- End-to-end workflow validation

### 10. **Statistical Tests**
- Average similarity calculations
- Min/max similarity bounds
- Distribution analysis

## Building and Running Tests

### Prerequisites

Install Google Test:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libgtest-dev cmake
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

**macOS:**
```bash
brew install googletest
```

**Windows (MSYS2):**
```bash
pacman -S mingw-w64-x86_64-gtest
```

### Build Tests

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build all targets (including tests if GTest is found)
cmake --build .

# Or build only tests
cmake --build . --target image_similarity_test
```

### Run Tests

```bash
# From build directory
./tests/image_similarity_test

# Run with verbose output
./tests/image_similarity_test --gtest_verbose

# Run specific test
./tests/image_similarity_test --gtest_filter=ImageSimilarityTest.ComputeHistogram_ValidImage

# Run tests matching a pattern
./tests/image_similarity_test --gtest_filter=*Histogram*

# List all tests
./tests/image_similarity_test --gtest_list_tests
```

### Using CTest

```bash
# From build directory
ctest

# Verbose output
ctest -V

# Run specific test
ctest -R ImageSimilarity
```

## Test Structure

Each test follows this pattern:

1. **Setup**: Create test fixtures and helper data  
2. **Execute**: Run the function/method being tested  
3. **Assert**: Verify expected outcomes using Google Test assertions  
4. **Teardown**: Clean up temporary files/resources  

## Test Helpers

The test suite includes several helper functions:

- `createSolidColorImage()`: Creates uniform color images
- `createGradientImage()`: Creates gradient patterns
- `createCheckerboardImage()`: Creates checkerboard patterns
- `createShapeImage()`: Creates images with geometric shapes
- `saveTestImage()`: Saves images to temporary directory

## Continuous Testing

For development:

```bash
# Watch mode (requires entr or similar)
find .. -name "*.cpp" -o -name "*.hpp" | entr -c cmake --build . && ./tests/image_similarity_test
```

## Test Results Interpretation

- **PASSED**: All assertions succeeded  
- **FAILED**: One or more assertions failed  
- **SKIPPED**: Test was disabled or not selected  

## Coverage Goals

Current test coverage aims for:
- 100% of public API functions
- All edge cases and boundary conditions
- Error handling paths
- Performance characteristics

## Adding New Tests

To add a new test:

1. Use the `TEST_F(ImageSimilarityTest, YourTestName)` macro  
2. Follow naming convention: `Category_Scenario_ExpectedResult`  
3. Include descriptive comments  
4. Test one concept per test case  
5. Use appropriate assertions (EXPECT_*, ASSERT_*)

Example:
```cpp
TEST_F(ImageSimilarityTest, NewFeature_ValidInput_ReturnsExpectedValue) {
    // Arrange
    cv::Mat testImage = createSolidColorImage(100, 100, cv::Scalar(128, 128, 128));
    
    // Act
    cv::Mat result;
    ImageSimilarityTestHelper::yourNewFunction(testImage, result);
    
    // Assert
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows, 100);
}
```

## Troubleshooting

### Tests Not Building

If Google Test is not found:
1. Verify installation: `dpkg -l | grep gtest` (Ubuntu) or `brew list googletest` (macOS)  
2. Set GTest path: `cmake -DGTest_DIR=/path/to/gtest ..`  
3. Check CMake output for error messages  

### Test Failures

1. Run failed test individually with verbose output  
2. Check for environment issues (missing directories, permissions)  
3. Verify OpenCV installation is correct  
4. Ensure test data is properly generated  

### Memory Issues

If tests crash or hang:
1. Reduce test data size in stress tests  
2. Check for memory leaks with valgrind: `valgrind ./tests/image_similarity_test`  
3. Monitor memory usage during execution  

## Contributing

When adding new features:
1. Write tests first (TDD approach)  
2. Ensure all existing tests pass  
3. Add tests for new functionality  
4. Update this README if adding new test categories  

## License

Same as main project license.