# Quick Start Guide - Testing Image Similarity Analyzer

## Quick Setup (5 minutes)

### Step 1: Install Google Test

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install libgtest-dev cmake build-essential libopencv-dev
```

**macOS:**
```bash
brew install googletest cmake opencv
```

**Windows (MSYS2):**
```bash
pacman -S mingw-w64-x86_64-gtest mingw-w64-x86_64-cmake mingw-w64-x86_64-opencv
```

### Step 2: Build Everything

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Step 3: Run Tests

```bash
./tests/image_similarity_test
```

## What Gets Tested

**37 comprehensive test cases** covering all major functionality.

See TEST_README.md for complete details.

## Running Specific Tests

```bash
# Run only histogram tests
./tests/image_similarity_test --gtest_filter=*Histogram*

# List all available tests
./tests/image_similarity_test --gtest_list_tests
```