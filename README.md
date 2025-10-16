# Pictograph Similarity Analyzer

A C++ application that analyzes similarity between images using multiple computer vision techniques.

## Features

- Recursively loads all images from a directory
- Computes similarity using:
  - **Color Histogram Comparison** (50% weight): Analyzes color distribution - most important for pictographs
  - **Edge-based Similarity** (40% weight): Analyzes shape and contours using Canny edge detection
  - **Binary Mask Structural Similarity** (10% weight): Compares pixel-level structure
  - **Non-linear Transformation**: Applies cubic transformation to spread similarity scores
- Generates a full similarity matrix
- Exports results to CSV format
- Displays statistics and top similar pairs

## Requirements

### Windows (MSYS2/MinGW)

1. **Install MSYS2** (if not already installed)
   - Download from: https://www.msys2.org/

2. **Install OpenCV and build tools**
   ```bash
   pacman -S mingw-w64-x86_64-opencv
   pacman -S mingw-w64-x86_64-cmake
   pacman -S mingw-w64-x86_64-gcc
   pacman -S make
   ```

3. **Add to PATH** (if not already added)
   - `C:\msys64\mingw64\bin`

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libopencv-dev
```

### macOS

```bash
brew install cmake
brew install opencv
```

## Compilation

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
cmake --build .
```

The executable `image_similarity.exe` (or `image_similarity` on Linux/macOS) will be created in the `build` directory.

### Manual Compilation (Windows with g++)

```bash
g++ -std=c++17 image_similarity.cpp -o image_similarity \
    -I/mingw64/include/opencv4 \
    -L/mingw64/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
```

### Manual Compilation (Linux)

```bash
g++ -std=c++17 image_similarity.cpp -o image_similarity \
    `pkg-config --cflags --libs opencv4`
```

## Usage

### Basic Usage

```bash
# Run with default directory (Data/Images)
./build/image_similarity

# Or specify custom directory
./build/image_similarity path/to/images

# Specify both directory and output file
./build/image_similarity path/to/images output.csv
```

### Command Line Arguments

- **Argument 1**: Input directory containing images (default: `Data/Images`)
- **Argument 2**: Output CSV filename (default: `similarity_matrix.csv`)

### Example

```bash
./build/image_similarity Data/Images pictograph_similarity.csv
```

## Output

The program generates:

1. **CSV File**: Full similarity matrix with values between 0 (completely different) and 1 (identical)
   - Rows and columns represent image filenames
   - Each cell contains the similarity score

2. **Console Output**:
   - Loading progress
   - Computation progress
   - Similarity statistics (average, min, max)
   - Top 15 most similar image pairs

### Sample Output

```
=== Image Similarity Analyzer ===
================================

Loading images from: Data/Images
  Loaded: Pictograph_Animals_Cat.png
  Loaded: Pictograph_Animals_Dog.png
  ...
Total images loaded: 100

Computing similarity matrix...
Processing image 1/100: Pictograph_Animals_Cat.png
...
Similarity matrix computed successfully!

Similarity matrix exported to: similarity_matrix.csv

=== Similarity Statistics ===
Number of image pairs: 4950
Average similarity: 0.4523
Minimum similarity: 0.1234
Maximum similarity: 0.8967

=== Top 15 Most Similar Image Pairs ===
  1. 0.8967 - Pictograph_Animals_Cat.png <-> Pictograph_Animals_Lion.png
  2. 0.8754 - Pictograph_Music_Guitar.png <-> Pictograph_Music_Banjo.png
  ...
```

## How It Works

### Similarity Metrics

The algorithm combines three complementary approaches, optimized for pictographs with uniform white backgrounds:

1. **Color Histogram Comparison (50% weight)** - MOST IMPORTANT
   - Computes BGR color histograms (256 bins per channel)
   - Uses correlation method (cv::HISTCMP_CORREL)
   - Captures overall color distribution
   - Best for distinguishing objects with different colors
   - Highest weight because color is the primary differentiator in pictographs

2. **Edge-based Similarity (40% weight)**
   - Resizes images to 128x128 for standardization
   - Converts to grayscale
   - Applies Canny edge detection (thresholds: 50, 150)
   - Compares edge maps to focus on shape and contours
   - Best for distinguishing objects with different shapes

3. **Binary Mask Structural Similarity (10% weight)** - REDUCED
   - Applies binary threshold (threshold: 200) to separate foreground from background
   - Compares binary masks pixel-by-pixel
   - Focuses on overall structure and filled regions
   - Lower weight because all pictographs have uniform white backgrounds

4. **Non-linear Transformation**
   - After combining the three metrics, applies cubic transformation (x³)
   - Spreads out similarity scores for better discrimination
   - Makes differences between images more apparent

### Similarity Score Interpretation

- **0.9 - 1.0**: Nearly identical images
- **0.7 - 0.9**: Very similar (same category, similar objects)
- **0.5 - 0.7**: Moderately similar (some common features)
- **0.3 - 0.5**: Slightly similar (few common features)
- **0.0 - 0.3**: Very different images

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

## Troubleshooting

### OpenCV not found

If CMake cannot find OpenCV:

```bash
# Windows (MSYS2)
export OpenCV_DIR=/mingw64/lib/cmake/opencv4

# Linux
export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

# Then run cmake again
cmake ..
```

### "Cannot open include file: opencv2/opencv.hpp"

Ensure OpenCV is properly installed and the include path is correct.

### Large number of images causing memory issues

The program loads all images into memory. For very large datasets (1000+ images), consider:
- Reducing image resolution
- Processing in batches
- Using a more memory-efficient approach

## Future Enhancements

Possible improvements:
- Add perceptual hashing (pHash, dHash)
- Implement SIFT/SURF feature matching
- Add deep learning-based similarity (CNN embeddings)
- Support for batch processing
- Parallel processing for faster computation
- Interactive visualization of similar images

## License

This project is open source and available for educational purposes.