# Pictograph Similarity Analyzer

Analyzes similarity between images using computer vision techniques and deep learning. Provides both traditional CV-based (C++) and CNN-based (Python) approaches.

## Features

- Recursively loads all images from a directory
- Computes similarity using:
  - **Foreground-Only Color Histogram** (60% weight): Analyzes color distribution of the object only, excluding white background
  - **Outer Boundary Detection** (30% weight): Compares external silhouettes, ignoring internal patterns
  - **Binary Mask Similarity** (10% weight): Compares overall shape and filled area
- Optimized for pictographs with uniform white backgrounds
- Generates a full similarity matrix
- Exports results to CSV format
- Displays statistics and top 15 similar pairs

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

### Approach 1: Traditional CV-based (C++)

Hand-crafted features optimized for pictographs with white backgrounds.

#### Basic Usage

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

#### Example

```bash
./build/image_similarity Data/Images pictograph_similarity.csv
```

### Approach 2: CNN-based (Python)

Uses pre-trained deep learning models (ResNet50) for automatic feature extraction.

#### Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run CNN similarity analyzer:**
```bash
# Basic usage (default: Data/Images)
python cnn_similarity.py

# Specify directory and output
python cnn_similarity.py Data/Images similarity_cnn.csv

# Use different model
python cnn_similarity.py --model resnet101 Data/Images output.csv
```

#### Available Models

- `resnet50` (default): Fast, good balance
- `resnet101`: More accurate, slower
- `efficientnet_b0`: Efficient, compact

#### Example

```bash
python cnn_similarity.py Data/Images similarity_cnn.csv --model resnet50
```

#### Advantages of CNN Approach

- Automatically learns features from millions of images
- No manual feature engineering
- Captures high-level semantic similarity
- Works well for complex visual patterns

#### Disadvantages

- Requires PyTorch installation (~2GB)
- Slower than hand-crafted features
- Less interpretable

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

1. **Foreground-Only Color Histogram Comparison (60% weight)**
   - Computes BGR color histograms (256 bins per channel) **only on the foreground object**
   - Uses binary mask to exclude white background pixels from histogram calculation
   - Uses correlation method (cv::HISTCMP_CORREL)
   - Captures the actual object's color distribution without background contamination
   - Best for distinguishing objects with different colors
   - Highest weight because color is the primary differentiator in pictographs

2. **Outer Boundary Similarity (30% weight)**
   - Resizes images to 128x128 for standardization
   - Extracts only the **external contour/silhouette** of the object
   - Ignores internal details (like basketball lines or pumpkin ridges)
   - Uses RETR_EXTERNAL to get outer boundary only
   - Compares boundary shapes to focus on overall silhouette
   - Best for distinguishing objects with different overall shapes

3. **Binary Mask Structural Similarity (10% weight)**
   - Applies binary threshold (threshold: 200) to separate foreground from background
   - Compares binary masks pixel-by-pixel
   - Focuses on overall filled area and structure
   - Lower weight because outer boundary already captures shape information

4. **Linear Combination (No Transformation)**
   - Combines the three metrics using weighted sum: 0.6 × color + 0.3 × boundary + 0.1 × structure
   - No power transformation needed since foreground-only histograms provide excellent discrimination

### Similarity Score Interpretation

- **0.9 - 1.0**: Nearly identical images (duplicates or extremely similar)
- **0.7 - 0.9**: Very similar (similar color and shape)
- **0.5 - 0.7**: Moderately similar (similar color or shape)
- **0.3 - 0.5**: Somewhat similar (some shared features)
- **0.2 - 0.3**: Slightly similar (few shared features)

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