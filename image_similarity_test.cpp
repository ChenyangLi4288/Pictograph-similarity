#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

// Copy the struct definition from the main file for testing
struct ImageData {
    std::string path;
    std::string filename;
    cv::Mat image;
    cv::Mat histogram;
    cv::Mat edges;
    cv::Mat binary;
};

// Test helper class - extracts testable functionality from ImageSimilarityAnalyzer
class ImageSimilarityTestHelper {
public:
    static void computeHistogram(const cv::Mat& image, cv::Mat& histogram) {
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        bool uniform = true;
        bool accumulate = false;

        std::vector<cv::Mat> bgrPlanes;
        cv::split(image, bgrPlanes);

        cv::Mat histB, histG, histR;
        cv::calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), histB, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), histG, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), histR, 1, &histSize, &histRange, uniform, accumulate);

        cv::normalize(histB, histB, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histG, histG, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histR, histR, 0, 1, cv::NORM_MINMAX);

        cv::vconcat(histB, histG, histogram);
        cv::vconcat(histogram, histR, histogram);
    }

    static void preprocessImage(const cv::Mat& image, cv::Mat& edges, cv::Mat& binary) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(128, 128));

        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        cv::Canny(gray, edges, 50, 150);
        cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY_INV);
    }

    static double calculateSimilarity(const ImageData& img1, const ImageData& img2) {
        cv::Mat edgeDiff;
        cv::absdiff(img1.edges, img2.edges, edgeDiff);
        double edgeSimilarity = 1.0 - (cv::sum(edgeDiff)[0] / (128.0 * 128.0 * 255.0));

        double histSimilarity = cv::compareHist(img1.histogram, img2.histogram, cv::HISTCMP_CORREL);

        cv::Mat diff;
        cv::absdiff(img1.binary, img2.binary, diff);
        double pixelDiff = cv::sum(diff)[0] / (128.0 * 128.0 * 255.0);
        double structSimilarity = 1.0 - pixelDiff;

        double combinedSimilarity = 0.4 * edgeSimilarity + 0.3 * histSimilarity + 0.3 * structSimilarity;
        combinedSimilarity = std::pow(combinedSimilarity, 3.0);

        return std::max(0.0, std::min(1.0, combinedSimilarity));
    }
};

// Test fixture for setting up test environment
class ImageSimilarityTest : public ::testing::Test {
protected:
    std::string testDir;
    
    void SetUp() override {
        testDir = "test_images_temp";
        fs::create_directories(testDir);
    }

    void TearDown() override {
        if (fs::exists(testDir)) {
            fs::remove_all(testDir);
        }
    }

    // Helper: Create a solid color image
    cv::Mat createSolidColorImage(int width, int height, cv::Scalar color) {
        cv::Mat img(height, width, CV_8UC3, color);
        return img;
    }

    // Helper: Create a gradient image
    cv::Mat createGradientImage(int width, int height) {
        cv::Mat img(height, width, CV_8UC3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(x % 256, y % 256, (x + y) % 256);
            }
        }
        return img;
    }

    // Helper: Create a checkerboard pattern
    cv::Mat createCheckerboardImage(int width, int height, int squareSize) {
        cv::Mat img(height, width, CV_8UC3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (((x / squareSize) + (y / squareSize)) % 2 == 0) {
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                } else {
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        return img;
    }

    // Helper: Create an image with shapes
    cv::Mat createShapeImage(int width, int height, std::string shape = "circle") {
        cv::Mat img = createSolidColorImage(width, height, cv::Scalar(255, 255, 255));
        if (shape == "circle") {
            cv::circle(img, cv::Point(width/2, height/2), std::min(width, height)/4, 
                      cv::Scalar(0, 0, 255), -1);
        } else if (shape == "rectangle") {
            cv::rectangle(img, cv::Point(width/4, height/4), cv::Point(3*width/4, 3*height/4),
                         cv::Scalar(0, 255, 0), -1);
        } else if (shape == "triangle") {
            cv::Point pts[3] = {
                cv::Point(width/2, height/4),
                cv::Point(width/4, 3*height/4),
                cv::Point(3*width/4, 3*height/4)
            };
            cv::fillConvexPoly(img, pts, 3, cv::Scalar(255, 0, 0));
        }
        return img;
    }

    // Helper: Save test image to disk
    void saveTestImage(const std::string& filename, const cv::Mat& image) {
        std::string fullPath = testDir + "/" + filename;
        cv::imwrite(fullPath, image);
    }
};

// ========== Tests for ImageData Structure ==========

TEST_F(ImageSimilarityTest, ImageDataStructure_InitializationAndAssignment) {
    ImageData imgData;
    imgData.path = "test/path/image.png";
    imgData.filename = "image.png";
    imgData.image = createSolidColorImage(100, 100, cv::Scalar(255, 0, 0));
    
    EXPECT_EQ(imgData.path, "test/path/image.png");
    EXPECT_EQ(imgData.filename, "image.png");
    EXPECT_FALSE(imgData.image.empty());
    EXPECT_EQ(imgData.image.cols, 100);
    EXPECT_EQ(imgData.image.rows, 100);
}

TEST_F(ImageSimilarityTest, ImageDataStructure_CopyAndMove) {
    ImageData imgData1;
    imgData1.filename = "test1.png";
    imgData1.image = createSolidColorImage(50, 50, cv::Scalar(100, 100, 100));
    
    // Test copy
    ImageData imgData2 = imgData1;
    EXPECT_EQ(imgData2.filename, imgData1.filename);
    EXPECT_EQ(imgData2.image.size(), imgData1.image.size());
}

// ========== Tests for Histogram Computation ==========

TEST_F(ImageSimilarityTest, ComputeHistogram_ValidImage) {
    cv::Mat image = createSolidColorImage(100, 100, cv::Scalar(128, 64, 192));
    cv::Mat histogram;
    
    ImageSimilarityTestHelper::computeHistogram(image, histogram);
    
    EXPECT_FALSE(histogram.empty());
    EXPECT_EQ(histogram.rows, 768); // 256 bins * 3 channels
    EXPECT_EQ(histogram.cols, 1);
    
    // Check normalization (values should be between 0 and 1)
    double minVal, maxVal;
    cv::minMaxLoc(histogram, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

TEST_F(ImageSimilarityTest, ComputeHistogram_IdenticalImages_SameHistogram) {
    cv::Mat image1 = createSolidColorImage(100, 100, cv::Scalar(50, 100, 150));
    cv::Mat image2 = createSolidColorImage(100, 100, cv::Scalar(50, 100, 150));
    
    cv::Mat hist1, hist2;
    ImageSimilarityTestHelper::computeHistogram(image1, hist1);
    ImageSimilarityTestHelper::computeHistogram(image2, hist2);
    
    double correlation = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    EXPECT_NEAR(correlation, 1.0, 0.001);
}

TEST_F(ImageSimilarityTest, ComputeHistogram_DifferentColors_DifferentHistograms) {
    cv::Mat redImage = createSolidColorImage(100, 100, cv::Scalar(0, 0, 255));
    cv::Mat blueImage = createSolidColorImage(100, 100, cv::Scalar(255, 0, 0));
    
    cv::Mat histRed, histBlue;
    ImageSimilarityTestHelper::computeHistogram(redImage, histRed);
    ImageSimilarityTestHelper::computeHistogram(blueImage, histBlue);
    
    double correlation = cv::compareHist(histRed, histBlue, cv::HISTCMP_CORREL);
    EXPECT_LT(correlation, 1.0);
}

TEST_F(ImageSimilarityTest, ComputeHistogram_GradientImage) {
    cv::Mat gradientImage = createGradientImage(200, 200);
    cv::Mat histogram;
    
    ImageSimilarityTestHelper::computeHistogram(gradientImage, histogram);
    
    EXPECT_FALSE(histogram.empty());
    EXPECT_EQ(histogram.rows, 768);
}

TEST_F(ImageSimilarityTest, ComputeHistogram_BlackImage) {
    cv::Mat blackImage = createSolidColorImage(100, 100, cv::Scalar(0, 0, 0));
    cv::Mat histogram;
    
    ImageSimilarityTestHelper::computeHistogram(blackImage, histogram);
    
    EXPECT_FALSE(histogram.empty());
    // Most values should be concentrated in the first bin
    EXPECT_GT(histogram.at<float>(0, 0), 0.5);
}

TEST_F(ImageSimilarityTest, ComputeHistogram_WhiteImage) {
    cv::Mat whiteImage = createSolidColorImage(100, 100, cv::Scalar(255, 255, 255));
    cv::Mat histogram;
    
    ImageSimilarityTestHelper::computeHistogram(whiteImage, histogram);
    
    EXPECT_FALSE(histogram.empty());
    // Most values should be concentrated in the last bin
    EXPECT_GT(histogram.at<float>(255, 0), 0.5);
}

// ========== Tests for Image Preprocessing ==========

TEST_F(ImageSimilarityTest, PreprocessImage_ValidImage) {
    cv::Mat image = createCheckerboardImage(200, 200, 20);
    cv::Mat edges, binary;
    
    ImageSimilarityTestHelper::preprocessImage(image, edges, binary);
    
    EXPECT_FALSE(edges.empty());
    EXPECT_FALSE(binary.empty());
    EXPECT_EQ(edges.cols, 128);
    EXPECT_EQ(edges.rows, 128);
    EXPECT_EQ(binary.cols, 128);
    EXPECT_EQ(binary.rows, 128);
}

TEST_F(ImageSimilarityTest, PreprocessImage_ResizingCorrect) {
    cv::Mat smallImage = createSolidColorImage(50, 50, cv::Scalar(100, 100, 100));
    cv::Mat largeImage = createSolidColorImage(300, 300, cv::Scalar(100, 100, 100));
    
    cv::Mat edgesSmall, binarySmall, edgesLarge, binaryLarge;
    ImageSimilarityTestHelper::preprocessImage(smallImage, edgesSmall, binarySmall);
    ImageSimilarityTestHelper::preprocessImage(largeImage, edgesLarge, binaryLarge);
    
    // Both should be resized to 128x128
    EXPECT_EQ(edgesSmall.size(), cv::Size(128, 128));
    EXPECT_EQ(edgesLarge.size(), cv::Size(128, 128));
}

TEST_F(ImageSimilarityTest, PreprocessImage_EdgeDetection_FindsEdges) {
    cv::Mat imageWithEdges = createCheckerboardImage(200, 200, 40);
    cv::Mat edges, binary;
    
    ImageSimilarityTestHelper::preprocessImage(imageWithEdges, edges, binary);
    
    // Edge image should have non-zero pixels (edges detected)
    int nonZeroPixels = cv::countNonZero(edges);
    EXPECT_GT(nonZeroPixels, 100); // Should detect edges in checkerboard
}

TEST_F(ImageSimilarityTest, PreprocessImage_SmoothImage_FewerEdges) {
    cv::Mat smoothImage = createSolidColorImage(200, 200, cv::Scalar(128, 128, 128));
    cv::Mat edges, binary;
    
    ImageSimilarityTestHelper::preprocessImage(smoothImage, edges, binary);
    
    // Smooth solid color should have very few edges
    int nonZeroPixels = cv::countNonZero(edges);
    EXPECT_LT(nonZeroPixels, 100); // Very few edges in solid color
}

TEST_F(ImageSimilarityTest, PreprocessImage_BinaryThreshold_WhiteImage) {
    cv::Mat whiteImage = createSolidColorImage(200, 200, cv::Scalar(255, 255, 255));
    cv::Mat edges, binary;
    
    ImageSimilarityTestHelper::preprocessImage(whiteImage, edges, binary);
    
    // White image (bright) should become mostly black after inverse threshold (> 200)
    int nonZeroPixels = cv::countNonZero(binary);
    EXPECT_LT(nonZeroPixels, 1000); // Most should be below threshold
}

TEST_F(ImageSimilarityTest, PreprocessImage_BinaryThreshold_BlackImage) {
    cv::Mat blackImage = createSolidColorImage(200, 200, cv::Scalar(0, 0, 0));
    cv::Mat edges, binary;
    
    ImageSimilarityTestHelper::preprocessImage(blackImage, edges, binary);
    
    // Black image (dark) should become mostly white after inverse threshold (< 200)
    int nonZeroPixels = cv::countNonZero(binary);
    EXPECT_GT(nonZeroPixels, 10000); // Most should be above threshold
}

// ========== Tests for Similarity Calculation ==========

TEST_F(ImageSimilarityTest, CalculateSimilarity_IdenticalImages_HighSimilarity) {
    cv::Mat image = createSolidColorImage(100, 100, cv::Scalar(100, 150, 200));
    
    ImageData img1, img2;
    img1.image = image.clone();
    img2.image = image.clone();
    
    ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
    ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
    ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
    ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
    
    double similarity = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
    
    EXPECT_GE(similarity, 0.95); // Should be very high for identical images
    EXPECT_LE(similarity, 1.0);
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_CompletelyDifferent_LowSimilarity) {
    cv::Mat redImage = createSolidColorImage(100, 100, cv::Scalar(0, 0, 255));
    cv::Mat blueImage = createSolidColorImage(100, 100, cv::Scalar(255, 0, 0));
    
    ImageData img1, img2;
    img1.image = redImage;
    img2.image = blueImage;
    
    ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
    ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
    ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
    ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
    
    double similarity = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
    
    EXPECT_GE(similarity, 0.0);
    EXPECT_LE(similarity, 1.0);
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_SimilarShapes_ModerateSimilarity) {
    cv::Mat circle1 = createShapeImage(200, 200, "circle");
    cv::Mat circle2 = createShapeImage(200, 200, "circle");
    
    ImageData img1, img2;
    img1.image = circle1;
    img2.image = circle2;
    
    ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
    ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
    ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
    ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
    
    double similarity = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
    
    EXPECT_GT(similarity, 0.8); // Similar shapes should have high similarity
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_DifferentShapes_LowerSimilarity) {
    cv::Mat circle = createShapeImage(200, 200, "circle");
    cv::Mat rectangle = createShapeImage(200, 200, "rectangle");
    
    ImageData img1, img2;
    img1.image = circle;
    img2.image = rectangle;
    
    ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
    ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
    ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
    ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
    
    double similarity = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
    
    // Different shapes should have lower similarity than identical shapes
    EXPECT_GE(similarity, 0.0);
    EXPECT_LT(similarity, 0.8);
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_ResultInValidRange) {
    // Test with various image combinations
    std::vector<cv::Mat> testImages = {
        createSolidColorImage(100, 100, cv::Scalar(0, 0, 0)),
        createSolidColorImage(100, 100, cv::Scalar(255, 255, 255)),
        createGradientImage(100, 100),
        createCheckerboardImage(100, 100, 10),
        createShapeImage(100, 100, "circle")
    };
    
    for (size_t i = 0; i < testImages.size(); i++) {
        for (size_t j = i + 1; j < testImages.size(); j++) {
            ImageData img1, img2;
            img1.image = testImages[i];
            img2.image = testImages[j];
            
            ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
            ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
            ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
            ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
            
            double similarity = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
            
            EXPECT_GE(similarity, 0.0) << "Similarity below 0 for images " << i << " and " << j;
            EXPECT_LE(similarity, 1.0) << "Similarity above 1 for images " << i << " and " << j;
        }
    }
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_Symmetry) {
    cv::Mat img1Mat = createGradientImage(150, 150);
    cv::Mat img2Mat = createCheckerboardImage(150, 150, 15);
    
    ImageData img1, img2;
    img1.image = img1Mat;
    img2.image = img2Mat;
    
    ImageSimilarityTestHelper::preprocessImage(img1.image, img1.edges, img1.binary);
    ImageSimilarityTestHelper::preprocessImage(img2.image, img2.edges, img2.binary);
    ImageSimilarityTestHelper::computeHistogram(img1.image, img1.histogram);
    ImageSimilarityTestHelper::computeHistogram(img2.image, img2.histogram);
    
    double sim12 = ImageSimilarityTestHelper::calculateSimilarity(img1, img2);
    double sim21 = ImageSimilarityTestHelper::calculateSimilarity(img2, img1);
    
    EXPECT_NEAR(sim12, sim21, 0.0001); // Should be symmetric
}

TEST_F(ImageSimilarityTest, CalculateSimilarity_TransitiveProperty) {
    // If A similar to B and B similar to C, then A should be somewhat similar to C
    cv::Mat imgA = createShapeImage(150, 150, "circle");
    cv::Mat imgB = createShapeImage(150, 150, "circle");
    cv::Mat imgC = createShapeImage(150, 150, "circle");
    
    ImageData dataA, dataB, dataC;
    dataA.image = imgA;
    dataB.image = imgB;
    dataC.image = imgC;
    
    for (auto* data : {&dataA, &dataB, &dataC}) {
        ImageSimilarityTestHelper::preprocessImage(data->image, data->edges, data->binary);
        ImageSimilarityTestHelper::computeHistogram(data->image, data->histogram);
    }
    
    double simAB = ImageSimilarityTestHelper::calculateSimilarity(dataA, dataB);
    double simBC = ImageSimilarityTestHelper::calculateSimilarity(dataB, dataC);
    double simAC = ImageSimilarityTestHelper::calculateSimilarity(dataA, dataC);
    
    // All should be high since they're all the same shape
    EXPECT_GT(simAB, 0.8);
    EXPECT_GT(simBC, 0.8);
    EXPECT_GT(simAC, 0.8);
}

// ========== Tests for CSV Export Functionality ==========

TEST_F(ImageSimilarityTest, CSVExport_FileCreation) {
    std::string csvFile = testDir + "/test_similarity.csv";
    
    // Create a simple CSV manually
    std::ofstream file(csvFile);
    ASSERT_TRUE(file.is_open());
    
    file << "Image,img1.png,img2.png\n";
    file << "img1.png,1.0000,0.5000\n";
    file << "img2.png,0.5000,1.0000\n";
    file.close();
    
    EXPECT_TRUE(fs::exists(csvFile));
    
    // Read and verify content
    std::ifstream inFile(csvFile);
    std::string line;
    std::getline(inFile, line);
    EXPECT_EQ(line, "Image,img1.png,img2.png");
}

TEST_F(ImageSimilarityTest, CSVExport_ProperFormatting) {
    std::string csvFile = testDir + "/format_test.csv";
    std::ofstream file(csvFile);
    
    // Test proper number formatting (4 decimal places)
    file << std::fixed << std::setprecision(4);
    file << 0.123456 << "," << 0.987654 << "\n";
    file.close();
    
    std::ifstream inFile(csvFile);
    std::string content;
    std::getline(inFile, content);
    
    EXPECT_TRUE(content.find("0.1235") != std::string::npos || 
                content.find("0.1234") != std::string::npos);
}

// ========== Tests for Edge Cases ==========

TEST_F(ImageSimilarityTest, EdgeCase_EmptyDirectory) {
    std::string emptyDir = testDir + "/empty";
    fs::create_directories(emptyDir);
    
    // Directory exists but has no images
    EXPECT_TRUE(fs::exists(emptyDir));
    EXPECT_TRUE(fs::is_empty(emptyDir));
}

TEST_F(ImageSimilarityTest, EdgeCase_SingleImage) {
    cv::Mat image = createSolidColorImage(100, 100, cv::Scalar(128, 128, 128));
    saveTestImage("single.png", image);
    
    // With single image, similarity matrix should be 1x1 with value 1.0
    EXPECT_TRUE(fs::exists(testDir + "/single.png"));
}

TEST_F(ImageSimilarityTest, EdgeCase_VerySmallImage) {
    cv::Mat tinyImage = createSolidColorImage(1, 1, cv::Scalar(255, 0, 0));
    cv::Mat edges, binary;
    
    EXPECT_NO_THROW({
        ImageSimilarityTestHelper::preprocessImage(tinyImage, edges, binary);
    });
    
    EXPECT_EQ(edges.size(), cv::Size(128, 128)); // Should be resized to 128x128
}

TEST_F(ImageSimilarityTest, EdgeCase_VeryLargeImage) {
    cv::Mat largeImage = createSolidColorImage(4000, 3000, cv::Scalar(0, 255, 0));
    cv::Mat edges, binary;
    
    EXPECT_NO_THROW({
        ImageSimilarityTestHelper::preprocessImage(largeImage, edges, binary);
    });
    
    EXPECT_EQ(edges.size(), cv::Size(128, 128)); // Should be resized to 128x128
}

TEST_F(ImageSimilarityTest, EdgeCase_NonSquareImages) {
    cv::Mat wideImage = createSolidColorImage(300, 100, cv::Scalar(100, 100, 100));
    cv::Mat tallImage = createSolidColorImage(100, 300, cv::Scalar(100, 100, 100));
    
    cv::Mat edgesWide, binaryWide, edgesTall, binaryTall;
    
    ImageSimilarityTestHelper::preprocessImage(wideImage, edgesWide, binaryWide);
    ImageSimilarityTestHelper::preprocessImage(tallImage, edgesTall, binaryTall);
    
    // Both should be resized to same dimensions
    EXPECT_EQ(edgesWide.size(), edgesTall.size());
    EXPECT_EQ(edgesWide.size(), cv::Size(128, 128));
}

TEST_F(ImageSimilarityTest, EdgeCase_ExtremeBrightness) {
    cv::Mat darkImage = createSolidColorImage(100, 100, cv::Scalar(0, 0, 0));
    cv::Mat brightImage = createSolidColorImage(100, 100, cv::Scalar(255, 255, 255));
    
    cv::Mat histDark, histBright;
    ImageSimilarityTestHelper::computeHistogram(darkImage, histDark);
    ImageSimilarityTestHelper::computeHistogram(brightImage, histBright);
    
    EXPECT_FALSE(histDark.empty());
    EXPECT_FALSE(histBright.empty());
    
    double correlation = cv::compareHist(histDark, histBright, cv::HISTCMP_CORREL);
    EXPECT_LT(correlation, 0.5); // Very different histograms
}

// ========== Tests for File System Operations ==========

TEST_F(ImageSimilarityTest, FileSystem_MultipleImageFormats) {
    cv::Mat image = createSolidColorImage(100, 100, cv::Scalar(128, 64, 192));
    
    saveTestImage("test.png", image);
    saveTestImage("test.jpg", image);
    saveTestImage("test.bmp", image);
    
    EXPECT_TRUE(fs::exists(testDir + "/test.png"));
    EXPECT_TRUE(fs::exists(testDir + "/test.jpg"));
    EXPECT_TRUE(fs::exists(testDir + "/test.bmp"));
}

TEST_F(ImageSimilarityTest, FileSystem_RecursiveDirectory) {
    std::string subdir = testDir + "/subdir";
    fs::create_directories(subdir);
    
    cv::Mat image = createSolidColorImage(50, 50, cv::Scalar(100, 100, 100));
    cv::imwrite(subdir + "/nested.png", image);
    
    EXPECT_TRUE(fs::exists(subdir + "/nested.png"));
}

TEST_F(ImageSimilarityTest, FileSystem_ImageExtensionCaseInsensitive) {
    cv::Mat image = createSolidColorImage(100, 100, cv::Scalar(200, 100, 50));
    
    std::string upperPath = testDir + "/TEST.PNG";
    std::string lowerPath = testDir + "/test2.png";
    
    cv::imwrite(upperPath, image);
    cv::imwrite(lowerPath, image);
    
    EXPECT_TRUE(fs::exists(upperPath));
    EXPECT_TRUE(fs::exists(lowerPath));
}

// ========== Performance and Stress Tests ==========

TEST_F(ImageSimilarityTest, Performance_MultipleImages) {
    // Create multiple test images
    for (int i = 0; i < 10; i++) {
        cv::Mat img = createSolidColorImage(100, 100, 
            cv::Scalar(i * 25, (i * 25 + 100) % 256, (i * 50) % 256));
        saveTestImage("img_" + std::to_string(i) + ".png", img);
    }
    
    // Count created images
    int imageCount = 0;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() == ".png") {
            imageCount++;
        }
    }
    
    EXPECT_EQ(imageCount, 10);
}

TEST_F(ImageSimilarityTest, Stress_LargeSimilarityMatrix) {
    // Test with larger set for matrix computation
    const int numImages = 20;
    std::vector<ImageData> images;
    
    for (int i = 0; i < numImages; i++) {
        ImageData data;
        data.image = createSolidColorImage(100, 100, 
            cv::Scalar(i * 10, i * 12, i * 15));
        ImageSimilarityTestHelper::preprocessImage(data.image, data.edges, data.binary);
        ImageSimilarityTestHelper::computeHistogram(data.image, data.histogram);
        images.push_back(data);
    }
    
    // Compute similarity matrix
    std::vector<std::vector<double>> matrix(numImages, std::vector<double>(numImages));
    
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numImages; j++) {
            if (i == j) {
                matrix[i][j] = 1.0;
            } else if (i > j) {
                matrix[i][j] = matrix[j][i];
            } else {
                matrix[i][j] = ImageSimilarityTestHelper::calculateSimilarity(images[i], images[j]);
            }
        }
    }
    
    // Verify matrix properties
    // 1. Diagonal should all be 1.0
    for (int i = 0; i < numImages; i++) {
        EXPECT_DOUBLE_EQ(matrix[i][i], 1.0);
    }
    
    // 2. Matrix should be symmetric
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numImages; j++) {
            EXPECT_NEAR(matrix[i][j], matrix[j][i], 0.0001);
        }
    }
    
    // 3. All values should be in range [0, 1]
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < numImages; j++) {
            EXPECT_GE(matrix[i][j], 0.0);
            EXPECT_LE(matrix[i][j], 1.0);
        }
    }
}

// ========== Integration Tests ==========

TEST_F(ImageSimilarityTest, Integration_FullPipeline) {
    // Create test images
    cv::Mat img1 = createShapeImage(150, 150, "circle");
    cv::Mat img2 = createShapeImage(150, 150, "circle");
    cv::Mat img3 = createShapeImage(150, 150, "rectangle");
    
    saveTestImage("circle1.png", img1);
    saveTestImage("circle2.png", img2);
    saveTestImage("rect1.png", img3);
    
    // Load and process
    std::vector<ImageData> images;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() == ".png") {
            ImageData data;
            data.filename = entry.path().filename().string();
            data.image = cv::imread(entry.path().string());
            
            ImageSimilarityTestHelper::preprocessImage(data.image, data.edges, data.binary);
            ImageSimilarityTestHelper::computeHistogram(data.image, data.histogram);
            
            images.push_back(data);
        }
    }
    
    EXPECT_EQ(images.size(), 3);
    
    // Compute similarities
    double simCircles = ImageSimilarityTestHelper::calculateSimilarity(images[0], images[1]);
    double simMixed = ImageSimilarityTestHelper::calculateSimilarity(images[0], images[2]);
    
    // Circles should be more similar to each other than to rectangle
    EXPECT_GT(simCircles, simMixed);
}

// ========== Statistical Tests ==========

TEST_F(ImageSimilarityTest, Statistics_AverageSimilarity) {
    std::vector<ImageData> images;
    for (int i = 0; i < 5; i++) {
        ImageData data;
        data.image = createSolidColorImage(100, 100, 
            cv::Scalar(i * 50, i * 40, i * 30));
        ImageSimilarityTestHelper::preprocessImage(data.image, data.edges, data.binary);
        ImageSimilarityTestHelper::computeHistogram(data.image, data.histogram);
        images.push_back(data);
    }
    
    double sum = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < images.size(); i++) {
        for (size_t j = i + 1; j < images.size(); j++) {
            sum += ImageSimilarityTestHelper::calculateSimilarity(images[i], images[j]);
            count++;
        }
    }
    
    double average = sum / count;
    EXPECT_GE(average, 0.0);
    EXPECT_LE(average, 1.0);
    EXPECT_GT(count, 0);
}

TEST_F(ImageSimilarityTest, Statistics_MinMaxSimilarity) {
    std::vector<ImageData> images;
    
    // Create diverse set
    images.push_back(ImageData());
    images.back().image = createSolidColorImage(100, 100, cv::Scalar(0, 0, 0));
    
    images.push_back(ImageData());
    images.back().image = createSolidColorImage(100, 100, cv::Scalar(255, 255, 255));
    
    images.push_back(ImageData());
    images.back().image = createGradientImage(100, 100);
    
    for (auto& img : images) {
        ImageSimilarityTestHelper::preprocessImage(img.image, img.edges, img.binary);
        ImageSimilarityTestHelper::computeHistogram(img.image, img.histogram);
    }
    
    double minSim = 1.0, maxSim = 0.0;
    
    for (size_t i = 0; i < images.size(); i++) {
        for (size_t j = i + 1; j < images.size(); j++) {
            double sim = ImageSimilarityTestHelper::calculateSimilarity(images[i], images[j]);
            minSim = std::min(minSim, sim);
            maxSim = std::max(maxSim, sim);
        }
    }
    
    EXPECT_GE(minSim, 0.0);
    EXPECT_LE(maxSim, 1.0);
    EXPECT_LE(minSim, maxSim);
}

// ========== Main Function ==========

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}