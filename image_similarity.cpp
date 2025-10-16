#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

struct ImageData {
    std::string path;
    std::string filename;
    cv::Mat image;
    cv::Mat histogram;
    cv::Mat edges;        // Precomputed edge map
    cv::Mat binary;       // Precomputed binary mask
};

class ImageSimilarityAnalyzer {
private:
    std::vector<ImageData> images;
    std::vector<std::vector<double>> similarityMatrix;

    // Load all images from directory recursively
    void loadImages(const std::string& directory) {
        std::cout << "Loading images from: " << directory << std::endl;

        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    cv::Mat img = cv::imread(path);
                    if (!img.empty()) {
                        ImageData imgData;
                        imgData.path = path;
                        imgData.filename = entry.path().filename().string();
                        imgData.image = img;

                        // Precompute all features
                        computeHistogram(img, imgData.histogram);
                        preprocessImage(img, imgData.edges, imgData.binary);

                        images.push_back(imgData);
                        std::cout << "  Loaded: " << imgData.filename << std::endl;
                    }
                }
            }
        }

        std::cout << "Total images loaded: " << images.size() << std::endl;
    }

    // Preprocess image: compute edges and binary mask
    void preprocessImage(const cv::Mat& image, cv::Mat& edges, cv::Mat& binary) {
        // Resize to standard size
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(128, 128));

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        // Compute edge map
        cv::Canny(gray, edges, 50, 150);

        // Compute binary threshold
        cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY_INV);
    }

    // Compute color histogram for an image
    void computeHistogram(const cv::Mat& image, cv::Mat& histogram) {
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

        // Normalize histograms
        cv::normalize(histB, histB, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histG, histG, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histR, histR, 0, 1, cv::NORM_MINMAX);

        // Concatenate histograms
        cv::vconcat(histB, histG, histogram);
        cv::vconcat(histogram, histR, histogram);
    }

    // Calculate similarity between two images using precomputed features
    double calculateSimilarity(const ImageData& img1, const ImageData& img2) {
        // Method 1: Edge-based similarity (focuses on shape/contour) - 40% weight
        cv::Mat edgeDiff;
        cv::absdiff(img1.edges, img2.edges, edgeDiff);
        double edgeSimilarity = 1.0 - (cv::sum(edgeDiff)[0] / (128.0 * 128.0 * 255.0));

        // Method 2: Histogram comparison (color distribution) - 30% weight
        double histSimilarity = cv::compareHist(img1.histogram, img2.histogram, cv::HISTCMP_CORREL);

        // Method 3: Binary mask similarity (structural similarity) - 30% weight
        cv::Mat diff;
        cv::absdiff(img1.binary, img2.binary, diff);
        double pixelDiff = cv::sum(diff)[0] / (128.0 * 128.0 * 255.0);
        double structSimilarity = 1.0 - pixelDiff;

        // Combine all methods with weights
        double combinedSimilarity = 0.4 * edgeSimilarity + 0.3 * histSimilarity + 0.3 * structSimilarity;

        // Apply non-linear transformation to spread out the scores
        // This makes differences more apparent
        combinedSimilarity = std::pow(combinedSimilarity, 3.0);

        // Ensure similarity is in [0, 1] range
        return std::max(0.0, std::min(1.0, combinedSimilarity));
    }

    // Compute the full similarity matrix
    void computeSimilarityMatrix() {
        size_t n = images.size();
        similarityMatrix.resize(n, std::vector<double>(n, 0.0));

        std::cout << "\nComputing similarity matrix..." << std::endl;

        for (size_t i = 0; i < n; ++i) {
            std::cout << "Processing image " << (i + 1) << "/" << n << ": "
                      << images[i].filename << std::endl;

            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    similarityMatrix[i][j] = 1.0; // Self-similarity is 1
                } else if (i > j) {
                    similarityMatrix[i][j] = similarityMatrix[j][i]; // Use symmetric property
                } else {
                    similarityMatrix[i][j] = calculateSimilarity(images[i], images[j]);
                }
            }
        }

        std::cout << "Similarity matrix computed successfully!" << std::endl;
    }

    // Export matrix to CSV file
    void exportToCSV(const std::string& filename) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }

        // Write header
        file << "Image";
        for (const auto& img : images) {
            file << "," << img.filename;
        }
        file << "\n";

        // Write data
        for (size_t i = 0; i < images.size(); ++i) {
            file << images[i].filename;
            for (size_t j = 0; j < images.size(); ++j) {
                file << "," << std::fixed << std::setprecision(4) << similarityMatrix[i][j];
            }
            file << "\n";
        }

        file.close();
        std::cout << "\nSimilarity matrix exported to: " << filename << std::endl;
    }

    // Print top N most similar pairs
    void printTopSimilarPairs(int topN = 10) {
        struct Pair {
            size_t i, j;
            double similarity;

            bool operator<(const Pair& other) const {
                return similarity > other.similarity; // Sort descending
            }
        };

        std::vector<Pair> pairs;

        for (size_t i = 0; i < images.size(); ++i) {
            for (size_t j = i + 1; j < images.size(); ++j) {
                pairs.push_back({i, j, similarityMatrix[i][j]});
            }
        }

        std::sort(pairs.begin(), pairs.end());

        std::cout << "\n=== Top " << topN << " Most Similar Image Pairs ===" << std::endl;
        for (int k = 0; k < std::min(topN, (int)pairs.size()); ++k) {
            const auto& p = pairs[k];
            std::cout << std::setw(3) << (k + 1) << ". "
                      << std::fixed << std::setprecision(4) << p.similarity << " - "
                      << images[p.i].filename << " <-> " << images[p.j].filename << std::endl;
        }
    }

    // Print statistics
    void printStatistics() {
        double sum = 0.0;
        double minSim = 1.0;
        double maxSim = 0.0;
        int count = 0;

        for (size_t i = 0; i < images.size(); ++i) {
            for (size_t j = i + 1; j < images.size(); ++j) {
                double sim = similarityMatrix[i][j];
                sum += sim;
                minSim = std::min(minSim, sim);
                maxSim = std::max(maxSim, sim);
                count++;
            }
        }

        double avgSim = sum / count;

        std::cout << "\n=== Similarity Statistics ===" << std::endl;
        std::cout << "Number of image pairs: " << count << std::endl;
        std::cout << "Average similarity: " << std::fixed << std::setprecision(4) << avgSim << std::endl;
        std::cout << "Minimum similarity: " << std::fixed << std::setprecision(4) << minSim << std::endl;
        std::cout << "Maximum similarity: " << std::fixed << std::setprecision(4) << maxSim << std::endl;
    }

public:
    void analyze(const std::string& directory, const std::string& outputFile = "similarity_matrix.csv") {
        // Load all images
        loadImages(directory);

        if (images.empty()) {
            std::cerr << "Error: No images found in directory: " << directory << std::endl;
            return;
        }

        // Compute similarity matrix
        computeSimilarityMatrix();

        // Export results
        exportToCSV(outputFile);

        // Print statistics and top pairs
        printStatistics();
        printTopSimilarPairs(15);
    }
};

int main(int argc, char** argv) {
    std::cout << "=== Image Similarity Analyzer ===" << std::endl;
    std::cout << "Author: Claude Code" << std::endl;
    std::cout << "================================\n" << std::endl;

    std::string directory = "Data/Images";
    std::string outputFile = "similarity_matrix.csv";

    // Parse command line arguments
    if (argc > 1) {
        directory = argv[1];
    }
    if (argc > 2) {
        outputFile = argv[2];
    }

    std::cout << "Input directory: " << directory << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "================================\n" << std::endl;

    try {
        ImageSimilarityAnalyzer analyzer;
        analyzer.analyze(directory, outputFile);

        std::cout << "\n=== Analysis Complete ===" << std::endl;
        std::cout << "Check " << outputFile << " for the full similarity matrix." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
