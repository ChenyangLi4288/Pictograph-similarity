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

    /**
     * @brief Load image files from a directory and precompute per-image features.
     *
     * Recursively traverses the given `directory`, reads regular files with extensions
     * .png, .jpg, .jpeg, or .bmp (case-insensitive), and for each successfully loaded image
     * stores its path, filename, original image, computed color histogram, edge map, and
     * binary mask into the analyzer's internal image list.
     *
     * @param directory Root directory to search for image files.
     */
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

    /**
     * @brief Prepare a fixed-size grayscale image and compute its outer contour and binary mask.
     *
     * Resizes the input to 128x128, converts it to grayscale, extracts the outer boundary/silhouette
     * of the object (ignoring internal details), and computes an inverted binary mask via thresholding.
     *
     * @param image Input BGR image.
     * @param edges Output single-channel contour map (128x128). Contains only the external boundary of the object, ignoring internal patterns.
     * @param binary Output single-channel binary mask (128x128). Produced by thresholding the grayscale image at 200 with inversion (resulting pixels are 0 or 255).
     */
    void preprocessImage(const cv::Mat& image, cv::Mat& edges, cv::Mat& binary) {
        // Resize to standard size
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(128, 128));

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        // Compute binary threshold to separate object from white background
        cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY_INV);

        // Find contours and draw only the EXTERNAL boundary (not internal details)
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Create edge map with only outer boundary
        edges = cv::Mat::zeros(binary.size(), CV_8UC1);
        cv::drawContours(edges, contours, -1, cv::Scalar(255), 2);  // Draw all external contours
    }

    /**
     * @brief Computes a concatenated color histogram for ONLY the foreground object, excluding white background.
     *
     * Produces a single-column histogram containing three 256-bin channel histograms
     * stacked in B, G, R order. Each channel histogram is normalized to the range [0, 1].
     * Uses a mask to compute histogram only on non-white pixels (the object itself).
     *
     * @param image Input 3-channel BGR image.
     * @param histogram Output single-column cv::Mat with size 768x1 (256 bins × 3 channels),
     *                  containing the normalized B, then G, then R histograms of the foreground object only.
     */
    void computeHistogram(const cv::Mat& image, cv::Mat& histogram) {
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        bool uniform = true;
        bool accumulate = false;

        // Create mask to exclude white background (threshold > 200 = white background)
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(128, 128));
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        cv::Mat mask;
        cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY_INV);  // Mask = foreground only

        // Split channels and compute histograms ONLY on foreground pixels
        std::vector<cv::Mat> bgrPlanes;
        cv::split(resized, bgrPlanes);

        cv::Mat histB, histG, histR;
        cv::calcHist(&bgrPlanes[0], 1, 0, mask, histB, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgrPlanes[1], 1, 0, mask, histG, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgrPlanes[2], 1, 0, mask, histR, 1, &histSize, &histRange, uniform, accumulate);

        // Normalize histograms
        cv::normalize(histB, histB, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histG, histG, 0, 1, cv::NORM_MINMAX);
        cv::normalize(histR, histR, 0, 1, cv::NORM_MINMAX);

        // Concatenate histograms
        cv::vconcat(histB, histG, histogram);
        cv::vconcat(histogram, histR, histogram);
    }

    /**
     * @brief Computes a similarity score between two images using their precomputed features.
     *
     * Combines outer-boundary shape, color-histogram, and binary-mask similarities into a single score
     * that expresses overall visual similarity between the two images. Optimized for pictographs with
     * uniform white backgrounds where color is most important and only outer silhouette matters for shape.
     *
     * @param img1 First image metadata and precomputed features (histogram, edges, binary).
     * @param img2 Second image metadata and precomputed features (histogram, edges, binary).
     * @return double Similarity score in the range [0.0, 1.0], where 1.0 indicates identical
     *                images according to the combined feature metrics and 0.0 indicates no similarity.
     */
    double calculateSimilarity(const ImageData& img1, const ImageData& img2) {
        // Method 1: Outer boundary similarity (focuses on silhouette only, not internal details) - 30% weight
        cv::Mat edgeDiff;
        cv::absdiff(img1.edges, img2.edges, edgeDiff);
        double edgeSimilarity = 1.0 - (cv::sum(edgeDiff)[0] / (128.0 * 128.0 * 255.0));

        // Method 2: Histogram comparison (color distribution) - 60% weight (color is important but balanced with shape)
        double histSimilarity = cv::compareHist(img1.histogram, img2.histogram, cv::HISTCMP_CORREL);

        // Method 3: Binary mask similarity (overall shape/area) - 10% weight (uniform backgrounds make this less critical)
        cv::Mat diff;
        cv::absdiff(img1.binary, img2.binary, diff);
        double pixelDiff = cv::sum(diff)[0] / (128.0 * 128.0 * 255.0);
        double structSimilarity = 1.0 - pixelDiff;

        // Combine all methods with weights: Color (60%) + Outer Shape (30%) + Overall Structure (10%)
        double combinedSimilarity = 0.6 * histSimilarity + 0.3 * edgeSimilarity + 0.1 * structSimilarity;

        // No transformation needed - foreground-only histograms already provide excellent discrimination
        // Return the linear combination directly

        // Ensure similarity is in [0, 1] range
        return std::max(0.0, std::min(1.0, combinedSimilarity));
    }

    /**
     * @brief Builds the full pairwise image similarity matrix.
     *
     * Resizes and populates the member `similarityMatrix` to N×N (N = number of loaded images)
     * with similarity scores for every image pair. Diagonal entries are set to 1.0 (self-similarity),
     * symmetric entries are mirrored to avoid redundant computation, and remaining pairs are computed
     * and stored. Progress messages are written to standard output during processing.
     */
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

    /**
     * @brief Writes the current similarity matrix and image filenames to a CSV file.
     *
     * Writes a CSV whose first row is a header of image filenames and whose subsequent
     * rows contain each image filename followed by its similarity scores to every image.
     * On failure to open the file, an error message is printed and no file is written.
     *
     * @param filename Path to the output CSV file.
     */
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

    /**
     * @brief Prints the top-N most similar image pairs found in the current dataset.
     *
     * Prints a ranked list to standard output showing pair rank, similarity score,
     * and the two filenames for the most similar unique image pairs, sorted by
     * descending similarity.
     *
     * @param topN Maximum number of pairs to print; if greater than the number of available unique pairs,
     *             all pairs are printed. Default is 10.
     */
    void printTopSimilarPairs(int topN = 10) {
        struct Pair {
            size_t i, j;
            double similarity;

            /**
             * @brief Define ordering between two Pair objects based on similarity for descending sort.
             *
             * @param other The Pair to compare against.
             * @return true if this object's similarity is greater than other's similarity, false otherwise.
             */
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

    /**
     * @brief Computes and prints aggregate similarity statistics for all unique image pairs.
     *
     * Iterates over the upper triangle of the similarity matrix (pairs where i < j),
     * accumulates sum, minimum, maximum, and count, then prints the number of pairs,
     * the average similarity, the minimum similarity, and the maximum similarity to stdout.
     */
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
    /**
     * @brief Orchestrates image similarity analysis for a directory and writes results to a CSV.
     *
     * Loads images from the given directory, computes the pairwise similarity matrix,
     * exports the matrix to a CSV file, and prints aggregate statistics and the top similar pairs.
     *
     * @param directory Path to the root directory containing images to analyze.
     * @param outputFile Destination CSV filename for the similarity matrix (default: "similarity_matrix.csv").
     */
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

/**
 * @brief Program entry point that runs the image similarity analysis and writes results to a CSV file.
 *
 * The program prints a header, optionally reads an input directory and output filename from command-line
 * arguments, constructs an ImageSimilarityAnalyzer, and runs the analysis which loads images, computes
 * pairwise similarities, exports the similarity matrix to the specified CSV, and prints summary statistics.
 *
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments where:
 *             - argv[1] (optional) specifies the input image directory (default: "Data/Images"),
 *             - argv[2] (optional) specifies the output CSV filename (default: "similarity_matrix.csv").
 * @return int `0` on successful completion, `1` if an exception is thrown during execution.
 */
int main(int argc, char** argv) {
    std::cout << "=== Image Similarity Analyzer ===" << std::endl;
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