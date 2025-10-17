#!/usr/bin/env python3
"""
CNN-based Image Similarity Analyzer using ResNet50 embeddings.

Uses a pre-trained ResNet50 model to extract feature embeddings from images
and computes pairwise cosine similarity. Optimized for pictographs.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import argparse


class CNNSimilarityAnalyzer:
    """Extract CNN embeddings and compute image similarity."""

    def __init__(self, model_name='resnet50'):
        """
        Initialize the analyzer with a pre-trained model.

        Args:
            model_name: Name of the model to use ('resnet50', 'resnet101', 'efficientnet_b0')
        """
        print(f"Loading {model_name} model...")

        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove the final classification layer to get embeddings
        if 'resnet' in model_name:
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif 'efficientnet' in model_name:
            self.model.classifier = torch.nn.Identity()

        self.model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_images(self, directory):
        """
        Load all images from directory recursively.

        Args:
            directory: Path to directory containing images

        Returns:
            List of (filename, filepath) tuples
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = []

        directory = Path(directory)
        for ext in image_extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))

        # Sort for consistent ordering
        image_files = sorted(image_files)

        return [(f.name, str(f)) for f in image_files]

    def extract_embedding(self, image_path):
        """
        Extract embedding vector from a single image.

        Args:
            image_path: Path to image file

        Returns:
            numpy array of embedding vector
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model(image_tensor)

            # Convert to numpy and flatten
            embedding = embedding.cpu().numpy().flatten()

            return embedding

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_all_embeddings(self, image_files):
        """
        Extract embeddings for all images.

        Args:
            image_files: List of (filename, filepath) tuples

        Returns:
            Dictionary mapping filenames to embeddings
        """
        embeddings = {}

        print(f"\nExtracting embeddings for {len(image_files)} images...")
        for filename, filepath in tqdm(image_files):
            emb = self.extract_embedding(filepath)
            if emb is not None:
                embeddings[filename] = emb

        print(f"Successfully extracted {len(embeddings)} embeddings")
        return embeddings

    def compute_similarity_matrix(self, embeddings):
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: Dictionary of filename -> embedding

        Returns:
            filenames (list), similarity_matrix (numpy array)
        """
        filenames = sorted(embeddings.keys())
        embedding_matrix = np.array([embeddings[f] for f in filenames])

        print("\nComputing similarity matrix...")
        similarity_matrix = cosine_similarity(embedding_matrix)

        return filenames, similarity_matrix

    def export_to_csv(self, filenames, similarity_matrix, output_file):
        """
        Export similarity matrix to CSV file.

        Args:
            filenames: List of image filenames
            similarity_matrix: NxN similarity matrix
            output_file: Output CSV filename
        """
        print(f"\nExporting similarity matrix to {output_file}...")

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['Image'] + filenames)

            # Write data rows
            for i, filename in enumerate(filenames):
                row = [filename] + [f"{similarity_matrix[i][j]:.4f}"
                                   for j in range(len(filenames))]
                writer.writerow(row)

        print(f"Similarity matrix exported successfully!")

    def print_statistics(self, similarity_matrix):
        """Print similarity statistics."""
        # Get upper triangle (excluding diagonal)
        n = len(similarity_matrix)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i][j])

        similarities = np.array(similarities)

        print("\n=== Similarity Statistics ===")
        print(f"Number of image pairs: {len(similarities)}")
        print(f"Average similarity: {similarities.mean():.4f}")
        print(f"Minimum similarity: {similarities.min():.4f}")
        print(f"Maximum similarity: {similarities.max():.4f}")
        print(f"Std deviation: {similarities.std():.4f}")

    def print_top_similar_pairs(self, filenames, similarity_matrix, top_n=15):
        """Print top N most similar image pairs."""
        n = len(filenames)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((similarity_matrix[i][j], filenames[i], filenames[j]))

        # Sort by similarity (descending)
        pairs.sort(reverse=True)

        print(f"\n=== Top {top_n} Most Similar Image Pairs ===")
        for k, (sim, img1, img2) in enumerate(pairs[:top_n], 1):
            print(f"{k:3d}. {sim:.4f} - {img1} <-> {img2}")

    def analyze(self, directory, output_file='similarity_matrix_cnn.csv'):
        """
        Run complete similarity analysis pipeline.

        Args:
            directory: Input directory containing images
            output_file: Output CSV filename
        """
        print("=== CNN Image Similarity Analyzer ===")
        print("=" * 40)

        # Load images
        image_files = self.load_images(directory)
        print(f"\nFound {len(image_files)} images in {directory}")

        if not image_files:
            print("Error: No images found!")
            return

        # Extract embeddings
        embeddings = self.extract_all_embeddings(image_files)

        if not embeddings:
            print("Error: No embeddings extracted!")
            return

        # Compute similarity matrix
        filenames, similarity_matrix = self.compute_similarity_matrix(embeddings)

        # Export to CSV
        self.export_to_csv(filenames, similarity_matrix, output_file)

        # Print statistics
        self.print_statistics(similarity_matrix)
        self.print_top_similar_pairs(filenames, similarity_matrix)

        print("\n=== Analysis Complete ===")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CNN-based image similarity analyzer using pre-trained models'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='Data/Images',
        help='Input directory containing images (default: Data/Images)'
    )
    parser.add_argument(
        'output',
        nargs='?',
        default='similarity_matrix_cnn.csv',
        help='Output CSV filename (default: similarity_matrix_cnn.csv)'
    )
    parser.add_argument(
        '--model',
        choices=['resnet50', 'resnet101', 'efficientnet_b0'],
        default='resnet50',
        help='Model to use for embeddings (default: resnet50)'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = CNNSimilarityAnalyzer(model_name=args.model)
    analyzer.analyze(args.directory, args.output)


if __name__ == '__main__':
    main()
