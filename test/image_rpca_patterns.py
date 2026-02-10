import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from rpca import RPCA
from scipy.ndimage import gaussian_filter
import os

class ImageRPCAPatternExtractor:
    """
    Class for extracting patterns from images using Robust Principal Component Analysis (RPCA).
    
    RPCA decomposes an image matrix X into:
    - L (Low-rank component): Main patterns, textures, and structural information
    - S (Sparse component): Noise, outliers, and sparse corruptions
    """
    
    def __init__(self, lambda_value=None, mu_value=None, tolerance=1e-6, max_iterations=1000):
        """
        Initialize the RPCA pattern extractor.
        
        Args:
            lambda_value: Regularization parameter controlling sparsity of S
            mu_value: Step size parameter for optimization
            tolerance: Convergence criterion
            max_iterations: Maximum number of iterations
        """
        self.lambda_value = lambda_value
        self.mu_value = mu_value
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def load_image(self, image_path, grayscale=True):
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            grayscale: Whether to convert to grayscale
            
        Returns:
            numpy.ndarray: Preprocessed image matrix
        """
        if isinstance(image_path, str):
            # Load from file
            image = Image.open(image_path)
        else:
            # Assume it's already a numpy array or PIL image
            image = image_path
            
        if grayscale and hasattr(image, 'convert'):
            image = image.convert('L')
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float64)
        if image_array.max() > 1:
            image_array = image_array / 255.0
            
        return image_array
    
    def create_synthetic_image_with_patterns(self, size=(128, 128)):
        """
        Create a synthetic image with known patterns for testing.
        
        Args:
            size: Tuple (height, width) of the image
            
        Returns:
            tuple: (original_image, clean_patterns, noise_corruption)
        """
        h, w = size
        
        # Create clean patterns (low-rank component)
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Multiple pattern components
        pattern1 = 0.3 * np.sin(X) * np.cos(Y)  # Sinusoidal pattern
        pattern2 = 0.2 * np.exp(-((X-2*np.pi)**2 + (Y-2*np.pi)**2) / 4)  # Gaussian blob
        pattern3 = 0.1 * np.sin(3*X) * np.sin(3*Y)  # High-frequency pattern
        
        clean_patterns = pattern1 + pattern2 + pattern3
        
        # Add sparse corruption (sparse component)
        sparse_corruption = np.zeros_like(clean_patterns)
        num_corruptions = int(0.1 * h * w)  # 10% of pixels corrupted
        corruption_positions = np.random.choice(h*w, num_corruptions, replace=False)
        corruption_coords = np.unravel_index(corruption_positions, (h, w))
        sparse_corruption[corruption_coords] = np.random.uniform(-0.5, 0.5, num_corruptions)
        
        # Add some Gaussian noise
        gaussian_noise = 0.05 * np.random.randn(h, w)
        
        # Combine all components
        noisy_image = clean_patterns + sparse_corruption + gaussian_noise
        
        return noisy_image, clean_patterns, sparse_corruption
    
    def extract_patterns(self, image):
        """
        Extract patterns from an image using RPCA.
        
        Args:
            image: 2D numpy array representing the image
            
        Returns:
            tuple: (low_rank_patterns, sparse_component, reconstruction_error)
        """
        # Ensure image is 2D
        if len(image.shape) > 2:
            raise ValueError("Image must be 2D (grayscale). Use grayscale=True in load_image()")
        
        # Apply RPCA decomposition
        L, S = RPCA(
            image, 
            lamb=self.lambda_value, 
            mu=self.mu_value, 
            tolerance=self.tolerance, 
            max_iteration=self.max_iterations
        )
        
        # Calculate reconstruction error
        reconstruction_error = np.linalg.norm(image - L - S, 'fro')
        
        return L, S, reconstruction_error
    
    def extract_patterns_from_video_frames(self, video_frames):
        """
        Extract patterns from a sequence of video frames.
        Each column of the matrix represents a vectorized frame.
        
        Args:
            video_frames: numpy array of shape (height, width, num_frames)
            
        Returns:
            tuple: (background_model, foreground_objects, reconstruction_error)
        """
        h, w, num_frames = video_frames.shape
        
        # Reshape frames into a matrix where each column is a vectorized frame
        frame_matrix = video_frames.reshape(h*w, num_frames)
        
        # Apply RPCA
        L, S = RPCA(
            frame_matrix,
            lamb=self.lambda_value,
            mu=self.mu_value,
            tolerance=self.tolerance,
            max_iteration=self.max_iterations
        )
        
        # Reshape back to original frame dimensions
        background_model = L.reshape(h, w, num_frames)
        foreground_objects = S.reshape(h, w, num_frames)
        
        reconstruction_error = np.linalg.norm(frame_matrix - L - S, 'fro')
        
        return background_model, foreground_objects, reconstruction_error
    
    def visualize_decomposition(self, original, L, S, save_path=None):
        """
        Visualize the RPCA decomposition results.
        
        Args:
            original: Original image
            L: Low-rank component (patterns)
            S: Sparse component
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        im1 = axes[0, 0].imshow(original, cmap='gray', aspect='equal')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Low-rank component (patterns)
        im2 = axes[0, 1].imshow(L, cmap='gray', aspect='equal')
        axes[0, 1].set_title('Low-rank Component (L)\nMain Patterns')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Sparse component
        im3 = axes[0, 2].imshow(S, cmap='gray', aspect='equal')
        axes[0, 2].set_title('Sparse Component (S)\nNoise & Outliers')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Reconstruction
        reconstruction = L + S
        im4 = axes[1, 0].imshow(reconstruction, cmap='gray', aspect='equal')
        axes[1, 0].set_title('Reconstruction (L + S)')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Reconstruction error
        error = original - reconstruction
        im5 = axes[1, 1].imshow(error, cmap='RdBu', aspect='equal')
        axes[1, 1].set_title('Reconstruction Error')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Histogram comparison
        axes[1, 2].hist(original.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[1, 2].hist(L.flatten(), bins=50, alpha=0.7, label='Low-rank (L)', density=True)
        axes[1, 2].hist(S.flatten(), bins=50, alpha=0.7, label='Sparse (S)', density=True)
        axes[1, 2].set_title('Pixel Value Distributions')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Pixel Value')
        axes[1, 2].set_ylabel('Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_pattern_characteristics(self, L, S):
        """
        Analyze characteristics of extracted patterns.
        
        Args:
            L: Low-rank component
            S: Sparse component
            
        Returns:
            dict: Analysis results
        """
        analysis = {
            'L_rank': np.linalg.matrix_rank(L),
            'L_energy': np.sum(L**2),
            'S_sparsity': np.count_nonzero(S) / S.size,
            'S_energy': np.sum(S**2),
            'L_mean': np.mean(L),
            'L_std': np.std(L),
            'S_mean': np.mean(S),
            'S_std': np.std(S),
            'L_entropy': self._calculate_entropy(L),
            'S_entropy': self._calculate_entropy(S)
        }
        
        return analysis
    
    def _calculate_entropy(self, image):
        """Calculate Shannon entropy of an image."""
        # Normalize to [0, 255] and convert to integers
        img_int = ((image - image.min()) / (image.max() - image.min()) * 255).astype(int)
        
        # Calculate histogram
        hist, _ = np.histogram(img_int.flatten(), bins=256, range=(0, 256))
        
        # Calculate probabilities
        hist_norm = hist / hist.sum()
        
        # Calculate entropy (avoiding log(0))
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        return entropy


def demonstrate_image_pattern_extraction():
    """
    Demonstrate RPCA pattern extraction on various types of images.
    """
    # Initialize the pattern extractor
    extractor = ImageRPCAPatternExtractor(
        lambda_value=None,  # Will use default: 1/sqrt(min(m,n))
        mu_value=None,      # Will use default: 10*lambda
        tolerance=1e-6,
        max_iterations=1000
    )
    
    print("=== RPCA Image Pattern Extraction Demo ===\n")
    
    # 1. Synthetic image with known patterns
    print("1. Creating synthetic image with known patterns...")
    synthetic_image, true_patterns, true_corruption = extractor.create_synthetic_image_with_patterns()
    
    print("   Extracting patterns using RPCA...")
    L_synthetic, S_synthetic, error_synthetic = extractor.extract_patterns(synthetic_image)
    
    print(f"   Reconstruction error: {error_synthetic:.6f}")
    
    # Visualize results
    extractor.visualize_decomposition(
        synthetic_image, L_synthetic, S_synthetic,
        save_path="synthetic_rpca_decomposition.png"
    )
    
    # Compare with ground truth
    print("   Comparing with ground truth...")
    pattern_similarity = np.corrcoef(true_patterns.flatten(), L_synthetic.flatten())[0, 1]
    corruption_similarity = np.corrcoef(true_corruption.flatten(), S_synthetic.flatten())[0, 1]
    print(f"   Pattern correlation with ground truth: {pattern_similarity:.4f}")
    print(f"   Corruption correlation with ground truth: {corruption_similarity:.4f}")
    
    # 2. Analyze pattern characteristics
    print("\n2. Analyzing extracted pattern characteristics...")
    analysis = extractor.analyze_pattern_characteristics(L_synthetic, S_synthetic)
    
    print("   Low-rank component (L) analysis:")
    print(f"     - Matrix rank: {analysis['L_rank']}")
    print(f"     - Energy: {analysis['L_energy']:.4f}")
    print(f"     - Mean: {analysis['L_mean']:.4f}")
    print(f"     - Standard deviation: {analysis['L_std']:.4f}")
    print(f"     - Entropy: {analysis['L_entropy']:.4f} bits")
    
    print("   Sparse component (S) analysis:")
    print(f"     - Sparsity: {analysis['S_sparsity']:.4f} ({analysis['S_sparsity']*100:.1f}% non-zero)")
    print(f"     - Energy: {analysis['S_energy']:.4f}")
    print(f"     - Mean: {analysis['S_mean']:.4f}")
    print(f"     - Standard deviation: {analysis['S_std']:.4f}")
    print(f"     - Entropy: {analysis['S_entropy']:.4f} bits")
    
    # 3. Demonstrate video frame processing (background subtraction)
    print("\n3. Demonstrating video frame processing (background subtraction)...")
    
    # Create synthetic video frames
    num_frames = 20
    h, w = 64, 64
    
    # Static background pattern
    x = np.linspace(0, 2*np.pi, w)
    y = np.linspace(0, 2*np.pi, h)
    X, Y = np.meshgrid(x, y)
    background = 0.5 + 0.3 * np.sin(X) * np.cos(Y)
    
    # Create video frames with moving object
    video_frames = np.zeros((h, w, num_frames))
    for i in range(num_frames):
        frame = background.copy()
        
        # Add moving circular object
        center_x = int(w/4 + (w/2) * i / num_frames)
        center_y = int(h/4 + (h/2) * i / num_frames)
        
        # Create circular mask
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - center_x)**2 + (yy - center_y)**2 <= 100
        frame[mask] += 0.5
        
        video_frames[:, :, i] = frame
    
    # Extract background and foreground
    background_model, foreground_objects, video_error = extractor.extract_patterns_from_video_frames(video_frames)
    
    print(f"   Video reconstruction error: {video_error:.6f}")
    
    # Visualize a few frames
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    frame_indices = [0, 5, 10, 15, 19]
    
    for i, frame_idx in enumerate(frame_indices):
        # Original frame
        axes[0, i].imshow(video_frames[:, :, frame_idx], cmap='gray')
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Background model
        axes[1, i].imshow(background_model[:, :, frame_idx], cmap='gray')
        axes[1, i].set_title('Background')
        axes[1, i].axis('off')
        
        # Foreground objects
        axes[2, i].imshow(foreground_objects[:, :, frame_idx], cmap='gray')
        axes[2, i].set_title('Foreground')
        axes[2, i].axis('off')
    
    plt.suptitle('Video Background Subtraction using RPCA')
    plt.tight_layout()
    plt.savefig("video_background_subtraction.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Demo completed! ===")
    print("Generated visualizations:")
    print("- synthetic_rpca_decomposition.png")
    print("- video_background_subtraction.png")


if __name__ == "__main__":
    demonstrate_image_pattern_extraction()














