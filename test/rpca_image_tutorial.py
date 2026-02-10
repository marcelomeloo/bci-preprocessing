"""
Tutorial: Extracting Patterns from Images using RPCA

This tutorial demonstrates how to use your existing RPCA implementation
to extract patterns from images. The key insight is that RPCA decomposes
any matrix X into:

X = L + S

Where:
- L (Low-rank): Contains the main patterns and structure
- S (Sparse): Contains noise, outliers, and corruptions

For images, this separation allows us to:
1. Extract dominant visual patterns and textures (L)
2. Remove noise and sparse corruptions (S)
3. Perform background subtraction in videos
4. Denoise images while preserving structure
"""

import numpy as np
import matplotlib.pyplot as plt
from rpca import RPCA

def basic_image_rpca_example():
    """
    Basic example showing how to apply RPCA to a simple image.
    """
    print("=== Basic Image RPCA Example ===")
    
    # Create a simple test image with patterns
    size = 64
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Create clean patterns (what we want to extract)
    clean_image = (
        0.5 +                                    # Base level
        0.3 * np.sin(X) * np.cos(Y) +           # Main pattern
        0.1 * np.sin(3*X) * np.sin(3*Y)         # Fine detail
    )
    
    # Add sparse corruption (what we want to remove)
    corrupted_image = clean_image.copy()
    
    # Add random sparse noise to 15% of pixels
    num_corrupted = int(0.15 * size * size)
    corruption_indices = np.random.choice(size*size, num_corrupted, replace=False)
    corruption_coords = np.unravel_index(corruption_indices, (size, size))
    corrupted_image[corruption_coords] += np.random.uniform(-0.5, 0.5, num_corrupted)
    
    print(f"Image size: {size}x{size}")
    print(f"Corrupted pixels: {num_corrupted} ({100*num_corrupted/(size*size):.1f}%)")
    
    # Apply RPCA
    print("Applying RPCA decomposition...")
    L, S = RPCA(
        corrupted_image,
        lamb=None,        # Use default: 1/sqrt(min(m,n))
        mu=None,          # Use default: 10*lambda
        tolerance=1e-6,
        max_iteration=500
    )
    
    # Calculate metrics
    reconstruction_error = np.linalg.norm(corrupted_image - L - S, 'fro')
    pattern_recovery_error = np.linalg.norm(clean_image - L, 'fro')
    
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Pattern recovery error: {pattern_recovery_error:.6f}")
    print(f"Low-rank component rank: {np.linalg.matrix_rank(L)}")
    print(f"Sparse component sparsity: {np.count_nonzero(S)}/{S.size} ({100*np.count_nonzero(S)/S.size:.1f}%)")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Top row: inputs and clean patterns
    im1 = axes[0, 0].imshow(clean_image, cmap='viridis')
    axes[0, 0].set_title('Original Clean Image')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(corrupted_image, cmap='viridis')
    axes[0, 1].set_title('Corrupted Image (Input)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Show corruption
    corruption_only = corrupted_image - clean_image
    im3 = axes[0, 2].imshow(corruption_only, cmap='RdBu')
    axes[0, 2].set_title('True Corruption')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Bottom row: RPCA results
    im4 = axes[1, 0].imshow(L, cmap='viridis')
    axes[1, 0].set_title('Low-rank Component (L)\nExtracted Patterns')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(S, cmap='RdBu')
    axes[1, 1].set_title('Sparse Component (S)\nExtracted Corruption')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    # Error visualization
    error_map = np.abs(clean_image - L)
    im6 = axes[1, 2].imshow(error_map, cmap='Reds')
    axes[1, 2].set_title('Pattern Recovery Error')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('basic_rpca_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return L, S, clean_image, corrupted_image


def texture_pattern_extraction():
    """
    Example showing RPCA for texture pattern extraction.
    """
    print("\n=== Texture Pattern Extraction ===")
    
    # Create an image with multiple texture patterns
    size = 128
    texture_image = np.zeros((size, size))
    
    # Region 1: Vertical stripes
    for i in range(0, size//2, 4):
        texture_image[:size//2, i:i+2] = 0.7
    
    # Region 2: Horizontal stripes  
    for i in range(0, size//2, 6):
        texture_image[i:i+3, size//2:] = 0.8
    
    # Region 3: Checkerboard pattern
    for i in range(size//2, size, 8):
        for j in range(0, size//2, 8):
            texture_image[i:i+4, j:j+4] = 0.6
            texture_image[i+4:i+8, j+4:j+8] = 0.6
    
    # Region 4: Circular pattern
    center_x, center_y = 3*size//4, 3*size//4
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    texture_image[size//2:, size//2:] = 0.5 + 0.3 * np.sin(distance[size//2:, size//2:] * 0.5)
    
    # Add noise and sparse outliers
    noisy_texture = texture_image + 0.1 * np.random.randn(size, size)
    
    # Add sparse corruptions (simulating dirt, scratches, etc.)
    num_corruptions = int(0.05 * size * size)
    corruption_indices = np.random.choice(size*size, num_corruptions, replace=False)
    corruption_coords = np.unravel_index(corruption_indices, (size, size))
    noisy_texture[corruption_coords] = np.random.uniform(0, 1, num_corruptions)
    
    print(f"Texture image size: {size}x{size}")
    print(f"Added {num_corruptions} sparse corruptions")
    
    # Apply RPCA with parameters tuned for texture extraction
    lambda_val = 1 / np.sqrt(max(size, size))  # Standard choice
    print("Applying RPCA for texture pattern extraction...")
    
    L_texture, S_texture = RPCA(
        noisy_texture,
        lamb=lambda_val,
        mu=10 * lambda_val,
        tolerance=1e-7,
        max_iteration=800
    )
    
    # Analyze results
    print(f"Original texture rank: {np.linalg.matrix_rank(texture_image)}")
    print(f"Extracted pattern rank: {np.linalg.matrix_rank(L_texture)}")
    print(f"Rank reduction: {np.linalg.matrix_rank(texture_image)} → {np.linalg.matrix_rank(L_texture)}")
    
    # Calculate pattern preservation
    pattern_correlation = np.corrcoef(texture_image.flatten(), L_texture.flatten())[0, 1]
    print(f"Pattern correlation: {pattern_correlation:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    im1 = axes[0, 0].imshow(texture_image, cmap='gray')
    axes[0, 0].set_title('Original Texture Patterns')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(noisy_texture, cmap='gray')
    axes[0, 1].set_title('Noisy + Corrupted')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(L_texture, cmap='gray')
    axes[0, 2].set_title('Extracted Patterns (L)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    im4 = axes[1, 0].imshow(S_texture, cmap='RdBu')
    axes[1, 0].set_title('Extracted Noise/Corruption (S)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Show difference
    pattern_diff = np.abs(texture_image - L_texture)
    im5 = axes[1, 1].imshow(pattern_diff, cmap='Reds')
    axes[1, 1].set_title('Pattern Extraction Error')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Histogram comparison
    axes[1, 2].hist(texture_image.flatten(), bins=30, alpha=0.7, label='Original', density=True)
    axes[1, 2].hist(L_texture.flatten(), bins=30, alpha=0.7, label='Extracted (L)', density=True)
    axes[1, 2].set_title('Pixel Value Distributions')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('texture_rpca_extraction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return L_texture, S_texture


def parameter_sensitivity_analysis():
    """
    Analyze how RPCA parameters affect pattern extraction.
    """
    print("\n=== Parameter Sensitivity Analysis ===")
    
    # Create test image
    size = 64
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    test_image = 0.5 + 0.4 * np.sin(X) * np.cos(Y)
    
    # Add corruption
    corrupted = test_image.copy()
    num_corrupted = int(0.1 * size * size)
    corruption_indices = np.random.choice(size*size, num_corrupted, replace=False)
    corruption_coords = np.unravel_index(corruption_indices, (size, size))
    corrupted[corruption_coords] += 0.8
    
    # Test different lambda values
    base_lambda = 1 / np.sqrt(size)
    lambda_values = [base_lambda * factor for factor in [0.1, 0.5, 1.0, 2.0, 5.0]]
    
    print("Testing different lambda values...")
    results = []
    
    fig, axes = plt.subplots(2, len(lambda_values), figsize=(15, 6))
    
    for i, lamb in enumerate(lambda_values):
        print(f"  Lambda = {lamb:.4f} ({lamb/base_lambda:.1f}x base)")
        
        L, S = RPCA(corrupted, lamb=lamb, mu=10*lamb, tolerance=1e-6, max_iteration=300)
        
        # Calculate metrics
        pattern_error = np.linalg.norm(test_image - L, 'fro')
        rank = np.linalg.matrix_rank(L)
        sparsity = np.count_nonzero(S) / S.size
        
        results.append({
            'lambda': lamb,
            'lambda_factor': lamb/base_lambda,
            'pattern_error': pattern_error,
            'rank': rank,
            'sparsity': sparsity
        })
        
        print(f"    Pattern error: {pattern_error:.4f}")
        print(f"    Rank: {rank}")
        print(f"    Sparsity: {sparsity:.3f}")
        
        # Visualize
        im1 = axes[0, i].imshow(L, cmap='viridis')
        axes[0, i].set_title(f'L (λ={lamb/base_lambda:.1f}x)\nRank: {rank}')
        axes[0, i].axis('off')
        
        im2 = axes[1, i].imshow(S, cmap='RdBu')
        axes[1, i].set_title(f'S\nSparsity: {sparsity:.3f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot parameter effects
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    lambda_factors = [r['lambda_factor'] for r in results]
    pattern_errors = [r['pattern_error'] for r in results]
    ranks = [r['rank'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    
    axes[0].plot(lambda_factors, pattern_errors, 'o-')
    axes[0].set_xlabel('Lambda Factor (relative to 1/√n)')
    axes[0].set_ylabel('Pattern Recovery Error')
    axes[0].set_title('Effect of λ on Pattern Recovery')
    axes[0].grid(True)
    
    axes[1].plot(lambda_factors, ranks, 'o-')
    axes[1].set_xlabel('Lambda Factor')
    axes[1].set_ylabel('Rank of L')
    axes[1].set_title('Effect of λ on Low-rank Component')
    axes[1].grid(True)
    
    axes[2].plot(lambda_factors, sparsities, 'o-')
    axes[2].set_xlabel('Lambda Factor')
    axes[2].set_ylabel('Sparsity of S')
    axes[2].set_title('Effect of λ on Sparse Component')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_effects.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Find optimal lambda
    optimal_idx = np.argmin(pattern_errors)
    optimal_result = results[optimal_idx]
    print(f"\nOptimal lambda factor: {optimal_result['lambda_factor']:.1f}x")
    print(f"Best pattern error: {optimal_result['pattern_error']:.4f}")
    
    return results


def main():
    """
    Run all RPCA image pattern extraction examples.
    """
    print("RPCA Image Pattern Extraction Tutorial")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run examples
    L_basic, S_basic, clean, corrupted = basic_image_rpca_example()
    L_texture, S_texture = texture_pattern_extraction()
    param_results = parameter_sensitivity_analysis()
    
    print("\n" + "=" * 50)
    print("Tutorial completed!")
    print("\nKey takeaways:")
    print("1. RPCA separates images into low-rank patterns (L) and sparse corruptions (S)")
    print("2. L captures the main visual structure and patterns")
    print("3. S captures noise, outliers, and sparse corruptions")
    print("4. Parameter λ controls the trade-off between rank and sparsity")
    print("5. Smaller λ → more detailed patterns, less sparse corruption removal")
    print("6. Larger λ → simpler patterns, more aggressive corruption removal")
    print("\nGenerated files:")
    print("- basic_rpca_example.png")
    print("- texture_rpca_extraction.png") 
    print("- parameter_sensitivity.png")
    print("- parameter_effects.png")


if __name__ == "__main__":
    main()














