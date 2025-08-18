"""
Direction generation utilities for simulation data collection.

This module provides functions for generating direction patterns used in eye width simulations.
"""

import numpy as np
import warnings
from typing import List


def get_valid_block_sizes(n_lines: int) -> List[int]:
    """
    Find divisors of n_lines that result in an even number of blocks.
    
    Args:
        n_lines: Number of lines in the simulation
        
    Returns:
        List of valid block sizes that divide n_lines evenly and result in even number of blocks
        Note: Block size 1 is never returned as it would result in per-line assignment
    """
    divisors = []
    for i in range(2, int(n_lines**0.5) + 1):  # Start from 2, not 1
        if n_lines % i == 0:
            n_blocks = n_lines // i
            if n_blocks % 2 == 0:  # Only include if it results in even number of blocks
                divisors.append(i)
            if i * i != n_lines:
                j = n_lines // i
                n_blocks_j = n_lines // j
                if n_blocks_j % 2 == 0:  # Only include if it results in even number of blocks
                    divisors.append(j)
    
    # If no valid divisors found, we need to handle this case
    # For odd n_lines, we can't have even number of blocks, so we'll use the largest divisor
    if not divisors:
        # Find the largest divisor that's not n_lines itself and not 1
        largest_divisor = 2  # Start with 2 as minimum
        for i in range(2, n_lines):
            if n_lines % i == 0:
                largest_divisor = i
        divisors.append(largest_divisor)
    
    return divisors

def generate_directions(n_lines: int, enable_direction: bool = True, block_size: int = None) -> np.ndarray:
    """
    Generate direction pattern for simulation.
    
    Args:
        n_lines: Number of lines in the simulation
        enable_direction: Whether to use random directions (True) or all ones (False)
        block_size: Optional fixed block size. If provided and it's a valid divisor of n_lines, it will be used.
                    Otherwise, a random valid block size is chosen.
        
    Returns:
        numpy.ndarray: Array of direction values (0 or 1) for each line
        
    Notes:
        When enable_direction is True, generates a block-wise pattern where:
        - Lines are grouped into blocks of equal size
        - Each block is assigned either 0 or 1
        - The pattern is shuffled for randomness
        - This approach provides better performance than random per-line assignment
    """
    if not enable_direction:
        return np.ones(n_lines, dtype=int)
    
    # Determine block size
    if block_size is not None:
        if n_lines % block_size != 0:
            warnings.warn(f"block_size {block_size} is not a divisor of n_lines {n_lines}. A random valid block size will be used instead.")
            block_size = None
    
    if block_size is None:
        valid_block_sizes = get_valid_block_sizes(n_lines)
        block_size = np.random.choice(valid_block_sizes)
        
    n_blocks = n_lines // block_size
    
    # Create equal number of 0 and 1 blocks
    blocks = [0] * (n_blocks // 2) + [1] * (n_blocks // 2)
    if n_blocks % 2 != 0:
        blocks.append(np.random.randint(0, 2))
    
    np.random.shuffle(blocks)
    directions = np.repeat(blocks, block_size)
    
    # Truncate if needed
    if len(directions) > n_lines:
        directions = directions[:n_lines]
        
    return directions


if __name__ == "__main__":
    """Test the direction generation utilities when run directly"""
    import sys
    
    def test_direction_generation():
        """Test the direction generation function with various inputs"""
        print("Testing direction generation utilities...")
        print("=" * 50)
        
        # Test get_valid_block_sizes function
        print("Testing get_valid_block_sizes function:")
        test_cases = [4, 8, 12, 16, 24, 48, 96]
        
        for n_lines in test_cases:
            block_sizes = get_valid_block_sizes(n_lines)
            print(f"  n_lines={n_lines}: valid block sizes = {block_sizes}")
            
            # Verify that each block size divides n_lines evenly and is not 1
            for block_size in block_sizes:
                assert n_lines % block_size == 0, f"Block size {block_size} does not divide {n_lines}"
                assert block_size > 1, f"Block size {block_size} should not be 1"
                n_blocks = n_lines // block_size
                assert n_blocks % 2 == 0, f"Block size {block_size} results in odd number of blocks: {n_blocks}"
        
        print("  ✓ All block size tests passed!")
        
        # Test generate_directions function
        print("\nTesting generate_directions function:")
        
        # Test with enable_direction=False
        directions = generate_directions(48, enable_direction=False)
        assert np.all(directions == 1), f"Expected all ones, got: {directions}"
        print(f"  enable_direction=False: {directions[:10]}... (all ones)")
        
        # Test with enable_direction=True
        directions = generate_directions(48, enable_direction=True)
        assert len(directions) == 48, f"Expected length 48, got: {len(directions)}"
        assert np.all(np.isin(directions, [0, 1])), f"Expected only 0s and 1s, got: {directions}"
        print(f"  enable_direction=True: {directions[:10]}... (mixed 0s and 1s)")
        
        # Test with different line counts
        for n_lines in [4, 8, 12, 16, 24]:
            directions = generate_directions(n_lines, enable_direction=True)
            assert len(directions) == n_lines, f"Expected length {n_lines}, got: {len(directions)}"
            print(f"  n_lines={n_lines}: length={len(directions)}, unique_values={np.unique(directions)}")
        
        print("  ✓ All direction generation tests passed!")
        
        # Test consistency
        print("\nTesting consistency:")
        
        # Test that the same seed produces the same result
        np.random.seed(42)
        directions1 = generate_directions(48, enable_direction=True)
        
        np.random.seed(42)
        directions2 = generate_directions(48, enable_direction=True)
        
        assert np.array_equal(directions1, directions2), "Results should be identical with same seed"
        print(f"  ✓ Consistency test passed: {directions1[:10]}...")
        
        # Test that different seeds produce different results
        np.random.seed(42)
        directions1 = generate_directions(48, enable_direction=True)
        
        np.random.seed(43)
        directions2 = generate_directions(48, enable_direction=True)
        
        assert not np.array_equal(directions1, directions2), "Results should be different with different seeds"
        print(f"  ✓ Randomness test passed: different results with different seeds")
        
        print("\n" + "=" * 50)
        print("All tests passed! Direction generation utilities are working correctly.")
        return True
    
    def demo_direction_generation():
        """Demonstrate direction generation with different parameters"""
        print("\nDemonstrating direction generation:")
        print("-" * 30)
        
        # Demo with different line counts
        for n_lines in [8, 16, 32]:
            print(f"\n{n_lines} lines:")
            
            # All ones (disable direction)
            directions = generate_directions(n_lines, enable_direction=False)
            print(f"  All ones:     {directions}")
            
            # Random directions
            directions = generate_directions(n_lines, enable_direction=True)
            print(f"  Random:       {directions}")
            
            # Show block structure
            block_sizes = get_valid_block_sizes(n_lines)
            print(f"  Block sizes:  {block_sizes}")
    
    try:
        # Run tests
        test_direction_generation()
        
        # Run demo
        demo_direction_generation()
        
        print("\nAll tests and demonstrations completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)