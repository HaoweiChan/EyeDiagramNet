import scipy
import numpy as np

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators that do nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ===============================================
# LINEAR ALGEBRA UTILITIES
# ===============================================

@jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def rsolve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve x @ A = B using NumPy with vectorized operations.
    
    This is equivalent to B @ np.linalg.inv(A) but avoids calculating the inverse
    and should be numerically more accurate.
    
    Args:
        A: Left-hand side matrix
        B: Right-hand side matrix
        
    Returns:
        Solution matrix x
    """
    # Handle 3D arrays using vectorized operations
    if A.ndim == 3:
        # Vectorized solve for all frequency points at once
        # Transpose and conjugate for the solve operation
        A_t_conj = np.transpose(A, (0, 2, 1)).conj()
        B_t_conj = np.transpose(B, (0, 2, 1)).conj()
        
        # Solve for each frequency point
        result = np.zeros_like(B, dtype=A.dtype)
        for i in range(A.shape[0]):
            result[i] = np.linalg.solve(A_t_conj[i], B_t_conj[i]).T.conj()
        return result
    else:
        return np.linalg.solve(A.T.conj(), B.T.conj()).T.conj()

def nudge_eig(mat: np.ndarray, cond: float = 1e-9, min_eig: float = 1e-12) -> np.ndarray:
    """Nudge eigenvalues to avoid singularities in matrix equations.
    
    Eigenvalues with absolute value smaller than max(cond * max(eigenvalue), min_eig)
    are nudged to that value.
    
    Args:
        mat: Input matrix
        cond: Condition number threshold
        min_eig: Minimum eigenvalue threshold
        
    Returns:
        Matrix with nudged eigenvalues
    """
    # Process each frequency point
    eigw, eigv = [], []
    for mat_freq in mat:
        eigw_freq, eigv_freq = scipy.linalg.eig(mat_freq)
        eigw.append(eigw_freq)
        eigv.append(eigv_freq)
    
    # Convert to arrays
    eigw = np.array(eigw)
    eigv = np.array(eigv)

    # Calculate thresholds and apply nudging
    max_eig = np.amax(np.abs(eigw), axis=1)
    mask = (np.abs(eigw) < cond * max_eig[:, np.newaxis]) | (np.abs(eigw) < min_eig)
    
    if not np.any(mask):
        return mat

    # Apply nudging
    mask_cond = cond * np.tile(max_eig[:, np.newaxis], (1, eigw.shape[1]))[mask]
    mask_min = min_eig * np.ones_like(mask_cond)
    eigw[mask] = np.maximum(mask_cond, mask_min).astype(eigw.dtype)

    # Reconstruct matrix
    e = np.zeros_like(mat)
    for i in range(len(mat)):
        np.fill_diagonal(e[i], eigw[i])
    return rsolve(eigv, eigv @ e)

def nudge_svd(mat: np.ndarray, cond: float = 1e-9, min_svd: float = 1e-12) -> np.ndarray:
    """Nudge singular values to avoid singularities.
    
    Singular values smaller than max(cond * max(SVD), min_svd) are nudged to that value.
    
    Args:
        mat: Input matrix
        cond: Condition number threshold
        min_svd: Minimum singular value threshold
        
    Returns:
        Matrix with nudged singular values
    """
    # Compute SVD
    U, s, Vh = np.linalg.svd(mat)
    max_svd = np.amax(s, axis=1)
    
    # Calculate and apply nudging
    mask = (s < cond * max_svd[:, np.newaxis]) | (s < min_svd)
    if not np.any(mask):
        return mat

    mask_cond = cond * np.tile(max_svd[:, np.newaxis], (1, mat.shape[-1]))[mask]
    mask_min = min_svd * np.ones_like(mask_cond)
    s[mask] = np.maximum(mask_cond, mask_min)

    # Reconstruct matrix
    S = np.zeros_like(mat)
    for i in range(len(mat)):
        np.fill_diagonal(S[i], s[i])
    return np.einsum('...ij,...jk,...lk->...il', U, S, Vh)

# ===============================================
# NETWORK PARAMETER CONVERSION FUNCTIONS
# ===============================================

def s2z(s: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Convert scattering parameters to impedance parameters.
    
    Uses power-waves formulation from Kurokawa et al.
    
    Args:
        s: Scattering parameters
        z0: Characteristic impedance
        
    Returns:
        Impedance parameters
    """
    nfreqs, nports, _ = s.shape

    # Create identity matrix
    Id = np.tile(np.eye(nports, dtype=s.dtype), (nfreqs, 1, 1))
    
    # Vectorized creation of F and G matrices
    F = np.zeros_like(s)
    G = np.zeros_like(s)
    
    # Vectorized diagonal setting using broadcasting
    diag_indices = np.arange(nports)
    F[:, diag_indices, diag_indices] = 1.0 / (2 * np.sqrt(z0.real))
    G[:, diag_indices, diag_indices] = z0
    
    # Convert to impedance parameters
    lhs = nudge_eig((Id - s) @ F)
    rhs = (s @ G + G.conj()) @ F
    
    # Vectorized solve for all frequency points
    z = np.zeros_like(s)
    for i in range(nfreqs):
        z[i] = np.linalg.solve(lhs[i], rhs[i])
    
    return z


def z2s(z: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Convert impedance parameters to scattering parameters.
    
    Uses power-waves formulation from Kurokawa et al.
    
    Args:
        z: Impedance parameters
        z0: Characteristic impedance
        
    Returns:
        Scattering parameters
    """
    nfreqs, nports, _ = z.shape
    
    # Vectorized creation of helper matrices
    F = np.zeros_like(z)
    G = np.zeros_like(z)
    
    # Vectorized diagonal setting using broadcasting
    diag_indices = np.arange(nports)
    F[:, diag_indices, diag_indices] = 1.0 / (2 * np.sqrt(z0.real))
    G[:, diag_indices, diag_indices] = z0
    
    # Convert to scattering parameters
    s = rsolve(F @ (z + G), F @ (z - G.conj()))
    return s

def s2y(s: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Convert scattering parameters to admittance parameters.
    
    Args:
        s: Scattering parameters
        z0: Characteristic impedance
        
    Returns:
        Admittance parameters
    """
    nfreqs, nports, _ = s.shape

    # Create identity matrix and helper matrices
    Id = np.tile(np.eye(nports, dtype=s.dtype), (nfreqs, 1, 1))
    F = np.zeros_like(s)
    G = np.zeros_like(s)
    
    # Vectorized diagonal setting using broadcasting
    diag_indices = np.arange(nports)
    F[:, diag_indices, diag_indices] = 1.0 / (2 * np.sqrt(z0.real))
    G[:, diag_indices, diag_indices] = z0
    
    # Convert to admittance parameters
    y = rsolve((s @ G + G.conj()) @ F, (Id - s) @ F)
    return y

def y2s(y: np.ndarray, z0: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Convert admittance parameters to scattering parameters.
    
    Args:
        y: Admittance parameters
        z0: Characteristic impedance
        epsilon: Small value for numerical stability
        
    Returns:
        Scattering parameters
    """
    nfreqs, nports, _ = y.shape

    # Add small real part to pure imaginary characteristic impedance
    z0_copy = z0.copy()
    z0_copy[z0_copy.real == 0] += epsilon

    # Create identity matrix and helper matrices
    Id = np.tile(np.eye(nports, dtype=y.dtype), (nfreqs, 1, 1))
    F = np.zeros_like(y)
    G = np.zeros_like(y)
    
    # Vectorized diagonal setting using broadcasting
    diag_indices = np.arange(nports)
    F[:, diag_indices, diag_indices] = 1.0 / (2 * np.sqrt(z0_copy.real))
    G[:, diag_indices, diag_indices] = z0_copy
    
    # Convert to scattering parameters
    s = rsolve(F @ (Id + G @ y), F @ (Id - G.conj() @ y))
    return s

# ===============================================
# PATTERN GENERATION UTILITIES
# ===============================================

def generate_test_pattern(num_lines: int) -> np.ndarray:
    """Generate test pattern for eye diagram simulation.
    
    Args:
        num_lines: Number of transmission lines
        
    Returns:
        Test pattern array with shape (num_lines, num_lines, pattern_length)
    """
    # Helper function for bit inversion
    invert_bits = lambda x: -x + 1
    
    # Define basic patterns
    patterns = {
        'single_bit_9b': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'double_bit_10b': np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
        'alternating_3bit_6b': np.array([1, 0, 1, 0, 1, 0]),
        'burst_3pair_12b': np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
        'double_bit_18b': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'alternating_5bit_9b': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]),
        'burst_5pair_18b': np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    }
    
    # Create vertical patterns for odd bit testing
    vertical_odd_pattern = np.hstack((
        patterns['single_bit_9b'],
        patterns['double_bit_10b'],
        patterns['alternating_3bit_6b'],
        patterns['burst_3pair_12b'],
        invert_bits(patterns['single_bit_9b']),
        invert_bits(patterns['double_bit_10b'])
    ))
    
    adjacent_odd_pattern = invert_bits(vertical_odd_pattern)
    vertical_isi_pattern = vertical_odd_pattern
    adjacent_isi_pattern = np.zeros_like(vertical_isi_pattern)
    
    # Create patterns for crosstalk testing
    vertical_crosstalk_pattern = np.hstack((
        patterns['single_bit_9b'],
        patterns['double_bit_18b'],
        patterns['alternating_3bit_6b'],
        patterns['burst_3pair_12b'],
        invert_bits(patterns['single_bit_9b']),
        invert_bits(patterns['double_bit_18b'])
    ))
    
    adjacent_crosstalk_pattern = np.hstack((
        patterns['alternating_5bit_9b'],
        patterns['burst_5pair_18b'],
        invert_bits(patterns['alternating_3bit_6b']),
        invert_bits(patterns['burst_3pair_12b']),
        invert_bits(patterns['alternating_5bit_9b']),
        invert_bits(patterns['burst_5pair_18b'])
    ))
    
    # Create simultaneous switching output pattern
    sso_pattern = np.hstack((
        invert_bits(patterns['alternating_3bit_6b']),
        invert_bits(patterns['alternating_3bit_6b']),
        invert_bits(patterns['burst_3pair_12b']),
        invert_bits(patterns['burst_3pair_12b'])
    ))
    
    # Combine all patterns
    vertical_pattern = np.hstack((vertical_odd_pattern, vertical_isi_pattern, vertical_crosstalk_pattern))
    adjacent_pattern = np.hstack((adjacent_odd_pattern, adjacent_isi_pattern, adjacent_crosstalk_pattern))
    
    # Create base pattern
    base_pattern = np.hstack((sso_pattern, vertical_pattern, np.tile(adjacent_pattern, 8), sso_pattern))
    
    # Optimized pattern generation using vectorized operations
    test_pattern = np.zeros((num_lines, base_pattern.size))
    
    # Pre-compute adjacent pattern tiles for all possible remainders
    max_remainder = 8
    adjacent_tiles = [np.tile(adjacent_pattern, i) if i > 0 else np.array([]) for i in range(max_remainder + 1)]
    
    # Vectorized pattern assembly
    for i in range(num_lines):
        remainder = i % 9
        if remainder < len(adjacent_tiles):
            left_adjacent = adjacent_tiles[remainder]
            right_adjacent = adjacent_tiles[8 - remainder]
        else:
            left_adjacent = np.tile(adjacent_pattern, remainder)
            right_adjacent = np.tile(adjacent_pattern, 8 - remainder)
        
        test_pattern[i, :] = np.hstack((
            sso_pattern,
            left_adjacent,
            vertical_pattern,
            right_adjacent,
            sso_pattern
        ))
    
    return np.broadcast_to(test_pattern, (num_lines, num_lines, base_pattern.size))