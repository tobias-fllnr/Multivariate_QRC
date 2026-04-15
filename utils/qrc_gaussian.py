import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from itertools import combinations

class GaussianQRC:
    def __init__(self, n, encoding_strength, coupling_strength, gamma, dt, cov_measurements="full", encoding_mode="one_to_one", return_fourth_moments=False, T=0.0, seed=42):
        """
        Initialize the Quantum Reservoir.

        Parameters:
        - n: Number of oscillators.
        - encoding_strength: Scalar to scale the input signal before adding to omega.
        - coupling_strength: Scalar to tune the interaction strength (J_scale).
        - gamma_dt: Energy decay rate times decay time.
        - dt: Time step (time the systems interacts via as decribed by the Hamiltonian).
        - cov_measurements: String, one of ['full', 'diag', 'q_only', 'p_only', 'q_diag', 'q_only_3', 'full_3'].
                    'full': Upper triangular matrix.
                    'diag': Diagonal elements only (variances).
                    'q_only': Correlations between q variables (even indices).
                    'p_only': Correlations between p variables (odd indices).
                    'q_diag': Diagonal elements of q variables only.
                    'q_only_3': Correlations between q variables but only for the first 3 oscillators (q0, q1, q2).
                    'full_3': Full upper triangular matrix but only for the first 3 oscillators (q0, p0, q1, p1, q2, p2).
        - encoding_mode: 'one_to_one' (default) or 'dense' or 'fill'. 
                    'one_to_one': Maps input dimension i to oscillator i.
                    'dense': Linearly combines inputs via a random matrix and maps to all oscillators.
                    'fill': Repeats input dimensions to fill available oscillators (block encoding).
        - return_fourth_moments: If True, also returns fourth moments.
        - T: Temperature of the environment (for thermal noise).
        - seed: Random seed for reproducibility.
        """
        self.n = n
        self.encoding_strength = encoding_strength
        self.coupling_strength = coupling_strength
        self.gamma = gamma
        self.dt = dt
        self.cov_measurements = cov_measurements
        self.encoding_mode = encoding_mode
        self.return_fourth_moments = return_fourth_moments
        self.T = T

        # This will be initialized later if encoding_mode is 'dense'
        self.Win = None

        # This will be initialized later if encoding_mode is 'fill'
        self.noise_fill = None

        # Set random seed
        np.random.seed(seed)

        valid_cov_measurements = ['full', 'diag', 'q_only', 'p_only', 'q_diag', 'q_only_3', 'full_3']
        if self.cov_measurements not in valid_cov_measurements:
            raise ValueError(f"cov_measurements must be one of {valid_cov_measurements}")

        if self.n <= 0:
            raise ValueError("Number of oscillators 'n' must be positive.")
        if self.encoding_strength < 0 or self.encoding_strength > 1.0:
            raise ValueError("Encoding strength must be between 0 and 1.")
        if self.coupling_strength < 0:
            raise ValueError("Coupling strength must be non-negative.")
        if self.gamma < 0:
            raise ValueError("Factor 'gamma' must be non-negative.")
        if self.dt <= 0:
            raise ValueError("Time step 'dt' must be positive.")
        
        # Precompute static matrices and decay factors
        self._initialize_symplectic_form()
        self._initialize_coupling_matrix()
        self._precompute_thermal_state()
        self._precompute_output_indices()
        self._generate_all_bipartitions()

    def run(self, input_sequence, return_negativity=False, return_purity=False, return_squeezing=False) -> tuple:
        """
        Process an input sequence through the reservoir. Returns means and flattened covariances. If return_negativity is True, also returns negativities.

        Parameters:
        - input_sequence: np.array of shape (sequence_length, number_of_dimension)
        - return_negativity: If True, also calculates and returns the logarithmic negativity for each time step.
        - return_purity: If True, also calculates and returns the purity for each time step.
        - return_squeezing: If True, also calculates and returns the maximum squeezing in dB for each time step.

        Returns:
        - means_history: np.array (steps, 2*N)
        - covs_flattened: np.array (steps, flattened_dim)
        - fourth_moments_flattened: np.array (steps, num_unique_cov_elements^2) (only if return_fourth_moments is True)
        - negativities: list of log negativities for each time step (optional)
        - purities: list of purities for each time step (optional)
        - squeezing_db: list of maximum squeezing in dB for each time step (optional)
        """
        # 1. Validation
        if input_sequence.ndim == 1:
            # Handle 1D input by reshaping to (seq_len, 1)
            input_sequence = input_sequence[:, np.newaxis]
            
        seq_len, input_dim = input_sequence.shape
        
        if self.encoding_mode in ['one_to_one', 'fill'] and input_dim > self.n:
            raise ValueError(f"Input dimension ({input_dim}) cannot exceed number of oscillators ({self.n}) in {self.encoding_mode} mode.")

        if self.encoding_mode == 'dense' and self.Win is None:
            # 1. Create random raw weights
            self.Win = np.random.uniform(-1, 1, (self.n, input_dim))
            
            # 2. Calculate the sum of absolute values for each row (oscillator)
            # axis=1 sums across the input connections for a specific oscillator
            row_sums = np.sum(np.abs(self.Win), axis=1, keepdims=True)
            
            # 3. Normalize: Divide weights by the row sum
            # This guarantees the resulting signal stays in [-1, 1] if inputs are in [-1, 1]
            self.Win = self.Win / row_sums

        if self.encoding_mode == 'fill' and self.noise_fill is None:
            # Create the noise_fill vector that will be added to omega
            self.noise_fill = np.random.uniform(0, 1, self.n)

        # 2. Initialization of State (Vacuum)
        current_mean = np.zeros(2 * self.n)
        current_cov = self.sigma_thermal.copy()
        
        means_history = []
        covs_flattened = []
        fourth_moments_flattened = []
        negativities = []
        purities = []
        squeezing = []
        # 3. Time Loop
        for t in range(seq_len):
            # --- A. Frequency Encoding ---
            # Start with base frequencies (all 1.0)
            current_omega = np.full(self.n, 1.0, dtype=float)
            scaled_input = (2 * input_sequence[t] - 1) # Normalize input to [-1, 1]

            if self.encoding_mode == 'dense':
                # Project input vector onto all oscillators
                # freq_shifts shape is (n_oscillators,)
                freq_shifts = self.Win @ scaled_input
                current_omega += self.encoding_strength * freq_shifts
            elif self.encoding_mode == 'fill':
                # Calculate how many oscillators per input dimension
                oscillators_per_dim = self.n // input_dim
                
                # Vectorized repetition: [a, b] -> [a, a, b, b]
                expanded_input = np.repeat(scaled_input, oscillators_per_dim)
                
                # Apply to the omega vector (leaves remainder oscillators at base freq)
                current_omega[:len(expanded_input)] += self.encoding_strength * expanded_input * self.noise_fill[:len(expanded_input)]
            else: 
                current_omega[:input_dim] += self.encoding_strength * scaled_input

            # --- B. Hamiltonian Construction ---
            # H_mat construction (optimized vector operations where possible, but loop is safe)
            H_mat = np.zeros((2 * self.n, 2 * self.n))
            
            # Diagonal elements (Harmonic Oscillator terms)
            for i in range(self.n):
                idx_q, idx_p = 2*i, 2*i+1
                H_mat[idx_q, idx_q] = current_omega[i]**2 + self.L[i, i]
                H_mat[idx_p, idx_p] = 1

            # Off-diagonal elements (Coupling terms)
            # Utilizing the symmetric property of L and precomputed structure
            # TODO: We can initialize the coupling matrix L in the constructor and just reuse it here, as it does not depend on the input. This will save time during the run loop.
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    coupling_val = self.L[i, j]
                    
                    idx_q_i = 2 * i
                    idx_q_j = 2 * j
                    
                    # Position-Position coupling
                    H_mat[idx_q_i, idx_q_j] = coupling_val
                    H_mat[idx_q_j, idx_q_i] = coupling_val

            # --- C. Exact Langevin Evolution Step ---
            # Calculate the propagator M and the noise covariance for this step
            M, noise_cov = self._compute_langevin_update(H_mat)
            
            # Update Mean: mu(t+1) = M @ mu(t)
            current_mean = M @ current_mean
            
            # Update Cov: sigma(t+1) = M @ sigma(t) @ M.T + noise_cov
            current_cov = M @ current_cov @ M.T + noise_cov
            
            # --- D. Store Results ---
            means_history.append(current_mean)
            flat_cov = self._flatten_covariance(current_cov)
            covs_flattened.append(flat_cov)

            # --- Calculate 4th Moments (Covariance Pairwise Products) ---
            if self.return_fourth_moments:
                # 1. Multiply every covariance element by every other covariance element
                outer_product = np.outer(flat_cov, flat_cov)
                
                # 2. Extract only the unique combinations (upper triangle) 
                # This avoids feeding the IPC redundant features like A*B and B*A
                unique_fourth_moments = outer_product[np.triu_indices_from(outer_product)]
                
                fourth_moments_flattened.append(unique_fourth_moments)

            if return_negativity:
                step_negativities = []
                for p_indices in self.partitions:
                    step_negativities.append(self._calculate_negativity(current_cov, p_indices))
                negativities.append(np.mean(step_negativities) if step_negativities else 0.0)
            
            if return_purity:
                purities.append(self._calculate_purity(current_cov))
            
            if return_squeezing:
                squeezing.append(self._calculate_max_squeezing_db(current_cov))
        
        # Return logic (same as before)
        ret = [np.array(means_history), np.array(covs_flattened)]
        if self.return_fourth_moments: ret.append(np.array(fourth_moments_flattened))
        if return_negativity: ret.append(np.array(negativities))
        if return_purity: ret.append(np.array(purities))
        if return_squeezing: ret.append(np.array(squeezing))
        return tuple(ret)

    def _compute_langevin_update(self, H_mat):
        """
        Computes the exact evolution matrices for the equation:
        d_sigma/dt = A @ sigma + sigma @ A.T + D
        
        Where:
        A (Drift) = Omega @ H - (gamma/2) * I
        D (Diffusion) = gamma * sigma_thermal
        """
        dim = 2 * self.n
        
        # 1. Construct Drift Matrix A
        # (Generator of unitary rotation + Generator of amplitude decay)
        A = (self.Omega @ H_mat) - (0.5 * self.gamma * np.eye(dim))

        # Case 1: Lossless evolution (Gamma approx 0)
        # The system oscillates forever; we just compute the unitary rotation.
        if self.gamma < 1e-9:
            # M is just the Symplectic Matrix S = exp(Omega * H * dt)
            M = expm(A * self.dt)
            # No noise is added
            noise_cov = np.zeros((dim, dim))
            return M, noise_cov
        
        # Case 2: Dissipative evolution
        # The system has a valid steady state; we use the Lyapunov solver.
        else:
            # 2. Construct Diffusion Matrix D
            # (Rate of noise injection required to maintain Heisenberg uncertainty)
            D = self.gamma * self.sigma_thermal
            
            # 3. Solve for Instantaneous Steady State (Sigma_inf)
            # The Lyapunov equation: A @ Sigma_inf + Sigma_inf @ A.T = -D
            # This tells us where the state would go if input remained constant forever.
            # Note: A is stable (eigenvalues have real part -gamma/2), so solution exists.
            sigma_inf = solve_continuous_lyapunov(A, -D)
            # 4. Compute Propagator M over time dt
            M = expm(A * self.dt)
            
            # 5. Compute Noise Contribution over time dt
            # Based on relation: sigma(t) = M @ (sigma(0) - sigma_inf) @ M.T + sigma_inf
            # Rearranged: sigma(t) = M @ sigma(0) @ M.T + (sigma_inf - M @ sigma_inf @ M.T)
            noise_cov = sigma_inf - (M @ sigma_inf @ M.T)
            
            return M, noise_cov

    def _precompute_thermal_state(self):
        """Precomputes the target thermal state based on T."""
        n_thermal = 0.0 if self.T < 1e-10 else 1.0 / (np.exp(1 / (self.T)) - 1)
        self.sigma_thermal = np.eye(2 * self.n) * (0.5 + n_thermal)
    
    def _calculate_purity(self, cov_matrix: np.ndarray) -> float:
        """
        Calculates the purity of a Gaussian state covariance matrix.
        
        Parameters:
        - cov_matrix: np.array of shape (2N, 2N)
        
        Returns:
        - purity: Scalar float (0 < purity <= 1.0)
        """
        # 1. Calculate Determinant of the covariance matrix
        # Note: For numerical stability with large N, we use logdet usually, 
        # but for typical reservoir sizes (N<20), det is fine.
        det_sigma = np.linalg.det(cov_matrix)
        
        # 2. Apply Formula: 1 / (2^N * sqrt(det))
        # We use (2.0)**self.n because normalization depends on N
        purity = 1.0 / ((2.0 ** self.n) * np.sqrt(det_sigma))
        
        return purity
    
    def _initialize_symplectic_form(self):
        """Creates the symplectic matrix Omega (2N x 2N)."""
        self.Omega = np.zeros((2 * self.n, 2 * self.n))
        for i in range(self.n):
            self.Omega[2*i, 2*i+1] = 1.0
            self.Omega[2*i+1, 2*i] = -1.0

    def _initialize_coupling_matrix(self):
        """
        Initializes the Laplacian matrix J for a network of N harmonic oscillators.
        
        The interaction strengths g_ij are drawn uniformly from [0, coupling_strength].
        L is constructed such that:
        - Off-diagonal L_ij = -g_ij
        - Diagonal L_ii = sum(g_ik for all k connected to i)
        
        Parameters:
        N (int): Number of oscillators in the network.
        coupling_strength (float): The upper bound for the uniform distribution of couplings.
        
        Returns:
        np.ndarray: The NxN symmetric Laplacian matrix L.
        """
        # 1. Initialize the interaction matrix g with zeros
        g = np.zeros((self.n, self.n))
        
        # 2. Fill off-diagonal elements with random values and ensure symmetry
        # The paper uses random weights for fully connected networks [cite: 477, 478]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Draw random coupling uniformly from [0, coupling_strength]
                val = np.random.uniform(0, self.coupling_strength)
                g[i, j] = val
                g[j, i] = val # Symmetry ensures physical spring constants 
                
        # 3. Construct the Laplacian Matrix L
        # Formula: L_ij = delta_ij * sum_k(g_ik) - (1 - delta_ij) * g_ij 
        L = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            # Calculate the diagonal term: Sum of all couplings connected to node i
            # (axis=1 sums the row, which includes all g_ik for fixed i)
            L[i, i] = np.sum(g[i, :])
            
            # Calculate off-diagonal terms
            for j in range(self.n):
                if i != j:
                    L[i, j] = -g[i, j]
                    
        self.L = L

    def _precompute_output_indices(self):
        """Precomputes the indices for flattening the covariance matrix."""
        dim = 2 * self.n
        
        if self.cov_measurements == 'diag':
            # Option 2: Only diagonal elements (q_i q_i and p_i p_i)
            self.cov_r, self.cov_c = np.diag_indices(dim)
            
        elif self.cov_measurements == 'q_only':
            # Option 3: Only elements belonging to q variables
            # (Intersection of Upper Triangular AND Even Row indices AND Even Col indices)
            r, c = np.triu_indices(dim)
            # Filter: Keep only where both row and col are even numbers
            mask = (r % 2 == 0) & (c % 2 == 0)
            self.cov_r = r[mask]
            self.cov_c = c[mask]
            
        elif self.cov_measurements == 'q_diag':
            # Option 4: Only diagonal elements of q variables
            r, c = np.diag_indices(dim)
            mask = (r % 2 == 0)
            self.cov_r = r[mask]
            self.cov_c = c[mask]
        
        elif self.cov_measurements == 'p_only':
            # Option 5: Only elements belonging to p variables
            # (Intersection of Upper Triangular AND Odd Row indices AND Odd Col indices)
            r, c = np.triu_indices(dim)
            # Filter: Keep only where both row and col are odd numbers
            mask = (r % 2 == 1) & (c % 2 == 1)
            self.cov_r = r[mask]
            self.cov_c = c[mask]
            
        elif self.cov_measurements == 'q_only_3':
            # Option 6: Only elements belonging to q variables with a step of 3
            r, c = np.triu_indices(dim)
            # Filter: Keep only where both row and col are multiples of 3
            mask = (r % 2 == 0) & (c % 2 == 0) & (r <= 5) & (c <= 5) # Only q0, q1, q2
            self.cov_r = r[mask]
            self.cov_c = c[mask]

        elif self.cov_measurements == 'full_3':
            # Option 7: Full upper triangular but only for the first 3 oscillators (q0, p0, q1, p1, q2, p2)
            r, c = np.triu_indices(dim)
            mask = (r <= 5) & (c <= 5) # Only indices corresponding to first 3 oscillators
            self.cov_r = r[mask]
            self.cov_c = c[mask]
            
        else:
            # Option 1: Whole upper triangular matrix (Default)
            self.cov_r, self.cov_c = np.triu_indices(dim)

    def _flatten_covariance(self, cov_matrix):
            """
            Extracts specific elements of the covariance matrix based on initialized mode.
            """
            # Simply return the elements at the precomputed indices
            return cov_matrix[self.cov_r, self.cov_c]
    
    def _calculate_spectral_radius(self, matrix):
        """Calculates the spectral radius of a given matrix."""
        eigenvalues = np.linalg.eigvals(matrix)
        return max(abs(eigenvalues))

    def _get_symplectic_eigenvalues(self, cov):
        # M = Omega * sigma
        M = self.Omega @ cov
        vals = np.linalg.eigvals(M)
        
        # Eigenvalues are purely imaginary +/- i*nu
        # We take the absolute value of the imaginary part
        return np.sort(np.abs(np.imag(vals)))
    
    def _calculate_negativity(self, cov, partition_indices):
        """
        Calculates Standard Negativity (Not Logarithmic) for a bipartition.
        
        Returns:
        - negativity: (Sum of negative eigenvalues of partial transpose)
        """
        # 1. Partial Transpose (Same as before)
        dim = 2 * self.n
        T = np.eye(dim)
        for idx in partition_indices:
            p_idx = 2*idx + 1
            T[p_idx, p_idx] = -1.0
        
        cov_pt = T @ cov @ T
        
        # 2. Get Symplectic Eigenvalues (Same as before)
        symp_vals = self._get_symplectic_eigenvalues(cov_pt)
        
        # 3. Calculate Negativity
        # Formula: N = 0.5 * (Product(1/(2*nu)) - 1) for all nu < 0.5
        # If no nu < 0.5, the product is empty (value 1), so N = 0.
        
        product_term = 1.0
        for nu in symp_vals:
            if nu < 0.5:
                # Avoid division by zero
                val = max(nu, 1e-12)
                product_term *= (1.0 / (2.0 * val))
                
        return 0.5 * (product_term - 1.0)

    def _generate_all_bipartitions(self):
        """Generates a list of all unique bipartitions (Subsystem A indices)."""
        oscillators = list(range(self.n))
        all_partitions = []
        
        # We only need to go up to N // 2 to avoid checking (A, B) and (B, A) which are symmetric
        # Special handling for N/2 to avoid duplicates
        for r in range(1, self.n // 2 + 1):
            for subset in combinations(oscillators, r):
                # If n is even and we are at the halfway point, we need to handle symmetry
                # E.g. (0,1) vs (2,3) for N=4. We fix 0 to be in the first set.
                if r == self.n / 2:
                    if 0 not in subset:
                        continue
                all_partitions.append(list(subset))
                
        self.partitions = all_partitions

    def _calculate_max_squeezing_db(self, cov_matrix: np.ndarray) -> float:
        """
        Calculates the maximum squeezing in the system in dB.
        
        Returns:
        - squeezing_db: Positive value means squeezed (e.g., 3dB). 
                        Negative value means noisy/anti-squeezed.
        """
        # 1. Compute eigenvalues of the symmetric covariance matrix
        # We use eigvalsh because the matrix is real and symmetric
        evals = np.linalg.eigvalsh(cov_matrix)
        
        # 2. Find the minimum variance (smallest eigenvalue)
        min_variance = np.min(evals)
        
        # 3. Normalize by vacuum variance (0.5)
        # Your code sets sigma_thermal = 0.5 * I for T=0
        vacuum_variance = 0.5
        
        # 4. Convert to dB
        # Formula: -10 * log10(Var / Var_vac)
        squeezing_db = -10.0 * np.log10(min_variance / vacuum_variance)
        
        return squeezing_db