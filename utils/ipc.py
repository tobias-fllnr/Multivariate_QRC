from itertools import combinations, product, combinations_with_replacement
import numpy as np
from scipy.stats import chi2
from scipy.special import legendre

class IPC:
    def __init__(self, values: np.ndarray, targets: np.ndarray, washout: int, train_length: int):
        """Initializes the IPC class.
        Args:
            values (np.ndarray): The reservoir states or features.
            targets (np.ndarray): The target data for IPC evaluation.
            washout (int): Number of initial time steps to discard.
            train_length (int): Length of the data to be used to calculate IPC.
        """
        self.values = values
        self.targets = targets
        self.washout = washout
        self.train_length = train_length

    def ipc(
        self, max_delay: int = 1000, max_degree: int = 10,
        return_ipc: bool = True,
        return_capacity_mixing: bool = False,
    ) -> tuple[dict, dict, dict] | tuple[dict, dict, dict, dict]:
        """
        Optimized IPC task calculation.
        Major improvements:
        1. Pre-computes the pseudo-inverse of the reservoir states (eliminates repeated training).
        2. Caches Legendre polynomial transformations.
        3. Vectorizes delay calculations (computes all delays for a specific pattern at once).

        Args:
            max_delay: Maximum delay to consider.
            max_degree: Maximum polynomial degree.
            return_ipc: If True, returns the full IPC dictionary with all capacities.
            return_capacity_mixing: If True, also returns two additional dictionaries:
                1. `capacity_mixing`: A dictionary containing per-degree capacity from cross-dimension mixing terms,
                    i.e. terms where |unique(dims)| > 1.
                2. `mixing_capacity_breakdown`: A nested dictionary containing detailed breakdown of mixing capacities
                    by degree, dimension combination, position, and delay.
        """
        num_dims = self.targets.shape[1]
        n_observables = self.values.shape[1]
        
        # 1. Precompute Pseudo-Inverse for Linear Regression
        # slice data for training
        start = self.washout
        end = start + self.train_length
        X_train = self.values[start:end]
        
        # Add bias column (equivalent to LinearRegression(fit_intercept=True))
        X_b = np.c_[np.ones(X_train.shape[0]), X_train]
        
        # Compute Pinv: weights = X_pinv @ target
        # This is the most computationally expensive step, now done only once.
        X_pinv = np.linalg.pinv(X_b)
        
        # Pre-calculate constants
        t_stat = chi2.ppf(1 - 0.0001, df=n_observables)
        cut = 2 * t_stat / self.train_length
        
        # 2. Pre-cache Legendre transformed targets to avoid re-calculation
        # Cache structure: dict[degree][dimension] -> array
        legendre_cache = self._cache_legendre_targets(max_degree)

        capacity_dict = {}
        mixing_capacity_dict = {}
        mixing_capacity_breakdown = {}

        for degree in range(1, max_degree + 1):
            capacity_dict[degree] = []
            if return_capacity_mixing:
                mixing_capacity_dict[degree] = []

            for variables in range(1, degree + 1):
                power_list = self._generate_power_list(degree, variables)

                for powers in power_list:
                    dim_combinations = list(product(range(num_dims), repeat=len(powers)))
                    evaluated_equal_delay_signatures = set()
                    for dim_combination in dim_combinations:
                        is_mixing = len(set(dim_combination)) > 1 and len(set(dim_combination)) == degree
                        break_window = False

                        if return_ipc or (return_capacity_mixing and is_mixing):
                            for window in range(len(powers), max_delay + 1):
                                # Get all positions relative to window start (0 ... window-1)
                                positions = [
                                    pos
                                    for pos in combinations(range(window), len(powers))
                                    if pos[0] == 0 and pos[-1] == window - 1
                                ]

                                break_positions = False
                                for pos in positions:
                                    # 3. Optimization: Generate Base Target Once
                                    # Calculate the product for delay=0 (the relative pattern)
                                    base_target_full = self._generate_base_target_fast(
                                        powers, dim_combination, pos, legendre_cache
                                    )

                                    # 4. Optimization: Vectorized Batch Processing
                                    # Instead of looping delay 0..max, we solve all delays at once
                                    # Generate matrix of targets for all valid delays
                                    valid_delays = range(max_delay - window + 1)
                                    if not valid_delays:
                                        continue

                                    # Create batch: (Time, Num_Delays)
                                    # We shift the base_target for each delay
                                    targets_batch = self._create_delay_batch(
                                        base_target_full, valid_delays, start, end
                                    )

                                    # Solve all at once
                                    capacities = self._calculate_capacity_batch(
                                        X_train, X_pinv, targets_batch
                                    )

                                    # Analyze results (replicating the break logic)
                                    for i, delay in enumerate(valid_delays):
                                        cap = capacities[i]
                                        if cap < cut:
                                            if delay == 0:
                                                continue
                                            elif delay == 1:
                                                break_positions = True
                                                if pos == positions[0]:
                                                    break_window = True
                                            break # Stop processing delays for this pos
                                        else:
                                            if return_ipc:
                                                capacity_dict[degree].append(cap)
                                                
                                            if return_capacity_mixing and is_mixing:
                                                mixing_capacity_dict[degree].append(cap)
                                                mixing_capacity_breakdown.setdefault(degree, {}).setdefault(dim_combination, {}).setdefault(pos, {})[delay] = cap
                                                # print(f"Degree {degree}, Vars {variables}, Dims {dim_combination}, Pos {pos}, Delay {delay}, Cap {cap}")


                                    if break_positions:
                                        break
                                if break_window:
                                    break


                        if return_capacity_mixing and is_mixing:
                            break_window_eq = False
                            # Equal delays can occur in any window size, so start from 1
                            for window in range(1, max_delay + 1):
                                # Only get positions with at least one identical relative delay
                                positions_eq = [
                                    pos
                                    for pos in combinations_with_replacement(range(window), len(powers))
                                    if pos[0] == 0 and pos[-1] == window - 1 and len(set(pos)) < len(pos)
                                ]
                                
                                break_positions_eq = False
                                for pos in positions_eq:
                                    # Filter 1: Prevent variables from the SAME dimension having the SAME delay
                                    dim_pos_pairs = list(zip(dim_combination, pos))
                                    if len(set(dim_pos_pairs)) < len(dim_pos_pairs):
                                        continue
                                        
                                    # Filter 2: Prevent permutation duplicates
                                    sig = tuple(sorted(zip(powers, dim_combination, pos)))
                                    if sig in evaluated_equal_delay_signatures:
                                        continue
                                    evaluated_equal_delay_signatures.add(sig)
                                    
                                    # Evaluate mathematically valid missing terms
                                    base_target_full = self._generate_base_target_fast(
                                        powers, dim_combination, pos, legendre_cache
                                    )
                                    
                                    valid_delays = range(max_delay - window + 1)
                                    if not valid_delays:
                                        continue
                                        
                                    targets_batch = self._create_delay_batch(
                                        base_target_full, valid_delays, start, end
                                    )
                                    capacities = self._calculate_capacity_batch(
                                        X_train, X_pinv, targets_batch
                                    )
                                    
                                    for i, delay in enumerate(valid_delays):
                                        cap = capacities[i]
                                        if cap < cut:
                                            if delay == 0:
                                                continue
                                            elif delay == 1:
                                                break_positions_eq = True
                                                if len(positions_eq) > 0 and pos == positions_eq[0]:
                                                    break_window_eq = True
                                            break
                                        else:
                                            # We append ONLY to mixing_capacity_dict here!
                                            mixing_capacity_dict[degree].append(cap)
                                            mixing_capacity_breakdown.setdefault(degree, {}).setdefault(dim_combination, {}).setdefault(pos, {})[delay] = cap
                                            # print(f"Degree {degree}, Vars {variables}, Dims {dim_combination}, Pos {pos}, Delay {delay}, Cap {cap}")

                                            
                                    if break_positions_eq:
                                        break
                                if break_window_eq:
                                    break
        returns = []
        if return_ipc:
            capacity_dict_per_degree = {k: np.sum(v) for k, v in capacity_dict.items()}
            capacity_dict_per_degree_normalized = {
                k: v / n_observables for k, v in capacity_dict_per_degree.items()
            }
            returns.append(capacity_dict)
            returns.append(capacity_dict_per_degree)
            returns.append(capacity_dict_per_degree_normalized)

        if return_capacity_mixing:
            mixing_capacity_per_degree_normalized = {
                k: np.sum(v)/n_observables for k, v in mixing_capacity_dict.items()
            }
            returns.append(mixing_capacity_breakdown)
            returns.append(mixing_capacity_per_degree_normalized)

        
        return tuple(returns)

    def _cache_legendre_targets(self, max_degree: int) -> dict:
        """Pre-computes Legendre polynomials for raw targets."""
        cache = {}
        # Normalize once
        norm_targets = 2 * self.targets - 1
        for d in range(1, max_degree + 1):
            cache[d] = []
            poly_func = self._normalized_legendre(d)
            # Apply to all dimensions at once if possible, or loop dims
            # Assuming poly_func handles vectorization
            term = poly_func(norm_targets) 
            cache[d] = term 
        return cache

    def _generate_base_target_fast(
        self, powers: tuple, dim_combination: tuple, relative_delays: tuple, cache: dict
    ) -> np.ndarray:
        """
        Generates the polynomial target for the base position (delay=0).
        Uses cached Legendre features.
        """
        # We process the full length first, then slice later
        n_samples = self.targets.shape[0]
        result = np.ones(n_samples)
        
        for power, dim, delay in zip(powers, dim_combination, relative_delays):
            # Retrieve pre-calculated legendre column
            col_data = cache[power][:, dim]
            
            # Apply relative shift (within the window)
            if delay > 0:
                col_data = np.roll(col_data, delay)
                # Handle edge (optional based on your original logic)
                # col_data[:delay] = 0 
                
            result *= col_data
            
        return result

    def _create_delay_batch(
        self, base_target: np.ndarray, delays: range, start: int, end: int
    ) -> np.ndarray:
        """
        Creates a 2D matrix (Time x Delays) by rolling the base target.
        """
        n_delays = len(delays)
        train_len = end - start
        batch = np.zeros((train_len, n_delays))
        
        # We can construct this efficiently
        # base_target is full length. We need base_target[start+d : end+d] roughly
        # Since we use np.roll(target, delay), effectively we take different slices
        
        for i, d in enumerate(delays):
            # Rolling the full array by d, then slicing [start:end]
            # np.roll shifts elements to higher indices (right shift)
            # target[t] becomes target[t-d]
            
            # Optimization: Slicing is faster than rolling the whole array
            # shifted[k] = original[k - d]
            # We want indices [start, start+1, ..., end-1] from the shifted array
            # These map to [start-d, ..., end-1-d] in original array
            
            # Check bounds to avoid wrap-around logic if strictly non-cyclic
            s_idx = start - d
            e_idx = end - d
            
            # If using cyclic roll (standard RC usually ignores wrap-around issues 
            # due to washout, but let's stick to numpy roll behavior)
            rolled = np.roll(base_target, d)
            batch[:, i] = rolled[start:end]
            
        return batch

    def _calculate_capacity_batch(
        self, X: np.ndarray, X_pinv: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        """
        Calculates capacity for multiple targets Y simultaneously.
        Y shape: (Samples, n_targets)
        """
        # 1. Calculate Weights: W = X_pinv @ Y
        W = X_pinv @ Y
        
        # 2. Calculate Predictions: Y_hat = X_b @ W
        # Reconstruct X with bias for prediction
        X_b = np.c_[np.ones(X.shape[0]), X]
        Y_hat = X_b @ W
        
        # 3. Calculate MSE
        residuals = Y - Y_hat
        mse = np.mean(residuals**2, axis=0)
        
        # 4. Calculate Capacity
        # cap = 1 - MSE / MeanSquare(Target)
        y_sq_mean = np.mean(Y**2, axis=0)
        
        # Handle division by zero edge case
        with np.errstate(divide='ignore', invalid='ignore'):
            capacity = 1 - mse / y_sq_mean
            
        return capacity
    
    def _normalized_legendre(self, n):
        """Return a callable normalized Legendre polynomial of degree n."""
        Pn = legendre(n)
        norm_factor = np.sqrt((2 * n + 1) / 2)
        return lambda x: norm_factor * Pn(x)
    
    def _generate_power_list(self, degree: int, variables: int) -> list:
        """Generates a list of tuples representing the powers for each variable.
        This method generates all possible combinations of powers for a given degree
        and number of variables, ensuring that the sum of the powers equals the degree.
        Args:
            degree (int): The maximum degree of the polynomial.
            variables (int): The number of variables for which to generate the powers.
        Returns:
            list: A list of tuples, where each tuple represents a combination of powers
            that sum to the degree.
        """
        if variables == 1:
            return [(degree,)]
        # choose (variables - 1) cut points between 1 and degree - 1
        cuts = combinations(range(1, degree), variables - 1)
        power_list = []
        for cut in cuts:
            parts = []
            prev = 0
            for c in cut:
                parts.append(c - prev)
                prev = c
            parts.append(degree - prev)
            power_list.append(tuple(parts))
        return power_list