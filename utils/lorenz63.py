import numpy as np

class Lorenz63Generator:
    """
    Generates data using the classic Lorenz-63 atmospheric convection model.
    Uses 4th-order Runge-Kutta (RK4) integration.
    """
    def __init__(self,
                 length: int,
                 sigma: float = 10.0,
                 rho: float = 28.0,
                 beta: float = 8.0 / 3.0,
                 dt_int: float = 0.01,
                 dt_data: float = 0.1,
                 t_init_cutoff: float = 20.0,
                 seed: int = 42):
        
        self.length = length
        self.dimension = 3  # Lorenz-63 is strictly 3-dimensional
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt_int = dt_int
        self.dt_data = dt_data
        self.t_init_cutoff = t_init_cutoff
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        # Validation checks
        if self.dt_data < self.dt_int:
            raise ValueError("dt_data (sampling step) must be >= dt_int (integration step).")

    def _lorenz63(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the spatial derivatives for the Lorenz-63 model.
        """
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        return np.array([dx, dy, dz], dtype=np.float64)

    def generate(self) -> np.ndarray:
        """
        Generates the time series data using RK4 integration.
        """
        # Initialize slightly off the origin to avoid getting stuck at the trivial fixed point
        state = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        state += self.rng.uniform(-0.1, 0.1, self.dimension)
        
        cutoff_steps = int(self.t_init_cutoff / self.dt_int)
        steps_per_sample = int(self.dt_data / self.dt_int)
        
        # 1. Burn-in Phase: Discard transient steps without storing them
        for _ in range(cutoff_steps):
            k1 = self._lorenz63(state)
            k2 = self._lorenz63(state + 0.5 * self.dt_int * k1)
            k3 = self._lorenz63(state + 0.5 * self.dt_int * k2)
            k4 = self._lorenz63(state + self.dt_int * k3)
            state += (self.dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 2. Sampling Phase
        sampled_trajectory = np.empty((self.length, self.dimension), dtype=np.float64)
        
        for i in range(self.length):
            sampled_trajectory[i] = state
            
            # Integrate forward to the next sample point
            for _ in range(steps_per_sample):
                k1 = self._lorenz63(state)
                k2 = self._lorenz63(state + 0.5 * self.dt_int * k1)
                k3 = self._lorenz63(state + 0.5 * self.dt_int * k2)
                k4 = self._lorenz63(state + self.dt_int * k3)
                state += (self.dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return sampled_trajectory
    
    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the analytical 3x3 Jacobian matrix for the Lorenz-63 system at state.
        """
        x, y, z = state
        J = np.array([
            [-self.sigma, self.sigma, 0.0],
            [self.rho - z, -1.0, -x],
            [y, x, -self.beta]
        ], dtype=np.float64)
        
        return J
    
    def calculate_lyapunov_exponents(self, 
                                     steps: int = 1000000, 
                                     return_history: bool = False, 
                                     history_interval: int = 100, 
                                     qr_interval: int = 50,
                                     tolerance: float = 1e-3,
                                     patience: int = 10,
                                     min_steps: int = 10000):
        """
        Calculates the full spectrum of Lyapunov exponents (3 for Lorenz-63) 
        using Benettin's algorithm.
        """
        state = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        state += self.rng.uniform(-0.1, 0.1, self.dimension)
        
        # 1. Burn-in Phase
        cutoff_steps = int(self.t_init_cutoff / self.dt_int)
        if cutoff_steps == 0:
            cutoff_steps = 1000
            
        for _ in range(cutoff_steps):
            k1 = self._lorenz63(state)
            k2 = self._lorenz63(state + 0.5 * self.dt_int * k1)
            k3 = self._lorenz63(state + 0.5 * self.dt_int * k2)
            k4 = self._lorenz63(state + self.dt_int * k3)
            state += (self.dt_int / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 2. Setup for LE calculation
        Q = np.eye(self.dimension, dtype=np.float64)
        lyap_sum = np.zeros(self.dimension, dtype=np.float64)
        
        history_exponents = None
        history_times = None
        record_idx = 0
        
        if return_history:
            num_records = steps // history_interval
            history_exponents = np.zeros((num_records, self.dimension), dtype=np.float64)
            history_times = np.zeros(num_records, dtype=np.float64)

        # Convergence tracking variables
        prev_largest_le = None
        stable_checks = 0
        steps_taken = steps
        
        # 3. Integration Phase
        for step in range(1, steps + 1):
            # RK4 Integration for both state and tangent space (Q)
            k1_x = self._lorenz63(state)
            J1 = self._jacobian(state)
            k1_Q = J1 @ Q
            
            x2 = state + 0.5 * self.dt_int * k1_x
            Q2 = Q + 0.5 * self.dt_int * k1_Q
            k2_x = self._lorenz63(x2)
            J2 = self._jacobian(x2)
            k2_Q = J2 @ Q2
            
            x3 = state + 0.5 * self.dt_int * k2_x
            Q3 = Q + 0.5 * self.dt_int * k2_Q
            k3_x = self._lorenz63(x3)
            J3 = self._jacobian(x3)
            k3_Q = J3 @ Q3
            
            x4 = state + self.dt_int * k3_x
            Q4 = Q + self.dt_int * k3_Q
            k4_x = self._lorenz63(x4)
            J4 = self._jacobian(x4)
            k4_Q = J4 @ Q4
            
            state += (self.dt_int / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            Q += (self.dt_int / 6.0) * (k1_Q + 2*k2_Q + 2*k3_Q + k4_Q)
            
            # QR decomposition periodically to prevent overflow/alignment
            if step % qr_interval == 0:
                Q, R = np.linalg.qr(Q)
                # Ensure positive diagonal elements for log
                diag_R = np.abs(np.diag(R))
                lyap_sum += np.log(diag_R)
            
            # History recording and Convergence check
            if step % history_interval == 0:
                completed_qr_time = (step // qr_interval) * qr_interval * self.dt_int
                
                if completed_qr_time > 0:
                    current_le = np.sort(lyap_sum / completed_qr_time)[::-1]
                    largest_le = current_le[0]
                    
                    if return_history:
                        history_exponents[record_idx] = current_le
                        history_times[record_idx] = completed_qr_time
                        record_idx += 1
                        
                    # Evaluate Convergence
                    if step >= min_steps:
                        if prev_largest_le is not None:
                            if abs(largest_le - prev_largest_le) < tolerance:
                                stable_checks += 1
                                if stable_checks >= patience:
                                    steps_taken = step
                                    break
                            else:
                                stable_checks = 0
                        prev_largest_le = largest_le

        # Final cleanup for exact steps taken
        if steps_taken % qr_interval != 0:
            Q, R = np.linalg.qr(Q)
            lyap_sum += np.log(np.abs(np.diag(R)))

        total_time = steps_taken * self.dt_int
        final_le = np.sort(lyap_sum / total_time)[::-1]
        
        if stable_checks >= patience:
            print(f"Converged early after {steps_taken} steps (Integrated Time: {total_time:.2f})")
        else:
            print(f"Warning: Reached maximum limit of {steps} steps without fully converging to tolerance.")
        
        if return_history and history_times is not None and history_exponents is not None:
            return final_le, (history_times[:record_idx], history_exponents[:record_idx])
            
        return final_le