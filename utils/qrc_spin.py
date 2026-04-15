from itertools import combinations, product
import qutip as qt
import numpy as np

class SpinQRC:
    def __init__(self, n: int, encoding_strength: float, coupling_strength: float, gamma: float, dt: float, model: str = "TiltedTFIM", encoding_method: str = "one_to_one", observables: str = "local_and_twoqubit", seed: int = 42):
        """Initializes the SpinQRC class with the specified parameters.
        Args:
            n (int): Number of qubits in the system.
            encoding_strength (float): Strength of the data encoding into the Hamiltonian.
            coupling_strength (float): Strength of the coupling between qubits.
            gamma (float): Decay rate for the collapse operators.
            dt (float): Time step for the evolution.
            model (str, optional): Type of Hamiltonian model to use ("TiltedTFIM" or "TFIM"). Defaults to "TiltedTFIM".
            encoding_method (str, optional): Method for encoding data into the Hamiltonian. Options: "one_to_one", "dense" and "fill". Defaults to "one_to_one".
            observables (str, optional): Type of observables to measure ("local", "local_and_twoqubit", "z_zz", "z_aa", "a_aa", "allpauli", "a_aa_3", "z_zz_3"). Defaults to "local_and_twoqubit".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.n = n
        self.encoding_strength = encoding_strength
        self.coupling_strength = coupling_strength
        self.gamma = gamma
        self.dt = dt
        self.model = model
        self.encoding_method = encoding_method
        self.observables = observables
        self.seed = seed

        self.constant_field = 1.0
        self.Win = None
        self.noise_fill = None


        np.random.seed(self.seed) 

        if self.encoding_method == "fill":
            self.noise_fill = np.random.uniform(0, 1, self.n)
        
        # Cache single-site operators to avoid calling qt.tensor in loops
        self.sx_ops = [self._build_operator(qt.sigmax(), i) for i in range(n)]
        self.sz_ops = [self._build_operator(qt.sigmaz(), i) for i in range(n)]

        self.J = self._generate_coupling_matrix()
        self.initial_state = self._get_initial_state()
        self.collapse_operators = self._get_collapse_operators()
        self.observables = self._get_observables()
        # Build the parts of H that do not change with input data
        self.H_static = self._build_static_hamiltonian()

        self._generate_all_bipartitions()

    def run(self, data: np.ndarray, return_negativity: bool = False, return_coherence: bool = False) -> np.ndarray:
        """Evolves the density matrix over time using the Hamiltonian.
        Args:
            data (np.ndarray): Input data sequence of shape (steps, data_dim).
            return_negativity (bool): If True, calculates and returns standard negativity.
            return_coherence (bool): If True, calculates and returns the l1-norm of coherence.
        Returns:
            np.ndarray or tuple: Measurement results, and optionally negativities and/or coherences.
        """
        rho = self.initial_state
        measurement_results = np.zeros((data.shape[0], len(self.observables)))
        negativities = []
        coherences = []
        for i, dat in enumerate(data):
            # Calculate only the dynamic field part
            field = self._generate_field_strengths(dat)            
            # H = H_static + sum(field_i * sigma_x_i)
            # We subtract constant_field because it might be included in static, 
            # or we can just sum the differences. 
            # Here we reconstruct the dynamic part efficiently:
            H_dynamic = 0
            if self.model == "TiltedTFIM":
                # For TiltedTFIM, data maps to Sx fields
                # We use numpy broadcasting or a quick loop summing precomputed Qobjs
                # This avoids qt.tensor entirely
                for k in range(self.n):
                    if field[k] != 0: # Optimization: skip zero fields
                        H_dynamic += field[k] * self.sx_ops[k]
                H = self.H_static + H_dynamic
                
            elif self.model == "TFIM":
                # For TFIM, data maps to Sz fields (overwriting static const field if needed)
                # Note: Logic depends on if 'field' replaces or adds to constant. 
                # Assuming standard TFIM where field is the total Sz coefficient:
                H_dynamic = 0 
                for k in range(self.n):
                     H_dynamic += (field[k] - self.constant_field) * self.sz_ops[k]
                H = self.H_static + H_dynamic
            else:
                raise ValueError(f"Unknown model: {self.model}")

            # Evolve
            result = qt.mesolve(H, rho, tlist=[0, self.dt], c_ops=self.collapse_operators, options={"nsteps": 50000})
            rho = result.states[-1]
            
            # qt.expect handles the trace logic in C++ and accepts a list
            # resulting in massive speedup over Python loops.
            measurement_results[i] = qt.expect(self.observables, rho)

            if return_negativity:
                step_negativities = []
                for p_indices in self.partitions:
                    step_negativities.append(self._calculate_negativity(rho, p_indices))
                negativities.append(np.mean(step_negativities) if step_negativities else 0.0)
                
            if return_coherence:
                coherences.append(self._calculate_coherence(rho))
        
        # --- Handle dynamic returns cleanly ---
        returns = [measurement_results]
        if return_negativity:
            returns.append(np.array(negativities))
        if return_coherence:
            returns.append(np.array(coherences))
            
        # Return a single array if only measurements are requested, otherwise return a tuple
        if len(returns) == 1:
            return returns[0]
        return tuple(returns)

    def _calculate_coherence(self, rho: qt.Qobj) -> float:
        """
        Calculates the l1-norm of coherence for the density matrix.
        
        Returns:
        - coherence: The sum of the absolute values of the off-diagonal elements.
        """
        # Extract the dense numpy array for fast computation
        rho_matrix = rho.full()
        
        # Sum all absolute values, then subtract the absolute values of the diagonal
        total_abs_sum = np.sum(np.abs(rho_matrix))
        diag_abs_sum = np.sum(np.abs(np.diag(rho_matrix)))
        
        return float(total_abs_sum - diag_abs_sum)
    
    def _generate_all_bipartitions(self):
        """Generates a list of all unique bipartitions (Subsystem A indices)."""
        qubits = list(range(self.n))
        all_partitions = []
        
        # We only need to go up to n // 2 to avoid checking (A, B) and (B, A) which are symmetric
        for r in range(1, self.n // 2 + 1):
            for subset in combinations(qubits, r):
                if r == self.n / 2:
                    # Fix 0 to be in the first set to avoid duplicates (e.g., (0,1) vs (2,3) for N=4)
                    if 0 not in subset:
                        continue
                all_partitions.append(list(subset))
                
        self.partitions = all_partitions

    def _calculate_negativity(self, rho: qt.Qobj, partition_indices: list) -> float:
        """
        Calculates Standard Negativity for a bipartition.
        
        Returns:
        - negativity: The sum of the absolute values of the negative eigenvalues 
                      of the partially transposed density matrix.
        """
        # Create a mask for partial transpose: 1 if transposed, 0 if not
        mask = [1 if i in partition_indices else 0 for i in range(self.n)]
        
        # Perform partial transpose
        rho_pt = qt.partial_transpose(rho, mask)
        
        # Calculate eigenvalues (eigenenergies is faster since rho_pt is Hermitian)
        evals = rho_pt.eigenenergies()
        
        # Sum the absolute values of the strictly negative eigenvalues
        neg_evals = evals[evals < 0]
        if len(neg_evals) > 0:
            return float(np.sum(np.abs(neg_evals)))
        return 0.0

    def _build_static_hamiltonian(self) -> qt.Qobj:
        """Constructs the constant parts of the Hamiltonian once."""
        H_static = 0
        
        # Constant Transverse/Longitudinal fields (Base Hamiltonian)
        for i in range(self.n):
            if self.model == "TiltedTFIM":
                # Sz is constant in TiltedTFIM
                H_static += self.constant_field * self.sz_ops[i]
            elif self.model == "TFIM":
                # Sz is dynamic in TFIM, so we might add the baseline here
                H_static += self.constant_field * self.sz_ops[i]

        # Coupling Terms (Always static)
        # J is symmetric, iterate upper triangle to save time
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.J[i, j] != 0:
                    # Use precomputed operators
                    # Op(i,j) = sx_ops[i] * sx_ops[j] (Matrix multiplication of full operators)
                    H_static += self.J[i, j] * (self.sx_ops[i] * self.sx_ops[j])
                    
        return H_static
    
    def _get_observables(self) -> list:
        """Generates a list of observables for the system.
        This method creates a list of observables based on the specified type.
        If 'all' is specified, it generates all combinations of Pauli operators for the qubits.
        If 'local' is specified, it generates local Pauli-Z operators for each qubit.
        Returns:
            list: A list of qutip.Qobj objects representing the observables.
        """
        pauli_operators = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        observables = []
        if self.observables == "allpauli":
            pauli_operators = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()] # in this case include identity
            for ops in product(pauli_operators, repeat=self.n):
                observable = qt.tensor(ops)
                observables.append(observable)
        elif self.observables == "local":
            for i in range(self.n):
                for op in pauli_operators:
                    obs = self._build_operator(op, i)
                    observables.append(obs)
        elif self.observables == "local_and_twoqubit":
            for i in range(self.n):
                for op in pauli_operators:
                    obs = self._build_operator(op, i)
                    observables.append(obs)
            for i, j in combinations(range(self.n), 2):
                for op1 in pauli_operators:
                    for op2 in pauli_operators:
                        obs = qt.tensor([op1 if k == i else op2 if k == j else qt.qeye(2) for k in range(self.n)])
                        observables.append(obs)
        elif self.observables == "z_zz":
            for i in range(self.n):
                obs = self._build_operator(qt.sigmaz(), i)
                observables.append(obs)
            for i, j in combinations(range(self.n), 2):
                obs = qt.tensor([qt.sigmaz() if k == i or k == j else qt.qeye(2) for k in range(self.n)])
                observables.append(obs)
        elif self.observables == "z_zz_3":
            for i in range(3):
                obs = self._build_operator(qt.sigmaz(), i)
                observables.append(obs)
            for i, j in combinations(range(3), 2):
                obs = qt.tensor([qt.sigmaz() if k == i or k == j else qt.qeye(2) for k in range(self.n)])
                observables.append(obs)
        elif self.observables == "z_aa":
            for i in range(self.n):
                obs = self._build_operator(qt.sigmaz(), i)
                observables.append(obs)
            for i, j in combinations(range(self.n), 2):
                for op in pauli_operators:
                    obs = qt.tensor([op if k == i or k == j else qt.qeye(2) for k in range(self.n)])
                    observables.append(obs)
        elif self.observables == "a_aa":
            for i in range(self.n):
                for op in pauli_operators:
                    obs = self._build_operator(op, i)
                    observables.append(obs)
            for i, j in combinations(range(self.n), 2):
                for op in pauli_operators:
                    obs = qt.tensor([op if k == i or k == j else qt.qeye(2) for k in range(self.n)])
                    observables.append(obs)
        elif self.observables == "a_aa_3":
            for i in range(3):
                for op in pauli_operators:
                    obs = self._build_operator(op, i)
                    observables.append(obs)
            for i, j in combinations(range(3), 2):
                for op in pauli_operators:
                    obs = qt.tensor([op if k == i or k == j else qt.qeye(2) for k in range(self.n)])
                    observables.append(obs)
        else:
            raise ValueError(f"Unknown observables type: {self.observables}")
        return observables

    def _get_collapse_operators(self) -> list:
        """Generates collapse operators for the system.
        This method creates a list of collapse operators for each qubit in the system.
        Each collapse operator is defined as a square root of the gamma parameter
        multiplied by the lowering operator (sigma-) for that qubit.
        Returns:
            list: A list of qutip.Qobj objects representing the collapse operators for each qubit.
        """
        sm = qt.sigmam()
        operators = [np.sqrt(self.gamma)* qt.tensor([sm if k == i else qt.qeye(2) for k in range(self.n)]) for i in range(self.n)]
        return operators

    def _get_initial_state(self) -> qt.Qobj:
        """Initializes the density matrix for the system using tensor products.

        Returns:
            qutip.Qobj: The initial density matrix as a Qobj.
        """
        single_qubit_dm = qt.fock_dm(2, 0)  # Density matrix for |0⟩⟨0| in a 2-level system
        rho = qt.tensor([single_qubit_dm] * self.n)
        return rho
    
    def _hamiltonian(self, field: np.ndarray) -> qt.Qobj:
        """Constructs the Hamiltonian for an n-qubit system.
        Args:
            field (np.ndarray): An array of field strengths of shape (n,).
        Returns:
            qutip.Qobj: The Hamiltonian as a Qobj.
        """
        H = 0
        for i in range(self.n):
            szi = self._build_operator(qt.sigmaz(), i)
            sxi = self._build_operator(qt.sigmax(), i)
            if self.model == "TiltedTFIM":
                H += field[i] * sxi + self.constant_field * szi
            elif self.model == "TFIM":
                H += field[i] * szi
            else:
                raise ValueError(f"Unknown model: {self.model}")
            for j in range(i):
                sxj = self._build_operator(qt.sigmax(), j)
                H += self.J[i, j] * sxi * sxj

        return H

    def _generate_field_strengths(self, data_step: np.ndarray = None) -> np.ndarray:
        """
        Generates the field strengths for an n-qubit system based on the provided data step.
        Args:
            data_step (np.ndarray, optional): An array of data values to encode into the field strengths.
                If None, the field strengths will be set to a constant value.
        Returns:
            np.ndarray: An array of field strengths of shape (n,).
        """
        field = np.array([self.constant_field] * self.n)
        if data_step is None:
            return field
        else:
            if self.encoding_method == "one_to_one":
                for i, dat in enumerate(data_step):
                    field[i] = self.constant_field + 2*self.encoding_strength*(dat-0.5)
            elif self.encoding_method == "fill":
                qubits_per_dim = self.n // len(data_step)
                for i, dat in enumerate(data_step):
                    start = i * qubits_per_dim
                    end = start + qubits_per_dim
                    field[start:end] = self.constant_field + 2*self.encoding_strength*(dat-0.5)* self.noise_fill[start:end]
        
            elif self.encoding_method == "dense":
            # 1. Initialize Weight Matrix if it doesn't exist
                if self.Win is None:
                    input_dim = len(data_step)
                    
                    # Create random raw weights uniform [-1, 1]
                    self.Win = np.random.uniform(-1, 1, (self.n, input_dim))
                    
                    # Calculate the sum of absolute values for each row (qubit)
                    row_sums = np.sum(np.abs(self.Win), axis=1, keepdims=True)
                    
                    # Normalize to guarantee stability
                    row_sums[row_sums == 0] = 1.0
                    self.Win = self.Win / row_sums

                # 2. Scale input from [0, 1] to [-1, 1]
                scaled_input = 2 * (data_step - 0.5)

                # 3. Project input vector onto all qubits (Matrix Multiplication)
                # Win shape: (n, input_dim) | Input shape: (input_dim,) -> Result: (n,)
                field_shifts = self.Win @ scaled_input
                
                # 4. Apply strength and add to constant field
                field = self.constant_field + self.encoding_strength * field_shifts
            else:
                raise ValueError(f"Unknown encoding_method: {self.encoding_method}")
            
            return field

    def _generate_coupling_matrix(self) -> np.ndarray:
        """Generates a random coupling matrix for an n-qubit system.

        This function generates a random coupling matrix with entries uniformly distributed
        between -coupling_strength/2 and coupling_strength/2.

        Returns:
            np.ndarray: A symmetric coupling matrix of shape (n, n).
        """
        J = np.random.uniform(-self.coupling_strength/2, self.coupling_strength/2, size=(self.n, self.n))
        return J

    def _build_operator(self, operator: qt.Qobj, i: int) -> qt.Qobj:
        """Places an operator at some site in an n-qubit system.

        This function builds the tensor product of an operator acting on the i-th spin
        in a n-spin system with identity operators acting on all of the other spins.

        Args:
            operator: operator to be placed,
            i (int): site of the operator,

        Returns:
            tensor product of operators."""
        ops = [qt.identity(2)] * self.n
        ops[i] = operator
        return qt.tensor(ops)
