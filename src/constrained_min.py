import numpy as np

class InteriorPointMinimizer:
    def __init__(self, func, grad_func, hess_func, eq_constraints_mat, eq_constraints_rhs, ineq_constraints, ineq_grad_funcs, ineq_hess_funcs, x0, t=1, mu=10, epsilon=1e-8, obj_tol=1e-8, param_tol=1e-8, max_iter_outer=100, max_iter_inner=100, step_len="wolfe"):
        # Objective function we want to minimize
        self.func = func
        # Gradient of the objective
        self.grad_func = grad_func
        # Hessian of the objective
        self.hess_func = hess_func
        # Matrix for equality constraints Ax=b 
        self.eq_constraints_mat = eq_constraints_mat
        # Right hand side vector b
        self.eq_constraints_rhs = eq_constraints_rhs
        # List of inequality constraint functions g(x) >= 0
        self.ineq_constraints = ineq_constraints
        # List of gradient functions for inequality constraints
        self.ineq_grad_funcs = ineq_grad_funcs
        # List of Hessian functions for inequality constraints
        self.ineq_hess_funcs = ineq_hess_funcs
        # Initial guess for solution
        self.x = np.array(x0, dtype=float)
        # Initial barrier param
        self.t = t
        # Factor to increase the barrier param t
        self.mu = mu
        # Tolerance for the barrier method
        self.epsilon = epsilon
        # Tolerance for changes in the obj. function
        self.obj_tol = obj_tol
        # Tolerance for changes in the solution
        self.param_tol = param_tol
        # Max. number of outer iterations
        self.max_iter_outer = max_iter_outer
        # Max. number of inner iterations
        self.max_iter_inner = max_iter_inner
        # Step length method for line search 
        self.step_len = step_len
        # Store data
        self.path = []
        self.obj_values = []
        self.outer_obj_values = []

    def _phi(self, x):
        # Computes the log-barrier term for the inequality constraints at point x
        """Log-barrier term."""
        return -np.sum([np.log(g(x)) for g in self.ineq_constraints])

    def _grad_phi(self, x):
        # Computes the gradient of the log-barrier term
        """Gradient of log-barrier term."""
        return -np.sum([grad_g(x) / g(x) for g, grad_g in zip(self.ineq_constraints, self.ineq_grad_funcs)], axis=0)

    def _hess_phi(self, x):
        # Computes the Hessian of the log-barrier term
        """Hessian of log-barrier term."""
        n = len(x)
        hessian = np.zeros((n, n))
        for g, grad_g, hess_g in zip(self.ineq_constraints, self.ineq_grad_funcs, self.ineq_hess_funcs):
            grad_g_i = grad_g(x).reshape(-1, 1)
            hess_g_i = hess_g(x)
            hessian += (grad_g_i @ grad_g_i.T) / (g(x)**2) - hess_g_i / g(x)
        return hessian

    def __wolfe(self, func, p, x):
        # Find a suitable step size alpha
        """Wolfe condition for line search."""
        alpha = 1.0
        c1 = 0.01
        beta = 0.5
        while True:
            x_new = x + alpha * p
            # Checks of all inequality constraints are satisfied at x_new
            if all(g(x_new) >= 0 for g in self.ineq_constraints):
                f_new = func(x_new)[0]
                f_x = func(x)[0]
                if f_new <= f_x + c1 * alpha * np.dot(self.grad_func(x), p):
                    return alpha
            alpha *= beta

    def optimize_lp(self):
        x = self.x
        t = self.t
        outer_iter = 0
        outer_x_s = []
        outer_obj_values = []

        while outer_iter < self.max_iter_outer:
            f_x, g_x = self.func(x)[:2]
            print(f"Outer iteration {outer_iter}: x = {x}, f(x) = {f_x}")
            f_x_phi = self._phi(x)
            g_x_phi = self._grad_phi(x)

            if self.hess_func is not None:
                h_x = self.hess_func(x)
                h_x_phi = self._hess_phi(x)

                if self.eq_constraints_mat.size:
                    upper_block = np.concatenate([h_x, self.eq_constraints_mat.T], axis=1)
                    size_zeros = (self.eq_constraints_mat.shape[0], self.eq_constraints_mat.shape[0])
                    lower_block = np.concatenate(
                        [self.eq_constraints_mat, np.zeros(size_zeros)],
                        axis=1,
                    )
                    block_matrix = np.concatenate([upper_block, lower_block], axis=0)
                else:
                    block_matrix = h_x
            else:
                h_x = np.zeros((len(x), len(x)))
                block_matrix = h_x + self._hess_phi(x)

            eq_vec = np.concatenate([-g_x, np.zeros(block_matrix.shape[0] - len(g_x))])
            print(f"Block matrix: \n{block_matrix}")
            print(f"Equality vector: {eq_vec}")

            x_prev = x
            f_prev = f_x

            inner_iter = 0
            while inner_iter < self.max_iter_inner:
                if inner_iter != 0 and sum(abs(x - x_prev)) < self.param_tol:
                    print(f"Convergence check (param_tol): Iteration {inner_iter}, change = {sum(abs(x - x_prev))}")
                    break

                try:
                    p = np.linalg.solve(block_matrix, eq_vec)[: len(x)]
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered, applying regularization.")
                    epsilon = 1e-6
                    reg_block_matrix = block_matrix + epsilon * np.eye(block_matrix.shape[0])
                    p = np.linalg.lstsq(reg_block_matrix, eq_vec, rcond=None)[0][: len(x)]

                _lambda = np.matmul(p.transpose(), np.matmul(block_matrix, p)) ** 0.5
                print(f"Inner iteration {inner_iter}: _lambda = {_lambda}")
                if 0.5 * (_lambda**2) < self.obj_tol:
                    print(f"Convergence check (obj_tol): Iteration {inner_iter}, _lambda = {_lambda}")
                    break

                if inner_iter != 0 and (f_prev - f_x < self.obj_tol):
                    print(f"Convergence check (obj change): Iteration {inner_iter}, f_prev - f_x = {f_prev - f_x}")
                    break

                if self.step_len == "wolfe":
                    alpha = self.__wolfe(self.func, p, x)
                else:
                    alpha = self.step_len

                x_prev = x
                f_prev = f_x

                x = x + alpha * p
                f_x, g_x = self.func(x)[:2]
                f_x_phi = self._phi(x)
                g_x_phi = self._grad_phi(x)

                self.path.append(x.copy())
                self.obj_values.append(f_x)

                f_x = t * f_x + f_x_phi
                g_x = t * g_x + g_x_phi

                inner_iter += 1
                print(f"Inner iteration {inner_iter}: x = {x}, f(x) = {f_x}, alpha = {alpha}")

            outer_x_s.append(x)
            outer_obj_values.append((f_x - f_x_phi) / t)

            if len(self.ineq_constraints) / t < self.epsilon:
                print("Convergence achieved.")
                return self.path, self.obj_values, outer_x_s, outer_obj_values

            t = self.mu * t
            outer_iter += 1

        print("Maximum outer iterations reached.")
        return self.path, self.obj_values, outer_x_s, outer_obj_values
    
    
    def optimize_qp(self):
        x = self.x
        t = self.t
        outer_iter = 0
        outer_x_s = []
        outer_obj_values = []

        while outer_iter < self.max_iter_outer:
            f_x, g_x, h_x = self.func(x, True)
            f_x_phi = self._phi(x)
            g_x_phi = self._grad_phi(x)
            h_x_phi = self._hess_phi(x)

            # Form block matrix for KKT conditions
            if self.eq_constraints_mat.size:
                upper_block = np.concatenate([h_x, self.eq_constraints_mat.T], axis=1)
                size_zeros = (self.eq_constraints_mat.shape[0], self.eq_constraints_mat.shape[0])
                lower_block = np.concatenate(
                    [self.eq_constraints_mat, np.zeros(size_zeros)],
                    axis=1,
                )
                block_matrix = np.concatenate([upper_block, lower_block], axis=0)
            else:
                block_matrix = h_x
            eq_vec = np.concatenate([-g_x, np.zeros(block_matrix.shape[0] - len(g_x))])

            x_prev = x
            f_prev = f_x

            inner_iter = 0
            while inner_iter < self.max_iter_inner:
                if inner_iter != 0 and sum(abs(x - x_prev)) < self.param_tol:
                    break

                p = np.linalg.solve(block_matrix, eq_vec)[: len(x)]
                _lambda = np.matmul(p.transpose(), np.matmul(h_x, p)) ** 0.5
                if 0.5 * (_lambda**2) < self.obj_tol:
                    break

                if inner_iter != 0 and (f_prev - f_x < self.obj_tol):
                    break

                if self.step_len == "wolfe":
                    alpha = self.__wolfe(self.func, p, x)
                else:
                    alpha = self.step_len

                x_prev = x
                f_prev = f_x

                x = x + alpha * p
                f_x, g_x, h_x = self.func(x, True)
                f_x_phi = self._phi(x)
                g_x_phi = self._grad_phi(x)
                h_x_phi = self._hess_phi(x)

                self.path.append(x.copy())
                self.obj_values.append(f_x)

                f_x = t * f_x + f_x_phi
                g_x = t * g_x + g_x_phi
                h_x = t * h_x + h_x_phi

                inner_iter += 1

            outer_x_s.append(x)
            outer_obj_values.append((f_x - f_x_phi) / t)

            if len(self.ineq_constraints) / t < self.epsilon:
                return self.path, self.obj_values, outer_x_s, outer_obj_values

            t = self.mu * t
            outer_iter += 1

        return self.path, self.obj_values, outer_x_s, outer_obj_values
    
    
    def optimize(self):
        x = self.x
        t = self.t
        outer_iter = 0
        outer_x_s = []
        outer_obj_values = []

        while outer_iter < self.max_iter_outer:
            # Checks if hessian function is provided
            if self.hess_func:
                # If available, calculate the objective function, gradient and Hessian at 'x'
                f_x, g_x, h_x = self.func(x, return_hessian=True)
                # Calculate the Hessian of the barrier term 
                h_x_phi = self._hess_phi(x)
            else:
                # Hessian is not available, calculate the objective function and gradient at 'x'
                f_x, g_x = self.func(x)[:2]
                h_x = np.zeros((len(x), len(x))) # Initialize Hessian as a zero matrix
                h_x_phi = self._hess_phi(x) # Calculate the Hessian of the barrier term 

            f_x_phi = self._phi(x) # Calculate the barrier function value 
            g_x_phi = self._grad_phi(x) # Calculate the gradient of the barrier function

            # Construct the KKT matrix (block matrix)
            if self.eq_constraints_mat.size:
                upper_block = np.concatenate([h_x, self.eq_constraints_mat.T], axis=1)
                size_zeros = (self.eq_constraints_mat.shape[0], self.eq_constraints_mat.shape[0])
                lower_block = np.concatenate(
                    [self.eq_constraints_mat, np.zeros(size_zeros)],
                    axis=1,
                )
                block_matrix = np.concatenate([upper_block, lower_block], axis=0)
            else:
                block_matrix = h_x + self._hess_phi(x)
                
            # Construct the RHS vector for the KKT system
            eq_vec = np.concatenate([-g_x, np.zeros(block_matrix.shape[0] - len(g_x))])

            x_prev = x
            f_prev = f_x

            inner_iter = 0
            while inner_iter < self.max_iter_inner:
                if inner_iter != 0 and sum(abs(x - x_prev)) < self.param_tol:
                    break
                # Solve the KKT system
                try:
                    p = np.linalg.solve(block_matrix, eq_vec)[: len(x)]
                except np.linalg.LinAlgError:
                    epsilon = 1e-6
                    reg_block_matrix = block_matrix + epsilon * np.eye(block_matrix.shape[0])
                    p = np.linalg.lstsq(reg_block_matrix, eq_vec, rcond=None)[0][: len(x)]

                _lambda = np.sqrt(np.dot(p.transpose(), np.dot(h_x + h_x_phi, p)))
                if 0.5 * (_lambda**2) < self.obj_tol:
                    break

                if inner_iter != 0 and (f_prev - f_x < self.obj_tol):
                    break

                if self.step_len == "wolfe":
                    alpha = self.__wolfe(self.func, p, x)
                else:
                    alpha = self.step_len

                x_prev = x
                f_prev = f_x

                x = x + alpha * p
                if self.hess_func:
                    f_x, g_x, h_x = self.func(x, return_hessian=True)
                else:
                    f_x, g_x = self.func(x)[:2]
                f_x_phi = self._phi(x)
                g_x_phi = self._grad_phi(x)
                h_x_phi = self._hess_phi(x)

                self.path.append(x.copy())
                self.obj_values.append(f_x)

                f_x = t * f_x + f_x_phi
                g_x = t * g_x + g_x_phi
                if self.hess_func:
                    h_x = t * h_x + h_x_phi

                inner_iter += 1

            outer_x_s.append(x)
            outer_obj_values.append((f_x - f_x_phi) / t)

            if len(self.ineq_constraints) / t < self.epsilon:
                return self.path, self.obj_values, outer_x_s, outer_obj_values

            t = self.mu * t
            outer_iter += 1

        return self.path, self.obj_values, outer_x_s, outer_obj_values
