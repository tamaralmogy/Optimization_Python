import numpy as np

class LineSearchOptimizer:
    def __init__(self, func, x0, obj_tol, param_tol, max_iter):
        self.func = func
        self.x = np.array(x0, dtype=float)
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.path = []
        self.values = []
        
    def optimize(self, method='gradient_descent'):
        if method == 'gradient_descent':
            return self.gradient_descent()
        elif method == 'newton':
            return self.newton()
        else:
            raise ValueError("Unsupported optimization method.")


    def wolfe_condition(self, x, p, g, alpha_init=1.0, c1=0.01, beta=0.5, max_iter=20):
        alpha = alpha_init
        f_x = self.func(x)[0]
        for _ in range(max_iter + 1):
            x_new = x + alpha * p
            f_x_new, g_new = self.func(x_new)[:2]

            if f_x_new > f_x + c1 * alpha * np.dot(g, p):
                alpha *= beta
            else:
                return alpha
        return alpha

    def gradient_descent(self):
        # Flag to indicate whether we successfully converged
        success = False
        for i in range(self.max_iter):
            # Calculate f(x) and gradient at the current point x
            f_x, g, _ = self.func(self.x)[:3]
            self.path.append(self.x.copy())
            self.values.append(f_x)

            # Define search direction and step size
            p = -g
            alpha = self.wolfe_condition(self.x, p, g)
            
            # Calculate the new point x_new
            x_new = self.x + alpha * p
            f_x_new, g_new, _ = self.func(x_new)[:3]

            print(f"Iteration {i}: x = {x_new}, f(x) = {f_x_new}, grad = {np.linalg.norm(g)}, alpha = {alpha}")
            
            # Check for convergence - Gradient norm check & objective function change check 
            if np.linalg.norm(g) < self.param_tol or abs(f_x_new - f_x) < self.obj_tol:
                self.x = x_new
                success = True
                break
            
            # Update current point to be the new point
            self.x = x_new

        print(f"Final iteration report: x = {self.x}, f(x) = {f_x_new}, grad = {g_new}, alpha = {alpha}, success = {success}")
        
        return self.x, f_x_new


    def newton(self):
        success = False
        for i in range(self.max_iter):
            # Compute f(x) the gradient and the Hessian at the current point x
            f_x, g, H = self.func(self.x)[:3]
            self.path.append(self.x.copy())
            self.values.append(f_x)

            # Define search direction and step size
            p = -np.linalg.solve(H, g)
            alpha = self.wolfe_condition(self.x, p, g)
            
            # Update new point according to search direction and step size
            x_new = self.x + alpha * p
            f_x_new, g_new, _ = self.func(x_new)[:3]

            print(f"Iteration {i}: x = {x_new}, f(x) = {f_x_new}, grad = {np.linalg.norm(g)}, alpha = {alpha}")
            
            # Check convergence 
            if np.linalg.norm(g) < self.param_tol or abs(f_x_new - f_x) < self.obj_tol:
                self.x = x_new
                success = True
                break
            
            # Update new point to be the current point
            self.x = x_new
        
        #  Print final iteration report 
        print(f"Final iteration report: x = {self.x}, f(x) = {f_x_new}, grad = {g_new}, alpha = {alpha}, success = {success}")
        return self.x, f_x_new


    
