import unittest
import numpy as np
from tests.examples import quadratic_function, Q_i, Q_ii, Q_iii, rosenbrock, linear_function, exponential_function
from src.unconstrained_min import LineSearchOptimizer
from src.utils import plot_contours, plot_function_values
import os

# Create 'plots' directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

class TestOptimizationMethods(unittest.TestCase):
    def setUp(self):
        self.initial_guess_default = np.array([1.0, 1.0])
        self.initial_guess_rosenbrock = np.array([-1, 2])
        self.obj_tol = 1e-8
        self.param_tol = 1e-12
        self.max_iter_default = 100
        self.max_iter_rosenbrock = 10000
        self.a_linear = np.array([1, 1])

    def run_test(self, func, param, test_name, x0, max_iter):
        # Run Gradient Descent
        optimizer_gd = LineSearchOptimizer(
            lambda x: func(x, param, hessian_needed=False) if param is not None else func(x, hessian_needed=False),
            x0,
            self.obj_tol,
            self.param_tol,
            max_iter if func is rosenbrock else self.max_iter_default
        )
        final_x_gd, final_f_gd = optimizer_gd.optimize(method='gradient_descent')

        if func is not linear_function:
            # Run Newton's Method
            optimizer_newton = LineSearchOptimizer(
                lambda x: func(x, param, hessian_needed=True) if param is not None else func(x, hessian_needed=True),
                x0,
                self.obj_tol,
                self.param_tol,
                self.max_iter_default
            )
            final_x_newton, final_f_newton = optimizer_newton.optimize(method='newton')

            # Plot contours with both paths
            plot_contours(
                lambda x: func(x, param, hessian_needed=False) if param is not None else func(x, hessian_needed=False),
                [-2, 2], [-2, 2], 
                title=f"{test_name} Contour with Gradient Descent and Newton Paths",
                paths=[np.array(optimizer_gd.path), np.array(optimizer_newton.path)], 
                path_labels=["Gradient Descent Path", "Newton's Method Path"],
                filename=f'plots/{test_name}_gd_newton_contour.png'
            )

            # Plot function values over iterations for both methods
            plot_function_values(
                [
                    {'values': optimizer_gd.values, 'name': 'Gradient Descent'},
                    {'values': optimizer_newton.values, 'name': "Newton's Method"}
                ],
                f'Function Value over Iterations for Gradient Descent and Newton ({test_name})',
                filename=f'plots/{test_name}_gd_newton_function_values.png'
            )
        else:
            # Plot contours with only Gradient Descent path
            plot_contours(
                lambda x: func(x, param, hessian_needed=False) if param is not None else func(x, hessian_needed=False),
                [-100, 0], [-100, 0], 
                title=f"{test_name} Contour with Gradient Descent Path",
                paths=[np.array(optimizer_gd.path)], 
                path_labels=["Gradient Descent Path"],
                filename=f'plots/{test_name}_gd_contour.png'
            )

            # Plot function values over iterations for Gradient Descent
            plot_function_values(
                [{'values': optimizer_gd.values, 'name': 'Gradient Descent'}],
                f'Function Value over Iterations for Gradient Descent ({test_name})',
                filename=f'plots/{test_name}_gd_function_values.png'
            )

        # Check results for Gradient Descent
        if func is not linear_function:
            self.assertTrue(np.linalg.norm(final_x_gd) < self.param_tol, f"Gradient Descent did not converge to the minimum for {test_name}.")
            self.assertAlmostEqual(final_f_gd, 0, places=10, msg=f"Gradient Descent did not minimize the function value effectively for {test_name}.")
            self.assertTrue(np.linalg.norm(final_x_newton) < self.param_tol, f"Newton's Method did not converge to the minimum for {test_name}.")
            self.assertAlmostEqual(final_f_newton, 0, places=10, msg=f"Newton's Method did not minimize the function value effectively for {test_name}.")
        else:
            print(f"Final x for Gradient Descent in {test_name}: {final_x_gd}")
            print(f"Final function value for Gradient Descent in {test_name}: {final_f_gd}")


    # def test_quadratic_i(self):
    #     self.run_test(quadratic_function, Q_i, 'Quadratic_i', self.initial_guess_default, self.max_iter_default)

    # def test_quadratic_ii(self):
    #     self.run_test(quadratic_function, Q_ii, 'Quadratic_ii', self.initial_guess_default, self.max_iter_default)

    # def test_quadratic_iii(self):
    #     self.run_test(quadratic_function, Q_iii, 'Quadratic_iii', self.initial_guess_default, self.max_iter_default)

    # def test_rosenbrock(self):
    #     self.run_test(rosenbrock, None, 'Rosenbrock', self.initial_guess_rosenbrock, self.max_iter_rosenbrock)

    # def test_linear_function(self):
    #     self.run_test(linear_function, self.a_linear, 'Linear_Function', self.initial_guess_default, self.max_iter_default)
    
    def test_exponential_function(self):
        self.run_test(exponential_function, None, 'Exponential_Function', self.initial_guess_default, self.max_iter_default)


if __name__ == '__main__':
    unittest.main()