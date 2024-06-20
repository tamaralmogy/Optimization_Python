import unittest
import numpy as np
from src.constrained_min import InteriorPointMinimizer
from tests.examples import qp, lp
from src.utils import plot_qp_result, plot_obj_vs_iter, plot_lp_result
import os

class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        func, eq_constraints_mat, eq_constraints_rhs, ineq_constraints, ineq_grad_funcs, ineq_hess_funcs = qp()
        initial_point = np.array([0.1, 0.2, 0.7])
        minimizer = InteriorPointMinimizer(func, lambda x: func(x)[1], lambda x: func(x, True)[2], eq_constraints_mat, eq_constraints_rhs, ineq_constraints, ineq_grad_funcs, ineq_hess_funcs, initial_point)
        path, obj_values, outer_x_s, outer_obj_values = minimizer.optimize()
        result = path[-1]
        # Check the inequality constraints
        self.assertTrue(all([g(result) >= 0 for g in ineq_constraints]))
        # Print the point of convergence and the objective value
        x, y, z = result
        objective_value = func(result)[0]
        print(f"Point of convergence: x={x}, y={y}, z={z}")
        print(f"x + y + z = {x + y + z}")
        print(f"Objective value at point of convergence: {objective_value}")
        # Print the values to verify inequality constraints
        print(f"-x = {-x}, -y = {-y}, -z = {-z}")
        print(f"Inequality constraints satisfied: {all([g(result) >= 0 for g in ineq_constraints])}")
        
        # Ensure 'plots' directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Plot the results
        plot_qp_result(result, path, filename='plots/qp_path_final_point.png')
        plot_obj_vs_iter(outer_obj_values, filename='plots/qp_objective_vs_iter.png')
        
        
    def test_lp(self):
        func, eq_constraints_mat, eq_constraints_rhs, ineq_constraints, ineq_grad_funcs, ineq_hess_funcs = lp()
        initial_point = np.array([0.5, 0.75])

        minimizer = InteriorPointMinimizer(func, lambda x: func(x)[1], None, eq_constraints_mat, eq_constraints_rhs, ineq_constraints, ineq_grad_funcs, ineq_hess_funcs, initial_point)
        path, obj_values, outer_x_s, outer_obj_values = minimizer.optimize()

        if not path:
            print("Optimization did not produce any valid points.")
            return

        result = path[-1]

        # Check the inequality constraints
        constraints_satisfied = all([g(result) >= 0 for g in ineq_constraints])
        print(f"Inequality constraints satisfied: {constraints_satisfied}")

        # Print the point of convergence and the objective value
        x, y = result
        objective_value = func(result)[0]
        print(f"Point of convergence: x={x}, y={y}")
        print(f"Objective value at point of convergence: {objective_value}")

        # Print constraint values
        constraint_values = [g(result) for g in ineq_constraints]
        for i, val in enumerate(constraint_values):
            print(f"Constraint {i+1} value at point of convergence: {val}")

        # Ensure 'plots' directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')

        # Plot the results
        plot_lp_result(result, path, filename='plots/lp_path_final_point.png')
        plot_obj_vs_iter(outer_obj_values, filename='plots/lp_objective_vs_iter.png')
        
if __name__ == '__main__':
    unittest.main()
