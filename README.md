Hi there! 

# Numerical Optimization with Python

This repository contains the implementation of various numerical optimization methods using Python. The project includes examples and tests for different functions, such as quadratic functions, the Rosenbrock function, a linear function, and an exponential function.

## Installation
To run the project, ensure you have Python installed. You can install the required dependencies using:

```sh
pip install -r requirements.txt
```
This is how you should run the tests:
```sh
python -m unittest discover -s tests
```

## Optimization Methods
The project includes implementations for the following optimization methods:

# Gradient Descent:
A simple iterative method for unconstrained optimization problems.

# Newton's Method:
An optimization method that uses second-order information (the Hessian) to achieve faster convergence.

Functions Tested:
The following functions are included as examples and are tested using the optimization methods:

Quadratic Functions:
Q_i: Identity matrix
Q_ii: Scaled matrix
Q_iii: Rotated ellipse matrix
Rosenbrock Function:
A non-convex function used as a performance test problem for optimization algorithms.

Linear Function:
A simple linear function to test basic optimization capabilities.

Exponential Function:
A complex function to test the robustness of the optimization methods.
